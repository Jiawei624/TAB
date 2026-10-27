"""Regression coverage for data loading, metrics, errors, reporting and CLI."""

import argparse
import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch

from scripts.run_benchmark import build_data_config, build_evaluation_config, build_model_config, build_report_config
from ts_benchmark.baselines.self_impl import LOF
from ts_benchmark.baselines.utils import anomaly_detection_data_provider, SegLoader
from ts_benchmark.data.data_pool import DataPool
from ts_benchmark.data.utils import is_st, load_series_info, process_data_df
from ts_benchmark.evaluation.evaluate_model import EvalResult, eval_model
from ts_benchmark.evaluation.evaluator import Evaluator
from ts_benchmark.evaluation.metrics import CLASSIFICATION_METRICS_LABEL
from ts_benchmark.evaluation.metrics.vus_metrics import metricor
from ts_benchmark.evaluation.strategy import STRATEGY
from ts_benchmark.evaluation.strategy.anomaly_detect import UnFixedDetectLabel
from ts_benchmark.models import ModelFactory
from ts_benchmark.models.model_loader import get_models
from ts_benchmark.pipeline import filter_data, pipeline
from ts_benchmark.recording import load_record_data
from ts_benchmark.report import report_csv
from ts_benchmark.report.utils.leaderboard import get_leaderboard
from ts_benchmark.utils.compress import compress, decompress, get_compress_file_ext
from ts_benchmark.utils.parallel import ParallelBackend
from ts_benchmark.utils.parallel.sequential_backend import SequentialBackend


ROOT = Path(__file__).resolve().parents[1]


class DataTests(unittest.TestCase):
    def test_interleaved_data_and_label_in_first_position(self):
        raw = pd.DataFrame({
            "date": [1, 1, 1, 2, 2, 2],
            "data": [0, 10, 20, 1, 11, 21],
            "cols": ["label", "a", "b"] * 2,
        })
        result = process_data_df(raw)
        self.assertEqual(result.columns.tolist(), ["label", "a", "b"])
        np.testing.assert_array_equal(result.to_numpy(), [[0, 10, 20], [1, 11, 21]])
        self.assertEqual(len(process_data_df(raw, nrows=1)), 1)
        np.testing.assert_array_equal(process_data_df(raw.drop(columns="date")), result)
        self.assertFalse(is_st(raw.assign(extra=0)))
        self.assertTrue(is_st(raw.assign(id=0)))

    def test_rejects_unequal_columns_and_misaligned_timestamps(self):
        raw = pd.DataFrame({"date": [1, 2, 1, 2], "data": [1, 2, 3, 4],
                            "cols": ["a", "a", "label", "label"]})
        with self.assertRaisesRegex(ValueError, "same number"):
            process_data_df(raw.iloc[:-1])
        raw.loc[3, "date"] = 3
        with self.assertRaisesRegex(ValueError, "same timestamps"):
            process_data_df(raw)

    def test_metadata_counts_features_without_label(self):
        raw = pd.DataFrame({"data": [1, 2, 0, 1], "cols": ["a", "a", "label", "label"]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.csv"
            raw.to_csv(path, index=False)
            metadata = load_series_info(str(path))
        self.assertTrue(metadata["if_univariate"])
        self.assertEqual(metadata["length"], 2)

    def test_no_feature_filter_and_short_window_errors(self):
        meta = pd.DataFrame({"file_name": ["a", "b"], "size": ["large", "small"]})
        self.assertEqual(filter_data(meta, ["large"]), ["a"])
        self.assertEqual(len(SegLoader(np.zeros((2, 1)), 4, 1)), 0)
        with self.assertRaisesRegex(ValueError, "No complete windows"):
            anomaly_detection_data_provider(np.zeros((2, 1)), 2, win_size=4)


class MetricTests(unittest.TestCase):
    def test_single_class_boundaries_for_all_label_metrics(self):
        evaluator = Evaluator([{"name": name} for name in CLASSIFICATION_METRICS_LABEL])
        for actual, predicted, expected_f in [([0, 0], [0, 0], 0), ([1, 1], [1, 1], 1)]:
            values, log = evaluator.evaluate_with_log(np.array(actual), np.array(predicted))
            self.assertEqual(log, "")
            values = dict(zip(evaluator.metric_names, values))
            self.assertEqual(values["f_score"], expected_f)
            self.assertEqual(values["adjust_f_score"], expected_f)
            self.assertEqual(values["rf"], expected_f)
            if expected_f:
                self.assertEqual(values["precision_at_k"], 1)
                self.assertEqual(values["add"], 0)
            else:
                self.assertTrue(np.isnan(values["add"]))

    def test_complete_miss_and_precision_at_k_ties(self):
        actual, predicted = np.array([0, 1, 1, 0]), np.zeros(4)
        self.assertEqual(CLASSIFICATION_METRICS_LABEL["affiliation_f"](actual, predicted), 0)
        self.assertEqual(CLASSIFICATION_METRICS_LABEL["precision_at_k"](actual, actual), 1)
        self.assertEqual(CLASSIFICATION_METRICS_LABEL["precision_at_k"](actual, predicted), 0.5)
        self.assertEqual(metricor().range_convers_new([]), [])
        self.assertEqual(metricor().range_convers_new([0, 0]), [])
        self.assertEqual(metricor().range_convers_new([1, 1, 0, 1]), [(0, 1), (3, 3)])

    def test_parameterized_metric_exception_does_not_abort_other_metrics(self):
        evaluator = Evaluator([{"name": "mase", "seasonality": 1}, {"name": "mae"}])
        values, log = evaluator.evaluate_with_log(np.array([1., 2.]), np.array([2., 3.]))
        self.assertTrue(np.isnan(values[0]))
        self.assertEqual(values[1], 1)
        self.assertIn("Error in calculating mase;seasonality:1", log)
        self.assertEqual(Evaluator([{"name": "add"}]).evaluate(np.array([1]), np.array([0])), [1])


class CliTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "config/unfixed_detect_label_multi_config.json").read_text())

    def args(self, **kwargs):
        defaults = dict(data_name_list=None, data_set_name=None, adapter=None,
                        model_name=["self_impl.AnomalyTransformer"], model_hyper_params=None,
                        metrics=None, strategy_args=None, seed=None, aggregate_type=None, save_path=None)
        return argparse.Namespace(**dict(defaults, **kwargs))

    def test_metrics_all_bare_names_and_json(self):
        for supplied, expected in [(["all"], "all"), (["f_score", "add"], ["f_score", "add"]),
                                   (['{"name":"add"}'], [{"name": "add"}])]:
            result = build_evaluation_config(self.args(metrics=supplied), self.config)
            self.assertEqual(result["metrics"], expected)

    def test_preserves_config_values_and_rejects_mismatched_hyperparameters(self):
        self.config["data_config"]["data_name_list"] = ["saved.csv"]
        self.config["report_config"]["aggregate_type"] = "median"
        self.assertEqual(build_data_config(self.args(), self.config)["data_name_list"], ["saved.csv"])
        self.assertEqual(build_report_config(self.args(), self.config)["aggregate_type"], "median")
        for params in [["{}", "{}"], ["[]"]]:
            with self.assertRaises(ValueError):
                build_model_config(self.args(model_hyper_params=params), self.config)
        model_config = build_model_config(self.args(model_hyper_params=['{"multi_gpu":true}']), self.config)
        self.assertTrue(model_config["models"][0]["model_hyper_params"]["multi_gpu"])
        self.assertEqual(self.config["model_config"]["models"], [])

    def test_all_shipped_configs_are_valid(self):
        for path in (ROOT / "config").glob("*.json"):
            with self.subTest(config=path.name):
                config = json.loads(path.read_text())
                evaluation = config["evaluation_config"]
                strategy = STRATEGY[evaluation["strategy_args"]["strategy_name"]]
                self.assertFalse(set(strategy.get_required_configs()) - set(evaluation["strategy_args"]))

    def test_python_sources_parse(self):
        for directory in [ROOT / "scripts", ROOT / "ts_benchmark", ROOT / "tests"]:
            for path in directory.rglob("*.py"):
                with self.subTest(file=str(path.relative_to(ROOT))):
                    ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))


class FrameworkTests(unittest.TestCase):
    def test_model_loader_is_independent_of_optional_model_libraries(self):
        models = get_models({"models": [{"model_name": "self_impl.AnomalyTransformer"}]})
        self.assertEqual(models[0].model_name, "AnomalyTransformer")
        import ts_benchmark.baselines.self_impl as self_impl
        with patch.object(self_impl, "LOF", ModuleType("LOF")):
            lof = get_models({"models": [{"model_name": "self_impl.LOF"}]})
            self.assertTrue(callable(lof[0].model_factory))

    def test_all_six_detection_protocols_with_real_metrics(self):
        class Detector:
            def detect_fit(self, features, labels):
                if "label" in features:
                    raise AssertionError("Ground truth leaked into features")

            def detect_score(self, features):
                scores = features["value"].to_numpy()
                return scores, scores

            def detect_label(self, features):
                scores, _ = self.detect_score(features)
                return (scores > 0.5).astype(int), scores

        labels = np.array([0] * 10 + [0, 1, 1, 0] * 5)
        values = pd.DataFrame({"value": labels * 0.8 + 0.1, "label": labels})
        pool = Mock()
        pool.get_series.return_value = values
        pool.get_series_meta_info.return_value = pd.Series({"train_lens": 10})
        backend = ParallelBackend()
        with patch.dict("os.environ"), patch.object(DataPool(), "pool", pool):
            backend.init(backend="sequential", n_cpus=1, gpu_devices=[])
            try:
                for mode in ["fixed", "unfixed", "all"]:
                    for kind in ["label", "score"]:
                        with self.subTest(mode=mode, kind=kind):
                            args = {"strategy_name": f"{mode}_detect_{kind}", "seed": 42}
                            if mode == "fixed":
                                args["train_test_split"] = 1 / 3
                            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                                result = eval_model(ModelFactory("synthetic", Detector, {}), ["tiny.csv"],
                                                    {"metrics": "all", "strategy_args": args})
                                records = next(result.collect())
                            self.assertEqual(records.loc[0, "log_info"], "")
                            self.assertEqual(records.loc[0, "file_name"], "tiny.csv")
                            metric = "f_score" if kind == "label" else "auc_roc"
                            self.assertAlmostEqual(records.loc[0, metric], 1)
            finally:
                backend.close()

    def test_fixed_forecast_also_collects_one_complete_row(self):
        class Forecaster:
            def forecast_fit(self, values, train_ratio_in_tv):
                pass

            def forecast(self, horizon, history):
                return np.zeros((horizon, 1))

        pool = Mock()
        pool.get_series.return_value = pd.DataFrame({"value": np.arange(1, 11)})
        pool.get_covariates.return_value = None
        pool.get_series_meta_info.return_value = None
        backend = ParallelBackend()
        with patch.dict("os.environ"), patch.object(DataPool(), "pool", pool):
            backend.init(backend="sequential", n_cpus=1, gpu_devices=[])
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    result = eval_model(ModelFactory("forecast", Forecaster, {}), ["forecast.csv"], {
                        "metrics": ["mae"], "strategy_args": {"strategy_name": "fixed_forecast",
                        "seed": 42, "horizon": 3, "train_ratio_in_tv": 0.8}})
                    records = next(result.collect())
                self.assertEqual(records.loc[0, "log_info"], "")
                self.assertEqual(records.loc[0, "mae"], 9)
            finally:
                backend.close()

    def test_scalar_config_without_default_and_required_fixed_split(self):
        strategy = UnFixedDetectLabel({"strategy_name": "unfixed_detect_label", "seed": {"x": 7}},
                                      Evaluator([{"name": "add"}]))
        self.assertEqual(strategy._get_scalar_config_value("seed", "x"), 7)
        with self.assertRaisesRegex(RuntimeError, "train_test_split"):
            STRATEGY["fixed_detect_label"]({"strategy_name": "fixed_detect_label", "seed": 42},
                                           Evaluator([{"name": "add"}]))

    def test_task_failure_keeps_a_complete_result_row_and_series_name(self):
        strategy = UnFixedDetectLabel({"strategy_name": "unfixed_detect_label", "seed": 42},
                                      Evaluator([{"name": "add"}]))
        failure = Mock()
        failure.result.side_effect = TimeoutError("synthetic timeout")
        result = EvalResult(strategy, [failure], ModelFactory("broken", Mock(), {}), ["failed.csv"])
        with contextlib.redirect_stderr(io.StringIO()):
            frame = next(result.collect())
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.loc[0, "file_name"], "failed.csv")
        self.assertIn("synthetic timeout", frame.loc[0, "log_info"])

    def test_constructor_failure_is_recorded_and_releases_model(self):
        strategy = UnFixedDetectLabel({"strategy_name": "unfixed_detect_label", "seed": 42},
                                      Evaluator([{"name": "add"}]))
        result = strategy.execute("bad.csv", Mock(side_effect=ValueError("invalid config")))
        row = dict(zip(strategy.field_names, result[0]))
        self.assertEqual(row["file_name"], "bad.csv")
        self.assertIn("invalid config", row["log_info"])
        self.assertIsNone(strategy.model)

    def test_sequential_errors_are_collected_and_initializers_run(self):
        initializer = Mock()
        backend = SequentialBackend(gpu_devices=[], worker_initializers=[initializer])
        with patch.dict("os.environ"):
            backend.init()
        initializer.assert_called_once()
        result = backend.schedule(Mock(side_effect=ValueError("failed task")), ())
        with self.assertRaisesRegex(ValueError, "failed task"):
            result.result()

    def test_lof_preserves_multivariate_rows_and_returns_binary_labels(self):
        values = pd.DataFrame(np.random.default_rng(4).normal(size=(25, 3)))
        detector = LOF(n_neighbors=5, contamination=0.2)
        scores, _ = detector.detect_score(values)
        labels, _ = detector.detect_label(values)
        self.assertEqual(scores.shape, (25,))
        self.assertEqual(labels.shape, (25,))
        self.assertTrue(np.isin(labels, [0, 1]).all())

    def test_compression_round_trip_and_unsupported_formats(self):
        content = {"records.csv": "metric,value\nadd,1.5\n", "name": "测试"}
        self.assertEqual(decompress(compress(content)), content)
        with self.assertRaises(NotImplementedError):
            compress(content, "zip")
        with self.assertRaises(NotImplementedError):
            get_compress_file_ext("zip")

    def test_report_does_not_average_different_ratios_or_ignore_failed_series(self):
        records = pd.DataFrame({
            "model_name": ["A"] * 3, "model_params": ["{}"] * 3,
            "strategy_args": ["{}"] * 3, "file_name": ["ok", "ok", "failed"],
            "typical_anomaly_ratio": [1., 5., np.nan], "add": [2., 8., np.nan],
        })
        result = get_leaderboard(records, ["add"], "mean", "mean_value", 0.6)
        self.assertEqual(result.loc[0, "A;{};anomaly_ratio=1"], 2)
        self.assertEqual(result.loc[0, "A;{};anomaly_ratio=5"], 8)
        strict = get_leaderboard(records, ["add"], "mean", "mean_value", 0.3)
        self.assertTrue(strict.drop(columns="metric_name").isna().all().all())

    def test_real_file_to_training_to_compressed_records_to_csv(self):
        previous_threads = torch.get_num_threads()
        previous_pool = DataPool().pool
        torch.set_num_threads(2)
        try:
            with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ"), \
                    patch.object(torch.cuda, "is_available", return_value=False):
                root = Path(directory)
                (root / "data").mkdir()
                t = np.arange(45)
                values = pd.DataFrame({"a": np.sin(t / 3), "b": np.cos(t / 5),
                                       "label": ((t >= 34) & (t <= 38)).astype(int)})
                raw = pd.concat([pd.DataFrame({"date": t, "data": values[col], "cols": col})
                                 for col in values], ignore_index=True)
                raw.to_csv(root / "data/tiny.csv", index=False)
                pd.DataFrame({"file_name": ["tiny.csv"], "train_lens": [32], "size": ["large"]}) \
                    .to_csv(root / "DETECT_META.csv", index=False)
                config = json.loads((ROOT / "config/unfixed_detect_label_multi_config.json").read_text())
                model_config = {"models": [{"model_name": "self_impl.AnomalyTransformer",
                    "model_hyper_params": dict(win_size=4, d_model=8, n_heads=2, d_ff=8,
                                               e_layers=1, batch_size=7, num_epochs=1,
                                               anomaly_ratio=[10, 30])}]}
                backend = ParallelBackend()
                backend.init(backend="sequential", n_cpus=1, gpu_devices=[])
                try:
                    with patch("ts_benchmark.data.data_source.ANOMALY_DETECT_DATASET_PATH", directory), \
                            contextlib.redirect_stderr(io.StringIO()):
                        paths = pipeline({"data_name_list": ["tiny.csv", "tiny.csv"]}, model_config,
                                         config["evaluation_config"], str(root / "output"))
                finally:
                    backend.close()
                self.assertEqual(len(paths), 1)
                records = load_record_data(paths)
                self.assertEqual(len(records), 2)
                self.assertEqual(records["file_name"].tolist(), ["tiny.csv"] * 2)
                self.assertTrue(records["add"].notna().all())
                self.assertTrue(records["log_info"].fillna("").eq("").all(), records["log_info"].tolist())
                report_csv.report(dict(config["report_config"], log_files_list=records,
                                       save_path=str(root / "report/subdir"), leaderboard_file_name="metrics.csv"))
                report = pd.read_csv(root / "report/subdir/metrics.csv")
                self.assertIn("add", report["metric_name"].tolist())
                self.assertEqual(len([c for c in report if "anomaly_ratio=" in c]), 2)
        finally:
            DataPool().pool = previous_pool
            torch.set_num_threads(previous_threads)


@unittest.skipUnless(importlib.util.find_spec("ray"), "Optional Ray dependency is not installed")
class RayErrorTests(unittest.TestCase):
    def test_user_task_exception_does_not_kill_result_collection(self):
        from ts_benchmark.utils.parallel.ray_backend import RayActorPool, RayResult, RayTask
        pool = RayActorPool.__new__(RayActorPool)
        result = RayResult(threading.Event())
        task_object = object()
        pool._ray_task_to_id = {task_object: 0}
        pool._task_info = {0: RayTask(result=result, actor_id=0)}
        pool._actor_tasks, pool._idle_actors = [0], []
        pool.max_tasks_per_child = None
        with patch("ts_benchmark.utils.parallel.ray_backend.ray.get", side_effect=RuntimeError("task failed")):
            pool._handle_ready_tasks([task_object])
        self.assertEqual(pool._idle_actors, [0])
        with self.assertRaisesRegex(RuntimeError, "task failed"):
            result.result()


if __name__ == "__main__":
    unittest.main()
