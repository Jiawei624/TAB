"""ADD semantics and a small CPU-only TAB evaluation/reporting smoke test."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from ts_benchmark.data.data_pool import DataPool
from ts_benchmark.evaluation.evaluate_model import eval_model
from ts_benchmark.evaluation.evaluator import Evaluator
from ts_benchmark.evaluation.metrics import CLASSIFICATION_METRICS_SCORE, METRICS
from ts_benchmark.evaluation.metrics.classification_metrics_label import add
from ts_benchmark.evaluation.strategy.constants import FieldNames
from ts_benchmark.models import ModelFactory
from ts_benchmark.recording import load_record_data, write_record_file
from ts_benchmark.report import report_csv
from ts_benchmark.report.utils.leaderboard import get_leaderboard
from ts_benchmark.utils.parallel import ParallelBackend


ROOT = Path(__file__).resolve().parents[1]


class AddMetricTests(unittest.TestCase):
    def test_known_delays(self):
        cases = [
            # Immediate hit, late hit, and a missed span.
            ([0, 1, 1, 0], [0, 1, 0, 0], 0.0),
            ([0, 1, 1, 1, 0], [0, 0, 0, 1, 0], 2.0),
            ([0, 1, 1, 0], [0, 0, 0, 0], 2.0),
            # Spans at the boundaries, including isolated anomaly points.
            ([1, 1, 1], [0, 0, 1], 2.0),
            ([1, 1, 1], [0, 0, 0], 3.0),
            ([1, 0, 1], [0, 0, 1], 0.5),
            ([1], [1], 0.0),
            ([1], [0], 1.0),
            # Equal weight per span, rather than per anomalous point.
            ([1, 0, 1, 1, 1, 1], [1, 0, 0, 0, 0, 1], 1.5),
            # False alarms outside a span do not count as detecting it.
            ([0, 1, 1, 0], [1, 0, 0, 1], 2.0),
            # The documented example: one late hit and one missed span.
            ([0, 1, 1, 1, 0, 1, 1], [0, 0, 1, 0, 0, 0, 0], 1.5),
        ]
        for actual, predicted, expected in cases:
            with self.subTest(actual=actual, predicted=predicted):
                self.assertEqual(add(actual, predicted), expected)

    def test_no_anomalies_are_undefined(self):
        for actual, predicted in [([], []), ([0, 0], [0, 0]), ([0, 0], [1, 1])]:
            with self.subTest(actual=actual, predicted=predicted):
                self.assertTrue(np.isnan(add(actual, predicted)))

    def test_binary_input_forms(self):
        for dtype in (bool, int, float):
            for column_actual in (False, True):
                for column_predicted in (False, True):
                    actual = np.array([0, 1, 1, 0], dtype=dtype)
                    predicted = np.array([0, 0, 1, 0], dtype=dtype)
                    if column_actual:
                        actual = actual[:, None]
                    if column_predicted:
                        predicted = predicted[:, None]
                    self.assertEqual(add(actual, predicted), 1.0)

    def test_invalid_inputs_are_rejected(self):
        cases = [
            ([0, 1], [1]),
            ([0, 2], [0, 1]),
            ([0, 1], [0.1, 0.9]),
            ([0, 1], [0, np.nan]),
            ([0, np.inf], [0, 1]),
            ([0, 1], [-1, 1]),
            (1, 1),
            ([[0, 1], [1, 0]], [[0, 1], [1, 0]]),
            (np.zeros((2, 1, 1)), np.zeros((2, 1, 1))),
        ]
        for actual, predicted in cases:
            with self.subTest(actual=actual, predicted=predicted):
                with self.assertRaises(ValueError):
                    add(actual, predicted)

    def test_evaluator_uses_raw_predictions_without_mutation(self):
        actual = np.array([0, 1, 1, 1, 0])
        predicted = np.array([0, 0, 0, 1, 0])
        for names in (["adjust_f_score", "add"], ["add", "adjust_f_score"]):
            evaluator = Evaluator([{"name": name} for name in names])
            values, log = evaluator.evaluate_with_log(actual, predicted)
            self.assertEqual(log, "")
            self.assertEqual(dict(zip(names, values)), {"adjust_f_score": 1.0, "add": 2.0})
        np.testing.assert_array_equal(actual, [0, 1, 1, 1, 0])
        np.testing.assert_array_equal(predicted, [0, 0, 0, 1, 0])

    def test_invalid_input_is_logged_without_losing_other_metrics(self):
        evaluator = Evaluator([{"name": "accuracy"}, {"name": "add"}])
        values, log = evaluator.evaluate_with_log(np.array([0, 2]), np.array([0, 2]))
        self.assertEqual(values[0], 1.0)
        self.assertTrue(np.isnan(values[1]))
        self.assertIn("Error in calculating add", log)
        self.assertIn("binary labels", log)


class ThresholdDetector:
    """A deterministic detector to exercise the real pipeline without training."""

    def __init__(self, multiple_ratios=False, truncate=0):
        self.multiple_ratios = multiple_ratios
        self.truncate = truncate
        self.fitted = False

    def detect_fit(self, train_data, train_label):
        self.fitted = True

    def detect_label(self, test_data):
        if not self.fitted:
            raise RuntimeError("detect_fit must run before detect_label")
        scores = test_data["value"].to_numpy()
        if self.truncate:
            scores = scores[:-self.truncate]
        predictions = {20: (scores > 0.5).astype(int), 80: (scores > 0.1).astype(int)}
        return (predictions if self.multiple_ratios else predictions[20]), scores


class AddPipelineTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            "value": [0, 0, 0, 0, 0.2, 0.8, 0.3, 0, 0.2, 0.3, 0],
            "label": [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0],
        })
        self.pool = Mock()
        self.pool.get_series.return_value = self.data
        self.pool.get_series_meta_info.return_value = pd.Series({"train_lens": 3})
        pool_patch = patch.object(DataPool(), "pool", self.pool)
        pool_patch.start()
        self.addCleanup(pool_patch.stop)
        backend = ParallelBackend()
        backend.init(backend="sequential", n_cpus=1, gpu_devices=[])
        self.addCleanup(backend.close)

    def evaluate(self, evaluation_config, **model_params):
        factory = ModelFactory("ThresholdDetector", ThresholdDetector, model_params)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = eval_model(factory, ["synthetic.csv"], evaluation_config)
            return pd.concat(list(result.collect()), ignore_index=True)

    def test_default_label_configs_include_add_in_records_and_reports(self):
        for path in sorted((ROOT / "config").glob("*detect_label*_config.json")):
            with self.subTest(config=path.name):
                config = json.loads(path.read_text(encoding="utf-8"))
                self.pool.get_series.return_value = (
                    self.data if config["data_config"]["feature_dict"]["if_univariate"]
                    else self.data.assign(second_sensor=np.arange(len(self.data)))
                )
                records = self.evaluate(config["evaluation_config"])
                self.assertEqual(len(records), 1)
                self.assertEqual(records.loc[0, "log_info"], "")
                self.assertEqual(records.loc[0, "add"], 1.5)
                leaderboard = get_leaderboard(
                    records, config["report_config"]["report_metrics"], "mean", "mean_value", 0.3
                )
                self.assertIn("add", leaderboard["metric_name"].tolist())
                add_row = leaderboard.set_index("metric_name").loc["add"]
                self.assertEqual(add_row.iloc[0], 1.5)

    def test_explicit_metrics_and_multiple_ratios(self):
        for metrics in ("add", ["add"], {"name": "add"}, [{"name": "add"}]):
            with self.subTest(metrics=metrics):
                records = self.evaluate({
                    "metrics": metrics,
                    "strategy_args": {"strategy_name": "unfixed_detect_label", "seed": 42},
                }, multiple_ratios=True)
                self.assertEqual(records[FieldNames.ANOMALY_RATIO].tolist(), [20, 80])
                self.assertEqual(records["add"].tolist(), [1.5, 0.0])
                self.assertEqual(records["log_info"].tolist(), ["", ""])

    def test_existing_tail_padding_counts_as_undetected(self):
        self.data.loc[9, "value"] = 0.8
        config = {
            "metrics": ["add"],
            "strategy_args": {"strategy_name": "unfixed_detect_label", "seed": 42},
        }
        self.assertEqual(self.evaluate(config).loc[0, "add"], 1.0)
        self.assertEqual(self.evaluate(config, truncate=3).loc[0, "add"], 1.5)

    def test_no_anomalies_produce_nan_without_error(self):
        self.data["label"] = 0
        records = self.evaluate({
            "metrics": ["add"],
            "strategy_args": {"strategy_name": "unfixed_detect_label", "seed": 42},
        })
        self.assertTrue(np.isnan(records.loc[0, "add"]))
        self.assertEqual(records.loc[0, "log_info"], "")
        leaderboard = get_leaderboard(records, ["add"], "mean", "mean_value", 0.3)
        self.assertTrue(np.isnan(leaderboard.loc[0, "ThresholdDetector;{}"]))

    def test_score_strategies_reject_add(self):
        self.assertIs(METRICS["add"], add)
        self.assertNotIn("add", CLASSIFICATION_METRICS_SCORE)
        for strategy in ("all_detect_score", "fixed_detect_score", "unfixed_detect_score"):
            with self.subTest(strategy=strategy), self.assertRaisesRegex(RuntimeError, "add"):
                self.evaluate({"metrics": ["add"], "strategy_args": {"strategy_name": strategy}})

    def test_compressed_records_and_csv_report(self):
        config = json.loads((ROOT / "config" / "unfixed_detect_label_config.json").read_text())
        records = self.evaluate(config["evaluation_config"])
        with tempfile.TemporaryDirectory() as directory:
            record_path = write_record_file(records, str(Path(directory) / "records.csv"), "gz")
            loaded = load_record_data([record_path])
            self.assertEqual(loaded.loc[0, "add"], 1.5)
            report_config = dict(config["report_config"], log_files_list=[record_path],
                                 save_path=directory, leaderboard_file_name="report.csv")
            report_csv.report(report_config)
            report = pd.read_csv(Path(directory) / "report.csv").set_index("metric_name")
            self.assertEqual(report.loc["add", "ThresholdDetector;{}"], 1.5)


if __name__ == "__main__":
    unittest.main()
