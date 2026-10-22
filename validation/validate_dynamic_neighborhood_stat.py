"""Critical-path checks only; synthetic runs are not benchmark evidence.

Run from TAB: python validation/validate_dynamic_neighborhood_stat.py
"""

import contextlib
import io
import importlib.util
import json
import logging
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score

from ts_benchmark.baselines.self_impl.dynamic_neighborhood_stat import DynamicNeighborhoodStat
from ts_benchmark.baselines.self_impl.dynamic_neighborhood_stat.DynamicNeighborhoodStat import WindowDataset, window_split_metadata
from ts_benchmark.baselines.self_impl.dynamic_neighborhood_stat.config import Config
from ts_benchmark.baselines.self_impl.dynamic_neighborhood_stat.features import Observations, training_references
from ts_benchmark.baselines.self_impl.dynamic_neighborhood_stat.model import DualPathNetwork
from ts_benchmark.data.data_pool import DataPool
from ts_benchmark.data.data_source import LocalDataSource
from ts_benchmark.evaluation.evaluate_model import eval_model
from ts_benchmark.evaluation.evaluator import Evaluator
from ts_benchmark.evaluation.metrics import classification_metrics_label, classification_metrics_score
from ts_benchmark.evaluation.strategy.anomaly_detect import UnFixedDetectLabel
from ts_benchmark.evaluation.metrics.classification_metrics_label import add
from ts_benchmark.models import ModelFactory
from ts_benchmark.models.model_loader import get_model_info
from ts_benchmark.utils.parallel import ParallelBackend


def load_launcher():
    specification = importlib.util.spec_from_file_location("dns_launcher", ROOT / "scripts" / "run_dynamic_neighborhood_stat.py")
    launcher = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(launcher)
    return launcher


class MechanismChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(12)
        cls.config = Config(dropout=0.0, target_chunk_size=2)
        cls.training = np.random.default_rng(12).normal(size=(180, 4))
        cls.references = training_references(cls.training, cls.config)
        cls.model = DualPathNetwork(4, cls.config, cls.references).eval()

    def test_observations_and_reference_statistics(self):
        windows = torch.from_numpy(np.stack([self.training[start:start + 128] for start in range(53)]))
        coarse = self.model.observations.coarse(windows)
        targets = torch.stack([z[:, :, -1] for z in coarse], dim=2).numpy()
        np.testing.assert_allclose(targets.mean(axis=0), self.references["mean_n"], atol=1e-12)
        np.testing.assert_allclose(targets.var(axis=0), self.references["variance_n"], atol=1e-12)
        # Independent scalar least-squares construction, including fourth-order statistic.
        block = self.training[128 - 16:128, 0]
        design = np.column_stack([np.ones(16), np.linspace(-1, 1, 16)])
        residual = block - design @ np.linalg.lstsq(design, block, rcond=None)[0]
        energy = residual @ residual
        eps = self.references["energy_floor"][0, 0]
        expected = [np.log(energy / 14 + eps), residual[1:] @ residual[:-1] / (energy + 14 * eps),
                    np.power(residual, 4).sum() / (energy + 14 * eps) ** 2]
        actual = self.model.observations.statistics(windows[:1], self.model.energy_floor)[0, 0, 0].numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        line = torch.arange(128, dtype=torch.float64)[None, :, None].expand(1, 128, 4) * 0.25 + 7
        stat = self.model.observations.statistics(line, self.model.energy_floor)
        np.testing.assert_allclose(stat[..., 0].numpy(), np.log(self.references["energy_floor"])[None], atol=1e-12)
        np.testing.assert_allclose(stat[..., 1:].numpy(), 0, atol=1e-12)

    def test_excluded_inputs_have_zero_jacobians_and_invariant_predictions(self):
        windows = torch.from_numpy(np.stack([self.training[:128], self.training[1:129]])).requires_grad_()
        output = self.model(windows)
        target = 1
        gradient = torch.autograd.grad(output["prediction_n"][:, target].sum(), windows, retain_graph=True)[0]
        self.assertEqual(float(gradient[:, :, target].abs().max()), 0.0)
        self.assertGreater(float(gradient[:, :, 0].abs().max()), 0.0)
        changed = windows.detach().clone()
        changed[:, :, target] += 100
        torch.testing.assert_close(output["prediction_n"][:, target], self.model(changed)["prediction_n"][:, target], atol=1e-6, rtol=0)
        for index, span in enumerate(self.config.spans):
            gradient = torch.autograd.grad(output["prediction_s"][:, target, index].sum(), windows, retain_graph=True)[0]
            self.assertEqual(float(gradient[:, -span:, target].abs().max()), 0.0)
            self.assertEqual(float(gradient[:, :, 0].abs().max()), 0.0)
            changed = windows.detach().clone()
            changed[:, -span:, target] -= 100
            torch.testing.assert_close(output["prediction_s"][:, target, index], self.model(changed)["prediction_s"][:, target, index], atol=1e-6, rtol=0)
        gradient = torch.autograd.grad(output["prediction_p"][:, target].sum(), windows)[0]
        self.assertEqual(float(gradient[:, -1, target].abs().max()), 0.0)
        self.assertEqual(float(gradient[:, :, 0].abs().max()), 0.0)
        self.assertGreaterEqual(self.model.encoder_n.receptive_field, 32)
        self.assertGreaterEqual(self.model.encoder_s.receptive_field, 127)

    def test_one_neighbor_constant_targets_and_backward_with_checkpointing(self):
        config = Config(win_size=64, target_chunk_size=1)
        values = np.column_stack([np.zeros(90), np.random.default_rng(1).normal(size=90)])
        references = training_references(values, config)
        references["variance_s"][0] = 0  # explicitly exercise exact zero, not merely a tiny variance
        network = DualPathNetwork(2, config, references).train()
        captured = {}

        def capture_raw(module, inputs, output):
            captured["raw"] = output

        hook = network.stat_head.register_forward_hook(capture_raw)
        windows = torch.from_numpy(values[:64][None])
        output = network(windows)
        hook.remove()
        self.assertEqual(output["regularizer"].item(), 0.0)
        np.testing.assert_allclose(output["prediction_s"][0, 0].detach().numpy(), references["fixed_values_s"][0])
        statistical_error = network.errors(windows, output)[1]
        denominator = network.variance_s.clamp_min(config.epsilon_v)
        normalized = statistical_error / denominator
        self.assertEqual(float(denominator[0].min()), config.epsilon_v)
        self.assertLessEqual(float(normalized[:, 0].max().detach()), 1e-18)
        raw_gradient = torch.autograd.grad(normalized[:, 0].sum(), captured["raw"], retain_graph=True)[0]
        self.assertEqual(float(raw_gradient.abs().max()), 0.0)
        losses = network.losses(windows, output)
        self.assertTrue(all(bool(torch.isfinite(value)) for value in losses.values()))
        losses["loss"].backward()
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in network.parameters() if parameter.grad is not None))

        # Change only the raw statistics head: fixed-coordinate inference must remain identical.
        scorer = DynamicNeighborhoodStat(win_size=64)
        scorer.columns_, scorer.support_ = ["constant", "noise"], np.zeros(2, dtype=bool)
        scorer.q_n_, scorer.q_s_, scorer.q_p_ = np.ones((2, 6, 2)), np.ones((2, 2, 3)), np.ones(2)
        network.eval()
        with torch.no_grad():
            before_output = network(windows)
            before = scorer._score_errors(scorer._pack_errors(network.errors(windows, before_output)))[1]
            for parameter in network.stat_head.parameters():
                parameter.fill_(100)
            after_output = network(windows)
            after = scorer._score_errors(scorer._pack_errors(network.errors(windows, after_output)))[1]
            np.testing.assert_array_equal(before[:, 0], after[:, 0])
            # A changed observation is still scored against the fixed reference, not discarded.
            changed = windows.clone()
            changed[:, -1, 0] = 4
            changed_output = network(changed)
            changed_target = network.observations.statistics(changed, network.energy_floor)
            changed_error = network.errors(changed, changed_output)[1]
            torch.testing.assert_close(changed_error[:, 0], (changed_target[:, 0] - network.fixed_values_s[0]).square())
            self.assertGreater(float(changed_error[:, 0, :, 0].min()), 0)

    def test_freeze_bound_is_conditional_and_targets_remain_unclipped(self):
        config = Config(win_size=64, dropout=0.0)
        normal = np.random.default_rng(42).normal(size=(400, 2))
        with_freezes = normal.copy()
        with_freezes[100:220] = 0
        for name, training, positive in (("positive_margin", normal, True), ("legal_freezes", with_freezes, False)):
            with self.subTest(case=name):
                references = training_references(training, config)
                detector = DynamicNeighborhoodStat(win_size=64)
                detector.references_, detector.columns_ = references, ["a", "b"]
                diagnostics = detector.freeze_energy_diagnostics()
                self.assertTrue((np.asarray(diagnostics["has_positive_bound"]) == positive).all())
                self.assertTrue((~references["fixed_s"][..., 0]).all())
                network = DualPathNetwork(2, config, references).eval()
                with torch.no_grad():
                    # Approach a_low to check the bound at the most favorable prediction.
                    network.stat_head[-1].weight[0].zero_()
                    network.stat_head[-1].bias[0] = -80
                    windows = torch.zeros((1, 64, 2), dtype=torch.float64)
                    output = network(windows)
                    observed = network.observations.statistics(windows, network.energy_floor)
                    error = network.errors(windows, output)[1][0, ..., 0].numpy()
                np.testing.assert_allclose(observed[0, ..., 0].numpy(), np.log(references["energy_floor"]), atol=1e-12)
                bound = np.asarray(diagnostics["energy_squared_error_lower_bound"])
                if positive:
                    self.assertTrue((observed[0, ..., 0].numpy() < references["a_low"]).all())
                    self.assertTrue((error + 1e-10 >= bound).all())
                    self.assertTrue((bound > 0).all())
                else:
                    np.testing.assert_array_equal(bound, 0)
                    np.testing.assert_allclose(error, 0, atol=1e-24)

    def test_independent_window_bounds_and_recorded_score_endpoints(self):
        values = np.repeat(np.arange(400, dtype=float)[:, None], 2, axis=1)
        train, validation = WindowDataset(values[:320], 64), WindowDataset(values[320:], 64)
        np.testing.assert_array_equal(train[len(train) - 1][:, 0], np.arange(256, 320))
        np.testing.assert_array_equal(validation[0][:, 0], np.arange(320, 384))
        self.assertEqual(np.intersect1d(train[len(train) - 1][:, 0], validation[0][:, 0]).size, 0)
        metadata = window_split_metadata(320, 80, 64)
        self.assertEqual(metadata["train"]["window_count"], 257)
        self.assertEqual(metadata["validation"]["window_count"], 17)
        self.assertEqual(metadata["validation"]["first_score_index"], 383)
        self.assertEqual(metadata["validation"]["last_window_end_exclusive"], 400)
        with self.assertRaises(IndexError):
            validation[len(validation)]
        with self.assertRaises(ValueError):
            WindowDataset(values[:63], 64)


class TabLifecycleChecks(unittest.TestCase):
    def test_training_calibration_existing_strategies_and_checkpoint_reuse(self):
        torch.set_num_threads(2)
        runtime = Path(tempfile.mkdtemp(prefix="dns-check-", dir=str(ROOT / "validation"))).resolve()
        old_pool = DataPool().get_pool()
        backend = ParallelBackend()
        backend.init(backend="sequential", n_cpus=1, gpu_devices=[])
        try:
            rng = np.random.default_rng(42)
            values = np.round(rng.normal(size=(530, 3)), 6)
            values[360 + 82:360 + 96, 1] += 5
            data = pd.DataFrame(values, columns=["a", "b", "c"])
            data["label"] = 0
            for begin, end in ((50, 71), (82, 96), (165, 170)):
                data.loc[360 + begin:360 + end - 1, "label"] = 1
            pool = Mock()
            pool.get_series.return_value = data
            pool.get_series_meta_info.return_value = pd.Series({"train_lens": 360})
            DataPool().set_pool(pool)
            model_class = get_model_info({"model_name": "self_impl.DynamicNeighborhoodStat"})
            self.assertIs(model_class, DynamicNeighborhoodStat)
            parameters = dict(win_size=64, num_epochs=1, batch_size=32, device="cpu", artifact_dir=str(runtime / "launcher" / "artifacts"),
                              target_chunk_size=2, reuse_checkpoint=True)
            factory = ModelFactory("DynamicNeighborhoodStat", model_class, parameters)

            def evaluate(mode):
                config = json.loads((ROOT / "config" / ("dynamic_neighborhood_stat_" + mode + ".json")).read_text())
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    result = eval_model(factory, ["synthetic.csv"], config["evaluation_config"])
                    frame = pd.concat(list(result.collect()), ignore_index=True)
                self.assertEqual(frame.loc[0, "log_info"], "", frame.loc[0, "log_info"])
                return frame, result.strategy.model

            # The real TAB strategy disables cuDNN in its seed helper. All of this
            # model's train/validation/calibration/test forwards must override it.
            cudnn_states = []
            original_forward = DualPathNetwork.forward

            def record_runtime(network, windows):
                cudnn_states.append(torch.backends.cudnn.enabled)
                return original_forward(network, windows)

            with patch.object(DualPathNetwork, "forward", record_runtime), \
                 patch.object(DynamicNeighborhoodStat, "_predict", autospec=True,
                              side_effect=DynamicNeighborhoodStat._predict) as predict:
                labels, fitted = evaluate("label")
            self.assertEqual(predict.call_count, 1)  # all ten ratios share one test scoring pass
            ratios = Config().anomaly_ratio
            self.assertEqual(labels["typical_anomaly_ratio"].tolist(), ratios)
            self.assertTrue(set(classification_metrics_label.__all__).issubset(labels.columns))
            self.assertTrue(cudnn_states)
            self.assertTrue(all(cudnn_states))
            self.assertFalse(torch.backends.cudnn.enabled)  # restore TAB's inherited setting
            self.assertEqual(fitted.best_epoch_, 1)
            self.assertEqual(fitted.model.observations.win_size, 64)
            calibration = json.loads((fitted.run_dir_ / "calibration.json").read_text())
            self.assertEqual(calibration["split_metadata"]["validation"]["first_window_start"], 288)
            self.assertEqual(calibration["split_metadata"]["validation"]["first_score_index"], 351)
            self.assertEqual(calibration["split_metadata"]["validation"]["window_count"], 9)
            self.assertEqual(np.asarray(calibration["freeze_energy"]["margin"]).shape, (3, 2))
            thresholds = fitted._label_thresholds()
            np.testing.assert_allclose(list(thresholds.values()),
                                       np.percentile(fitted.validation_scores_, 100 - np.asarray(ratios)))
            self.assertAlmostEqual(thresholds[1.0], fitted.threshold_)
            predictions = pd.read_csv(fitted.run_dir_ / "test_predictions.csv")
            progress = json.loads((fitted.run_dir_ / "runtime_progress.json").read_text())
            self.assertIn("test inference", progress["stage"])
            self.assertEqual(progress["status"], "completed")
            environment = json.loads((fitted.run_dir_ / "runtime_environment.json").read_text())
            self.assertFalse(environment["inherited_cudnn_enabled"])
            self.assertTrue(environment["model_cudnn_enabled"])
            history = pd.read_csv(fitted.run_dir_ / "training_history.csv")
            self.assertGreater(history.loc[0, "train_seconds"], 0)
            self.assertGreater(history.loc[0, "validation_seconds"], 0)
            self.assertTrue(predictions["score"].iloc[:63].isna().all())
            self.assertTrue(np.isfinite(predictions["score"].iloc[63:]).all())
            actual = data["label"].to_numpy()[360:]
            alarm = predictions["alarm"].to_numpy()
            reference = labels.loc[labels["typical_anomaly_ratio"] == 1.0].iloc[0]
            self.assertAlmostEqual(reference["f_score"], f1_score(actual[63:], alarm[63:], zero_division=0))
            hits = np.flatnonzero(alarm[82:96])
            expected_delay = float(hits[0]) if len(hits) else 14.0
            self.assertEqual(reference["add"], expected_delay)
            events = pd.read_csv(fitted.run_dir_ / "test_events.csv")
            self.assertEqual(len(events), 3 * len(ratios))
            self.assertEqual(events.loc[events["typical_anomaly_ratio"] == 1.0, "eligible_for_add"].tolist(), [False, True, False])
            evaluation = json.loads((fitted.run_dir_ / "evaluation_label.json").read_text())
            self.assertEqual([group["typical_anomaly_ratio"] for group in evaluation["groups"]], ratios)
            label_predictions = pd.read_csv(fitted.run_dir_ / "test_labels.csv")
            for ratio, threshold in thresholds.items():
                np.testing.assert_array_equal(label_predictions["alarm_ratio_{}".format(ratio)].to_numpy()[63:],
                                              predictions["score"].to_numpy()[63:] > threshold)
            self.assertEqual(len(predictions), 170)  # last incomplete batch included
            # Missing diagnostic metadata in an older checkpoint must not force retraining.
            payload = torch.load(fitted.checkpoint_path_, map_location="cpu", weights_only=True)
            self.assertIn("split_metadata", payload)
            payload.pop("split_metadata")
            payload.pop("validation_scores")
            payload["config"].pop("anomaly_ratio")
            payload["config"].pop("factorized_routing")
            payload["config"].pop("prune_tcn")
            self.assertFalse(any("readout_plans" in name for name in payload["model"]))
            torch.save(payload, fitted.checkpoint_path_)
            legacy = DynamicNeighborhoodStat.load(fitted.checkpoint_path_, device="cpu")
            with self.assertRaisesRegex(RuntimeError, "normal training data"):
                legacy._label_thresholds()
            fixed = DynamicNeighborhoodStat.load(fitted.checkpoint_path_, device="cpu", anomaly_ratio=None)
            self.assertEqual(fixed._label_thresholds(), {"validation_q0.99": fitted.threshold_})
            calibration.pop("split_metadata")
            calibration.pop("freeze_energy")
            (fitted.run_dir_ / "calibration.json").write_text(json.dumps(calibration), encoding="utf-8")
            # The score strategy must load the SAME normal calibration, never refit it.
            with patch.object(DynamicNeighborhoodStat, "_train", side_effect=AssertionError("unexpected retraining")), \
                 patch.object(DynamicNeighborhoodStat, "_collect_validation_scores", autospec=True,
                              side_effect=DynamicNeighborhoodStat._collect_validation_scores) as collect_validation:
                scores, reused = evaluate("score")
            self.assertEqual(collect_validation.call_count, 1)
            self.assertTrue(set(classification_metrics_score.__all__).issubset(scores.columns))
            self.assertEqual(reused.checkpoint_path_, fitted.checkpoint_path_)
            self.assertEqual(reused.threshold_, fitted.threshold_)
            np.testing.assert_allclose(reused.validation_scores_, fitted.validation_scores_, rtol=1e-12, atol=1e-12)
            for name in ("q_n_", "q_s_", "q_p_", "support_"):
                np.testing.assert_array_equal(getattr(reused, name), getattr(fitted, name))
            for name, value in fitted.model.state_dict().items():
                torch.testing.assert_close(reused.model.state_dict()[name], value, rtol=0, atol=0)
            refreshed = json.loads((reused.run_dir_ / "calibration.json").read_text())
            self.assertEqual(refreshed["split_metadata"]["validation"]["first_score_index"], 351)
            self.assertIn("freeze_energy", refreshed)
            self.assertAlmostEqual(scores.loc[0, "auc_pr"], average_precision_score(actual[63:], predictions["score"].to_numpy()[63:]))
            restored = DynamicNeighborhoodStat.load(fitted.checkpoint_path_, device="cpu")
            np.testing.assert_allclose(list(restored._label_thresholds().values()), list(thresholds.values()), rtol=1e-12)
            restored_scores, _ = restored.detect_score(data.iloc[360:].drop(columns="label"))
            np.testing.assert_allclose(restored_scores, predictions["score"].to_numpy(), rtol=1e-12, atol=1e-12, equal_nan=True)
            with self.assertRaisesRegex(ValueError, "clean normal"):
                fitted.detect_fit(data.iloc[:360].drop(columns="label"), np.ones(360))
            # Unmasked TAB ADD callers preserve the established boundary-span policy.
            self.assertEqual(add(np.array([1, 1, 0, 1]), np.array([0, 1, 0, 0])), 1.0)
            # Cover the real long-table reader and multi-seed launcher without another training run.
            fixture = runtime / "dataset"
            (fixture / "data").mkdir(parents=True)
            pd.concat([pd.DataFrame({"date": np.arange(1, len(data) + 1), "data": data[column], "cols": column})
                       for column in data.columns], ignore_index=True).to_csv(fixture / "data" / "synthetic.csv", index=False)
            pd.DataFrame([{"file_name": "synthetic.csv", "train_lens": 360}]).to_csv(fixture / "META.csv", index=False)
            launcher = load_launcher()
            backend.close(force=True)
            launch_args = ["--data-name-list", "synthetic.csv", "--seeds", "42", "--device", "cpu", "--cpu-threads", "2",
                           "--output-dir", str(runtime / "launcher"), "--model-hyper-params",
                           json.dumps({"win_size": 64, "num_epochs": 1, "batch_size": 32, "target_chunk_size": 2})]
            with patch.object(launcher, "LocalAnomalyDetectDataSource", side_effect=lambda: LocalDataSource(str(fixture), "META.csv")), \
                 patch.object(DynamicNeighborhoodStat, "_train", side_effect=AssertionError("launcher unexpectedly retrained")), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                launcher.main(launch_args)
            summary = pd.read_csv(runtime / "launcher" / "summary.csv")
            self.assertEqual(len(summary), len(ratios) * len(classification_metrics_label.__all__) + len(classification_metrics_score.__all__))
            self.assertTrue((summary["valid_seeds"] == 1).all())
            self.assertTrue(summary["std"].isna().all())  # one seed cannot define sample std
            self.assertTrue((summary["valid_points_min"] == 107).all())
            self.assertTrue((summary["valid_points_max"] == 107).all())
            self.assertTrue((summary["excluded_boundary_events_max"] == 2).all())
            coverage = pd.read_csv(runtime / "launcher" / "evaluation_coverage.csv")
            self.assertEqual(len(coverage), len(ratios) + 1)
            self.assertTrue((coverage["test_length"] == 170).all())
            self.assertTrue((coverage["excluded_points"] == 63).all())
            self.assertTrue((coverage["valid_start"] == 63).all())
            self.assertTrue((coverage["valid_end_exclusive"] == 170).all())
            self.assertTrue((coverage["complete_events"] == 1).all())
            # Exercise the user's original CLI, including CSV loading and report
            # generation, against the same tiny fixture and calibrated checkpoint.
            from ts_benchmark.pipeline import PREDEFINED_DATASETS
            for mode, metrics in (("label", classification_metrics_label.__all__), ("score", classification_metrics_score.__all__)):
                entry_output = runtime / "original_entry" / mode
                entry_args = [str(ROOT / "scripts/run_benchmark.py"), "--config-path", "dynamic_neighborhood_stat_" + mode + ".json",
                              "--model-name", "self_impl.DynamicNeighborhoodStat", "--data-name-list", "synthetic.csv",
                              "--model-hyper-params", json.dumps(parameters), "--seed", "42", "--gpus", "0",
                              "--num-workers", "1", "--timeout", "-1", "--save-path", str(entry_output)]
                with patch.object(PREDEFINED_DATASETS["large_detect"], "datasrc_class", lambda: LocalDataSource(str(fixture), "META.csv")), \
                     patch.object(DynamicNeighborhoodStat, "_train", side_effect=AssertionError("original entry unexpectedly retrained")), \
                     patch.object(sys, "argv", entry_args), patch.object(logging, "basicConfig"), \
                     contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    runpy.run_path(str(ROOT / "scripts/run_benchmark.py"), run_name="__main__")
                entry_reports = list(entry_output.glob("test_report*"))
                self.assertEqual(len(entry_reports), 1)
                entry_report = pd.read_csv(entry_reports[0])
                self.assertEqual(set(entry_report["metric_name"]), set(metrics))
                self.assertEqual(len(entry_report), len(metrics) * (len(ratios) if mode == "label" else 1))
                self.assertTrue(np.isfinite(entry_report.drop(columns=["metric_name", "strategy_args", "typical_anomaly_ratio"],
                                                              errors="ignore").to_numpy(dtype=float)).all())
                # The normal TAB records must be readable CSVs with the same
                # complete metric columns as the corresponding AT strategy.
                entry_records = list(entry_output.glob("DynamicNeighborhoodStat.*.csv"))
                self.assertEqual(len(entry_records), 1)
                entry_record = pd.read_csv(entry_records[0])
                self.assertTrue(set(metrics).issubset(entry_record.columns))
                self.assertTrue((entry_record["log_info"].fillna("") == "").all())
                if mode == "label":
                    self.assertEqual(entry_record["typical_anomaly_ratio"].tolist(), ratios)
                    self.assertEqual(entry_report["typical_anomaly_ratio"].drop_duplicates().tolist(), ratios)
            # Rejected training data must remain attributable and must not turn into imputed results.
            invalid = data.copy()
            invalid.loc[0, "label"] = 1
            pool.get_series.return_value = invalid
            DataPool().set_pool(pool)
            strategy = UnFixedDetectLabel({"strategy_name": "unfixed_detect_label", "seed": 42}, Evaluator([{"name": "f_score"}]))
            with self.assertLogs("ts_benchmark.evaluation.strategy.anomaly_detect", level="ERROR") as error_log:
                failed = pd.DataFrame(strategy.execute("synthetic.csv", factory), columns=strategy.field_names)
            self.assertIn("Anomaly detection failed", error_log.output[0])
            self.assertEqual(failed.loc[0, "file_name"], "synthetic.csv")
            self.assertIn("clean normal", failed.loc[0, "log_info"])
            _, _, failed_summary = launcher.summarize([(failed, "label", 42, ["f_score"])], {})
            self.assertEqual(failed_summary.loc[0, "valid_seeds"], 0)
            self.assertEqual(failed_summary.loc[0, "coverage_seeds"], 0)
            self.assertTrue(np.isnan(failed_summary.loc[0, "valid_points_min"]))
        finally:
            DataPool().set_pool(old_pool)
            backend.close(force=True)
            # Verify the recursive cleanup target is a temporary child of validation/.
            if runtime.parent != (ROOT / "validation").resolve() or not runtime.name.startswith("dns-check-"):
                raise RuntimeError("Unexpected validation cleanup path")
            import shutil
            shutil.rmtree(runtime)

    def test_coverage_summary_sums_entities_without_multiplying_seeds(self):
        launcher = load_launcher()
        frame = pd.DataFrame({"file_name": ["one.csv", "two.csv"], "log_info": ["", ""], "f_score": [0.5, 1.0],
                              "test_length": [170, 80], "valid_points": [107, 17], "excluded_points": [63, 63],
                              "total_events": [3, 1], "complete_events": [1, 1], "excluded_boundary_events": [2, 0],
                              "valid_start": [63, 63], "valid_end_exclusive": [170, 80]})
        groups = {"one.csv": "dataset", "two.csv": "dataset"}
        records = [(frame, "label", 42, ["f_score"]), (frame.copy(), "label", 43, ["f_score"])]
        _, by_seed, summary = launcher.summarize(records, groups)
        self.assertEqual(by_seed["valid_points"].tolist(), [124, 124])
        self.assertEqual(summary.loc[0, "valid_points_min"], 124)
        self.assertEqual(summary.loc[0, "valid_points_max"], 124)
        self.assertEqual(summary.loc[0, "excluded_boundary_events_max"], 2)
        self.assertEqual(summary.loc[0, "coverage_seeds"], 2)
        self.assertEqual(summary.loc[0, "mean"], 0.75)
        # Ratios must not be averaged together or multiply coverage/entity counts.
        multi = pd.concat([frame.assign(typical_anomaly_ratio=0.5),
                           frame.assign(typical_anomaly_ratio=5.0, f_score=0.0)], ignore_index=True)
        _, grouped_seeds, grouped_summary = launcher.summarize(
            [(multi, "label", 42, ["f_score"]), (multi.copy(), "label", 43, ["f_score"])], groups)
        self.assertEqual(grouped_summary["mean"].tolist(), [0.75, 0.0])
        self.assertTrue((grouped_seeds["valid_points"] == 124).all())
        self.assertTrue((grouped_seeds["expected_entities"] == 2).all())
        incomplete = frame.copy()
        incomplete.loc[0, list(launcher.COVERAGE_FIELDS)] = np.nan
        incomplete.loc[0, "f_score"] = np.nan
        incomplete.loc[0, "log_info"] = "run failed"
        _, by_seed, summary = launcher.summarize([records[0], (incomplete, "label", 43, ["f_score"])], groups)
        self.assertTrue(np.isnan(by_seed.loc[1, "valid_points"]))
        self.assertEqual(summary.loc[0, "valid_seeds"], 1)
        self.assertEqual(summary.loc[0, "coverage_seeds"], 1)
        self.assertEqual(summary.loc[0, "expected_seeds"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
