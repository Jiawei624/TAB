"""Small real ASEI fits and formula/protocol checks; no benchmark training."""

import copy
import importlib
import tempfile
from pathlib import Path
from dataclasses import asdict
import json
from types import SimpleNamespace
import unittest
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

from ts_benchmark.baselines.self_impl import ASEI
from ts_benchmark.baselines.self_impl.ASEI.asei import BlockWindowDataset, Config, WindowDataset, collate_blocks
from ts_benchmark.baselines.self_impl.ASEI.auxiliary import TaskSampler, prepare_objective
from ts_benchmark.baselines.self_impl.ASEI.calibration import EvidenceReference, empirical_threshold, upper_tail
from ts_benchmark.baselines.self_impl.ASEI.model import ErrorCorrector, EvidenceScorer, Predictor, cpu_state, lag_graph, project_l1_, temporal_layout, unwrap
from ts_benchmark.baselines.self_impl.ASEI.optimization import LinearARObjective, fit_projected_ar
from ts_benchmark.baselines.self_impl.ASEI.protocol import make_protocol, select_protocol
from ts_benchmark.baselines.self_impl.ASEI.progress import Progress
from ts_benchmark.models.model_loader import get_models


SMALL = dict(hidden_dim=4, embedding_dim=2, dilations=[1], graph_lags=[1, 2],
             ar_order=2, rolling_window=2, predictor_epochs=1,
             correction_epochs=1, score_epochs=1, batch_size=128,
             correction_batch_size=64, auxiliary_per_epoch=2, auxiliary_batch_size=2,
             normal_prefix=32, recovery_suffix=20, validation_gap=12,
             checkpoint_epochs=[1], validation_seeds=[1001], event_lengths=[1, 4, 8],
             rank_pairs=4, inference_chunk_size=128, seed=3, alpha=0.01)


def data(length=1600):
    time = np.arange(length, dtype=np.float64)
    return pd.DataFrame({"a": np.sin(time / 7) + 0.1 * np.cos(time / 3),
                         "b": np.cos(time / 11) + 0.2 * np.sin(time / 7)})


class ConfigInterfaceTests(unittest.TestCase):
    def test_anomaly_transformer_style_parameters_are_effective(self):
        model = ASEI(batch_size=128, lr=0.0001, num_epochs=10, win_size=100, device="cpu")
        self.assertEqual(model.config.lr, 0.0001)
        self.assertEqual(model.config.learning_rate, 0.0001)
        self.assertEqual(model.config.num_epochs, 10)
        self.assertEqual(model.config.predictor_epochs, 10)
        self.assertEqual(model.config.correction_learning_rate, 1e-3)
        self.assertEqual(model.config.score_learning_rate, 1e-3)
        self.assertEqual(model.config.score_epochs, 30)
        self.assertEqual(model.config.checkpoint_epochs, (5, 10, 15, 20, 25, 30))
        self.assertEqual(model.receptive_field, 100)
        self.assertEqual(model.history, 115)
        self.assertEqual(model.raw_history, 126)
        self.assertEqual(model.temporal_layout_[-1], (2, 37))
        reconstructed = ASEI(**asdict(model.config))
        self.assertEqual(reconstructed.config.recovery_suffix, model.config.recovery_suffix)
        self.assertEqual(reconstructed.config.validation_gap, model.config.validation_gap)
        protocol = make_protocol(model.config, model.raw_history, 8, 0.125)
        self.assertEqual(protocol.recovery_suffix, 126 + int(np.ceil(679 * 0.125)))
        self.assertEqual(protocol.separated_gap, 126 + int(np.ceil(1319 * 0.125)))
        for kwargs in [dict(lr=0), dict(lr=True), dict(num_epochs=0), dict(win_size=0), dict(win_size=3.5),
                       dict(lr=1e-4, learning_rate=1e-3), dict(num_epochs=10, predictor_epochs=20)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ASEI(**kwargs, device="cpu")
        self.assertEqual(ASEI(lr=1e-4, learning_rate=1e-4).config.learning_rate, 1e-4)

    def test_exact_window_layout_and_causal_sequence_predictions(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            for window in [1, 2, 4, 63, 64, 100, 256]:
                layout, receptive_field = temporal_layout(3, [1, 2, 4, 8, 16], window)
                self.assertEqual(receptive_field, window)
                self.assertEqual(1 + sum((width - 1) * lag for width, lag in layout), window)
            graph = lag_graph(data(50).values, [1, 2], 1)
            model = Predictor(2, *graph, hidden=4, embedding=2, win_size=100)
            values = torch.randn(1, 110, 2)
            with torch.no_grad():
                actual = model(values, output_steps=10)
                expected = torch.cat([model(values[:, index:index + 101]) for index in range(10)], dim=1)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        finally:
            torch.set_num_threads(previous_threads)

    def test_original_tab_configs_compute_and_report_all_metrics(self):
        from ts_benchmark.evaluation.metrics import CLASSIFICATION_METRICS_LABEL, CLASSIFICATION_METRICS_SCORE
        root = Path(__file__).resolve().parents[1]
        for mode, metrics in [("label", CLASSIFICATION_METRICS_LABEL), ("score", CLASSIFICATION_METRICS_SCORE)]:
            config = json.loads((root / f"config/unfixed_detect_{mode}_multi_config.json").read_text())
            self.assertEqual(config["evaluation_config"]["metrics"], "all")
            self.assertEqual(set(config["report_config"]["report_metrics"]), set(metrics))


class ExecutionProgressTests(unittest.TestCase):
    def test_heartbeat_reports_a_waiting_phase_without_fabricating_progress(self):
        with self.assertLogs("ts_benchmark.baselines.self_impl.ASEI.progress", level="INFO") as logs:
            with Progress("waiting-test", total=4, interval=0.01) as progress:
                progress.phase("batch=1 forward")
                deadline = time.monotonic() + 1
                while not any("heartbeat" in record.getMessage() for record in logs.records):
                    if time.monotonic() >= deadline:
                        self.fail("No progress heartbeat was emitted")
                    time.sleep(0.01)
                progress.update(1)
        heartbeats = [record.getMessage() for record in logs.records if "heartbeat" in record.getMessage()]
        self.assertTrue(any("progress=0/4" in message and "phase=batch=1 forward" in message for message in heartbeats))
        self.assertTrue(any("finished" in record.getMessage() for record in logs.records))
        self.assertFalse(progress._thread.is_alive())

    def test_block_windows_cover_each_target_once_and_match_point_window_gradients(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            values = data(25).values.astype(np.float32)
            history, block_size = 8, 4
            point_dataset = WindowDataset(values, history)
            blocks = BlockWindowDataset(values, history, block_size)
            histories, targets, mask = collate_blocks([blocks[index] for index in range(len(blocks))])
            expected_targets = torch.from_numpy(values[history:])
            torch.testing.assert_close(targets[mask], expected_targets)
            self.assertEqual(int(mask.sum()), len(point_dataset))
            point_histories = torch.stack([point_dataset[index][0] for index in range(len(point_dataset))])
            graph = lag_graph(values, [1, 2], 1)
            point = Predictor(2, *graph, hidden=4, embedding=2, dilations=[1, 2]).double()
            blocked = copy.deepcopy(point)
            expected = point(point_histories.double())[:, 0]
            actual = blocked(histories.double(), output_steps=targets.shape[1])
            torch.testing.assert_close(actual[mask], expected, rtol=1e-10, atol=1e-11)
            loss_point = torch.nn.functional.huber_loss(expected, expected_targets.double()[..., None].expand_as(expected))
            losses = torch.nn.functional.huber_loss(actual, targets.double()[..., None].expand_as(actual), reduction="none").mean((-1, -2))
            loss_blocked = (losses * mask).sum() / mask.sum()
            torch.testing.assert_close(loss_point, loss_blocked, rtol=1e-10, atol=1e-11)
            loss_point.backward()
            loss_blocked.backward()
            for first, second in zip(point.parameters(), blocked.parameters()):
                torch.testing.assert_close(first.grad, second.grad, rtol=1e-9, atol=1e-10)
        finally:
            torch.set_num_threads(previous_threads)


class FormulaTests(unittest.TestCase):
    def test_upper_tail_ties_branch_fusion_and_centering(self):
        np.testing.assert_allclose(upper_tail(np.array([0, 1, 2, 3]), np.array([1, 1, 2])),
                                   [1, 1, 0.5, 0.25])
        errors = np.arange(16, dtype=float).reshape(4, 2, 2)
        reference = EvidenceReference(6).fit(errors, errors - 3)
        evidence = reference.transform(errors, errors - 3)
        np.testing.assert_allclose(evidence.mean(0), 0, atol=1e-15)
        self.assertTrue(np.all(np.abs(evidence) <= 6))
        threshold, actual = empirical_threshold(np.array([-2, -1, -1, 3, 3]), 0.4)
        self.assertEqual(threshold, -1)
        self.assertEqual(actual, 2)
        self.assertEqual(empirical_threshold(np.arange(100), 0.58), (42, 57))

    def test_exact_threshold_budget_with_ties_small_n_and_negative_scores(self):
        for scores in [np.arange(250), np.array([-4.] * 150 + [-2.] * 50),
                       np.array([-7., -6., -5.]), np.ones(100)]:
            with self.subTest(length=len(scores)):
                threshold, actual = empirical_threshold(scores, 0.01)
                budget = int(np.floor(0.01 * len(scores)))
                self.assertEqual(threshold, np.sort(scores)[len(scores) - budget - 1])
                self.assertEqual(actual, np.count_nonzero(scores > threshold))
                self.assertLessEqual(actual, budget)
        self.assertEqual(empirical_threshold(np.array([-4., -3., -2.]), 0.01), (-2., 0))

    def test_graph_empty_neighborhood_and_deterministic_ties(self):
        graph = lag_graph(np.ones((30, 3)), [1, 2], 2)
        self.assertFalse(graph[2].any())
        trend = np.arange(32, dtype=float)
        sources, delays, mask = lag_graph(np.stack([trend] * 3, -1), [1, 2], 1)
        np.testing.assert_array_equal(sources[:, 0], [1, 0, 0])
        np.testing.assert_array_equal(delays[:, 0], [1, 1, 1])
        self.assertTrue(mask.all())

    def test_constraints_compensation_and_signed_recurrence(self):
        values = torch.tensor([[3., -1., 0.], [0.1, 0.2, -0.1]])
        project_l1_(values, 1)
        torch.testing.assert_close(values[0], torch.tensor([1., 0., 0.]))
        self.assertTrue(torch.all(values.abs().sum(-1) <= 1))
        double_values = torch.tensor([[3., -1., 0.], [0.1, 0.2, -0.1]], dtype=torch.float64)
        project_l1_(double_values, 0.95)
        torch.testing.assert_close(double_values, torch.tensor([[0.95, 0., 0.], [0.1, 0.2, -0.1]], dtype=torch.float64))
        corrector = ErrorCorrector(2, 2, 0.95)
        corrector.eta.fill_(0.1)
        corrector.mlp[-1].bias.data.fill_(100)
        history = torch.ones(3, 2, 2, 2)
        difference = corrector(history) - corrector(history, nonlinear=False)
        self.assertTrue(torch.all(difference.abs() <= 0.100001))
        scorer = EvidenceScorer(6, 0.05, 0.8, 0.98, 0.9, 0.02, 0.25)
        features = torch.zeros(1, 3, 2, 4)
        features[:, 0, :, :2] = 1
        features[:, 1:, :, :2] = -1
        scores, state = scorer.integrate(features)
        torch.testing.assert_close(scores[0], torch.tensor([0.98, -0.138, -1.02]), atol=1e-6, rtol=1e-5)
        self.assertTrue(torch.all(state == 0))

    def test_sequence_predictor_matches_independent_causal_windows(self):
        torch.manual_seed(2)
        graph = lag_graph(data(50).values, [1, 2], 1)
        model = Predictor(2, *graph, hidden=4, embedding=2, dilations=[1])
        values = torch.randn(1, 12, 2)
        with torch.no_grad():
            actual = model(values, output_steps=9)
            expected = torch.cat([model(values[:, index:index + 4]) for index in range(9)], 1)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


class ProtocolAndARTests(unittest.TestCase):
    def test_largest_feasible_scale_all_scenarios_and_seed_independence(self):
        cfg = Config()
        for factor in [1, 0.5, 0.25, 0.125]:
            protocol = make_protocol(cfg, 89, 8, factor)
            train_required, required = protocol.required_lengths()
            selected, _ = select_protocol(cfg, 89, 8, train_required, max(required.values()))
            self.assertEqual(selected.s, factor)
            self.assertEqual(protocol.normal_prefix, int(np.ceil(640 * factor)))
            self.assertEqual(protocol.recovery_suffix, 89 + int(np.ceil(679 * factor)))
            self.assertEqual(protocol.separated_gap, 89 + int(np.ceil(1319 * factor)))
            self.assertAlmostEqual(protocol.lambda_min, 0.8 ** (1 / factor))
            self.assertAlmostEqual(protocol.lambda_max, 0.98 ** (1 / factor))
            self.assertAlmostEqual(protocol.lambda_init, 0.9 ** (1 / factor))
            self.assertEqual(len(set(protocol.training_lengths)), len(protocol.training_lengths))
            self.assertEqual(len(set(protocol.short_lengths)), len(protocol.short_lengths))
            sampler = TaskSampler(data(max(required.values())).values, 89, 8, cfg, protocol)
            tasks = sampler.validation()
            self.assertEqual(len(tasks), 72)
            self.assertEqual(set(task.scenario for task in tasks), {"short", "long", "dense", "paired"})
            for task in tasks:
                self.assertLessEqual(task.length, required[task.scenario])
                self.assertLessEqual(task.start + task.length, len(sampler.values))
                self.assertGreaterEqual(len(task.events), 3)
            for seed in range(5):
                seeded = Config(seed=seed)
                self.assertEqual(select_protocol(seeded, 89, 8, train_required, max(required.values()))[0], selected)
        minimum = make_protocol(cfg, 89, 8, 0.125)
        train_required, required = minimum.required_lengths()
        with self.assertRaisesRegex(ValueError, "s=0.125 cannot fit"):
            select_protocol(cfg, 89, 8, train_required - 1, max(required.values()))
        with self.assertRaisesRegex(ValueError, "s=0.125 cannot fit"):
            select_protocol(cfg, 89, 8, train_required, max(required.values()) - 1)

    def test_full_batch_double_huber_backtracking_and_real_convergence(self):
        rng = np.random.default_rng(4)
        history = torch.tensor(rng.normal(size=(64, 1, 2, 4)), dtype=torch.float64)
        target = 0.3 + 0.4 * history[..., 0]
        module = ErrorCorrector(1, 4, 0.95)
        module.bias.data.copy_(target.mean(0))
        status = fit_projected_ar(LinearARObjective(module), history, target)
        self.assertTrue(status["converged"])
        self.assertLessEqual(status["gradient_mapping_inf"], 1e-5)
        self.assertLess(status["iterations"], 1000)
        self.assertGreater(status["backtracking_reductions"], 0)
        self.assertEqual(module.beta.dtype, torch.float64)
        self.assertLessEqual(module.beta.abs().sum(-1).max().item(), 0.95 + 1e-12)
        torch.testing.assert_close(module.beta[..., 0], torch.full((1, 2), 0.4, dtype=torch.float64), atol=1e-4, rtol=0)
        self.assertAlmostEqual(status["trace"][0]["step"], 0.01)
        for item in status["trace"]:
            self.assertLessEqual(item["candidate_loss"], item["quadratic_upper_bound"] + item["numerical_slack"])
        limited = ErrorCorrector(1, 4, 0.95)
        limited.bias.data.copy_(target.mean(0))
        capped = fit_projected_ar(LinearARObjective(limited), history, target, max_iter=1)
        self.assertFalse(capped["converged"])
        self.assertEqual(capped["stopping_reason"], "max_iter")
        constrained = ErrorCorrector(1, 4, 0.95)
        hard_target = 3 + 2 * history[..., 0]
        constrained.bias.data.copy_(hard_target.mean(0))
        constrained_status = fit_projected_ar(LinearARObjective(constrained), history, hard_target)
        self.assertTrue(constrained_status["converged"])
        self.assertGreater(constrained.beta.abs().sum(-1).min().item(), 0.94999)
        self.assertLessEqual(constrained.beta.abs().sum(-1).max().item(), 0.95 + 1e-12)


class SelectionTests(unittest.TestCase):
    def test_six_configs_all_checkpoints_tie_order_and_entity_isolation(self):
        def fit_selection(seed, losses):
            model = ASEI(**dict(SMALL, score_epochs=30, checkpoint_epochs=[5, 10, 15, 20, 25, 30], seed=seed), device="cpu")
            model.columns_, model.metadata_ = ["a", "b"], {}
            model.reference = SimpleNamespace(center=np.zeros((2, 2)))
            model.protocol_, _ = select_protocol(model.config, model.raw_history, 8, 1024, 320)
            def small_batch(tasks, sampler):
                return torch.zeros(len(tasks), 2, 2, 2, 4), []
            def training_loss(candidate, features, metadata):
                connected = sum(parameter.square().sum() for parameter in candidate.parameters()) * 0
                return connected + 1, connected + 1
            with patch.object(model, "_feature_chunks", return_value=[np.zeros((1024 - model.raw_history, 2, 4), dtype=np.float32)]), \
                    patch.object(model, "_batch", side_effect=small_batch), \
                    patch.object(model, "_loss", side_effect=training_loss), \
                    patch.object(model, "_validate_scorers", return_value=losses):
                model._fit_scorer(data(1024).values, data(320).values, 8)
            return model
        first = fit_selection(0, [1] * 6)
        snapshot = copy.deepcopy(first.metadata_)
        second = fit_selection(1, [4, 3, 0, 2, 1, 5])
        self.assertEqual(first.metadata_, snapshot)
        self.assertEqual(first.metadata_["selection"]["epoch"], 5)
        self.assertEqual(first.metadata_["selection"]["config_index"], 0)
        self.assertEqual(second.metadata_["selection"]["config_index"], 2)
        self.assertEqual(first.metadata_["saved_scoring_checkpoints"], 36)
        self.assertEqual(list(first.scoring_checkpoints_), [5, 10, 15, 20, 25, 30])
        self.assertEqual([(item["kappa"], item["gamma"]) for item in first.metadata_["scoring_candidates"]],
                         [(0.01, 0.5), (0.01, 1), (0.02, 0.5), (0.02, 1), (0.04, 0.5), (0.04, 1)])
        second.scoring_checkpoints_[5][0]["state"]["gate_bias"].add_(10)
        self.assertEqual(first.scoring_checkpoints_[5][0]["state"]["gate_bias"].item(), 0)

    def test_validation_group_weights_and_gamma_independence(self):
        model = ASEI(**SMALL, device="cpu")
        candidates = [dict(model=nn.Identity(), gamma=0.5), dict(model=nn.Identity(), gamma=1)]
        tasks = [SimpleNamespace(family=0, scenario="short", loss=1)] * 10
        tasks += [SimpleNamespace(family=1, scenario="long", loss=9)]
        with patch.object(model, "_batch", side_effect=lambda batch, sampler: (batch[0].loss, [])), \
                patch.object(model, "_loss", side_effect=lambda candidate, features, metadata:
                             (torch.tensor(features / 3), torch.tensor(features * 2 / 3))):
            losses = model._validate_scorers(candidates, None, tasks)
        self.assertEqual(losses, [5, 5])


class FitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.detector = ASEI(**SMALL, device="cpu")
        cls.detector.detect_fit(data())

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_full_stages_splits_frozen_parameters_and_reference(self):
        model = self.detector
        np.testing.assert_allclose(model.scaler.mean_, data().iloc[:1024].mean())
        self.assertEqual(model.metadata_["reference_points"], 128 - model.history - 2)
        self.assertEqual(len(model.metadata_["selection_history"][0]["losses"]), 6)
        self.assertEqual(model.metadata_["saved_scoring_checkpoints"], 6)
        self.assertTrue(all(not parameter.requires_grad for module in
                            [model.predictor, model.corrector, model.scorer] for parameter in module.parameters()))
        self.assertLessEqual(model.corrector.beta.abs().sum(-1).max().item(), 0.950001)
        self.assertLessEqual(model.scorer.gate_weight.abs().sum().item(), 2.000001)
        self.assertTrue(all(item["empirical_fpr"] <= item["alpha"] for item in model.metadata_["calibration"]))
        for item in model.metadata_["calibration"]:
            self.assertEqual(item["N"], len(model.validation_scores_))
            self.assertEqual(item["false_alarm_count"], np.count_nonzero(model.validation_scores_ > item["threshold"]))
            self.assertLessEqual(item["false_alarm_count"], item["K"])
        self.assertEqual(model.corrector.beta.dtype, torch.float64)
        self.assertEqual(model.corrector.mlp[0].weight.dtype, torch.float32)
        self.assertEqual(model.config.learning_rate, 1e-3)
        self.assertEqual(model.config.correction_learning_rate, 1e-3)
        self.assertEqual(model.config.score_learning_rate, 1e-3)
        ar_status = model.metadata_["corrector"]["ar_optimization"]
        self.assertEqual(ar_status["converged"], ar_status["gradient_mapping_inf"] <= 1e-5)
        history = model.metadata_["selection_history"][0]
        for loss, groups in zip(history["losses"], history["groups"]):
            self.assertEqual(len(groups), 12)
            self.assertAlmostEqual(loss, np.mean([group["loss"] for group in groups.values()]))

    def test_separate_entity_model_references_thresholds_and_protocols(self):
        first = self.detector
        before = copy.deepcopy(first.metadata_)
        reference = first.reference.errors.copy()
        second = ASEI(**dict(SMALL, seed=4), device="cpu").detect_fit(data(1700))
        self.assertEqual(first.metadata_, before)
        np.testing.assert_array_equal(first.reference.errors, reference)
        self.assertFalse(np.shares_memory(first.reference.errors, second.reference.errors))
        self.assertNotEqual(first.corrector.mlp[0].weight.data_ptr(), second.corrector.mlp[0].weight.data_ptr())
        self.assertNotEqual(first.metadata_["calibration"][0]["N"], second.metadata_["calibration"][0]["N"])
        self.assertEqual(first.config.lambda_min, 0.8)
        self.assertEqual(second.config.lambda_min, 0.8)
        self.assertIsNot(first.scoring_checkpoints_, second.scoring_checkpoints_)

    def test_alias_parameters_train_and_survive_checkpoint_reload(self):
        model = ASEI(**dict(SMALL, lr=1e-4, num_epochs=1, win_size=6), device="cpu")
        model.detect_fit(data())
        self.assertEqual(model.metadata_["predictor_learning_rate"], 1e-4)
        self.assertEqual(model.metadata_["predictor_max_epochs"], 1)
        self.assertEqual(unwrap(model.predictor).receptive_field, 6)
        self.assertEqual(model.raw_history, 10)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aliases.pt"
            model.save(path)
            restored = ASEI.load(path, device="cpu")
            self.assertEqual(restored.config.learning_rate, 1e-4)
            self.assertEqual(restored.config.predictor_epochs, 1)
            self.assertEqual(restored.temporal_layout_, model.temporal_layout_)
            self.assertEqual(restored.config.recovery_suffix, model.config.recovery_suffix)
            np.testing.assert_array_equal(restored.detect_score(data(40))[0], model.detect_score(data(40))[0])

    def test_blocked_fit_progress_checkpoint_and_partial_tail(self):
        model = ASEI(**dict(SMALL, predictor_block_size=16), device="cpu")
        with self.assertLogs("ts_benchmark.baselines.self_impl.ASEI", level="INFO") as logs:
            model.detect_fit(data())
        messages = "\n".join(record.getMessage() for record in logs.records)
        self.assertIn("predictor epoch=1 train", messages)
        self.assertIn("predictor epoch=1 validation", messages)
        self.assertIn("progress=1/", messages)
        self.assertIn("block_size=16", messages)
        self.assertIn("stage=3/4 independent_reference", messages)
        self.assertIn("normal validation calibration", messages)
        loader = model._predictor_loader(data().iloc[:1024].values)
        self.assertEqual(sum(int(batch[2].sum()) for batch in loader), 1024 - model.history)
        self.assertEqual(len(loader), int(np.ceil((1024 - model.history) / 128)))
        baseline = model._predictor_loss(model._loader(WindowDataset(data(137).values, model.history), 128))
        blocked = model._predictor_loss(model._predictor_loader(data(137).values))
        self.assertAlmostEqual(baseline, blocked, places=5)
        self.assertEqual(len(model.metadata_["stage_seconds"]), 5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blocked.pt"
            model.save(path)
            restored = ASEI.load(path, device="cpu")
            self.assertEqual(restored.config.predictor_block_size, 16)
            np.testing.assert_array_equal(restored.detect_score(data(40))[0], model.detect_score(data(40))[0])

    def test_every_timestamp_and_streaming_causality(self):
        model = self.detector
        values = data(83)
        labels, whole = model.detect_label(values)
        model.reset_state()
        parts, label_parts = [], {key: [] for key in labels}
        for begin, end in [(0, 2), (2, 5), (5, 13), (13, 31), (31, 83)]:
            current_labels, current = model.detect_stream(values.iloc[begin:end])
            parts.append(current)
            for key in labels:
                label_parts[key].append(current_labels[key])
        np.testing.assert_allclose(np.concatenate(parts), whole, rtol=2e-5, atol=2e-5)
        for key in labels:
            np.testing.assert_array_equal(np.concatenate(label_parts[key]), labels[key])
            self.assertFalse(labels[key][:model.raw_history].any())
        changed = values.copy()
        changed.iloc[50:] += 100
        _, future = model.detect_score(changed)
        np.testing.assert_allclose(future[:50], whole[:50], rtol=1e-5, atol=1e-5)
        for length in [1, 3, 7, 8, 17]:
            scores, _ = model.detect_score(data(length))
            self.assertEqual(scores.shape, (length,))
            self.assertTrue(np.isfinite(scores).all())

    def test_calibration_independent_of_test_and_complete_checkpoint(self):
        model = self.detector
        thresholds = model.thresholds_.copy()
        scores, _ = model.detect_score(data(35))
        adapter = importlib.import_module("ts_benchmark.baselines.self_impl.ASEI.asei")
        with patch.object(adapter, "empirical_threshold", side_effect=AssertionError("test data cannot calibrate")):
            model.detect_label(data(35) * 100)
        self.assertEqual(thresholds, model.thresholds_)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asei.pt"
            model.save(path)
            restored = ASEI.load(path, device="cpu")
            actual, _ = restored.detect_score(data(35))
            np.testing.assert_array_equal(actual, scores)
            self.assertEqual(restored.thresholds_, thresholds)
            self.assertEqual(restored.protocol_, model.protocol_)
            self.assertEqual(len(restored.scoring_checkpoints_[1]), 6)
            self.assertFalse(any(key.startswith("module.") for key in cpu_state(restored.predictor)))

    def test_perturbations_restore_suffix_and_objective_excludes_prefix(self):
        model = self.detector
        values = model.scaler.transform(data().iloc[:1024].values).astype(np.float32)
        sampler = TaskSampler(values, model.raw_history, model.metadata_["T0"], model.config, model.protocol_)
        task = sampler.training(np.random.default_rng(4))
        normal, changed = sampler.materialize(task)
        start, end = task.events[0]
        np.testing.assert_array_equal(normal[:start], changed[:start])
        np.testing.assert_array_equal(normal[end:], changed[end:])
        self.assertTrue(np.any(changed[start] != normal[start]))
        self.assertTrue(np.any(changed[end - 1] != normal[end - 1]))
        metadata = prepare_objective(task, model.raw_history, 4, np.random.default_rng(4))
        self.assertFalse(metadata["negative"][:, :model.protocol_.normal_prefix].any())
        self.assertAlmostEqual(metadata["positive"].sum(), 1)

    def test_loader_and_explicit_errors(self):
        factory = get_models({"models": [{"model_name": "self_impl.ASEI", "model_hyper_params": SMALL}]})[0]
        self.assertIsInstance(factory(), ASEI)
        with self.assertRaisesRegex(ValueError, "split has"):
            ASEI(**SMALL, device="cpu").detect_fit(data(30))
        with self.assertRaisesRegex(ValueError, "s=0.125 cannot fit"):
            ASEI(**dict(SMALL, normal_prefix=6400), device="cpu").detect_fit(data())
        with self.assertRaisesRegex(ValueError, "columns"):
            self.detector.detect_score(data(10)[["b", "a"]])
        with patch.object(torch.cuda, "is_available", return_value=False):
            self.assertEqual(ASEI(**SMALL).device.type, "cpu")
        with self.assertRaisesRegex(ValueError, "Unknown ASEI hyperparameters"):
            ASEI(**SMALL, multi_gpu=True)
        with self.assertRaisesRegex(ValueError, "corrector split has"):
            ASEI(device="cpu").detect_fit(data(1000))

    def test_tab_file_training_all_metrics_records_and_csv(self):
        from ts_benchmark.data.data_pool import DataPool
        from ts_benchmark.pipeline import pipeline
        from ts_benchmark.recording import load_record_data
        from ts_benchmark.report import report_csv
        from ts_benchmark.utils.parallel import ParallelBackend

        previous_pool = DataPool().pool
        try:
            with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ"):
                root = Path(directory)
                (root / "data").mkdir()
                values = data(1645)
                values.loc[1616:1621, "a"] += 4
                values["label"] = ((np.arange(len(values)) >= 1616) & (np.arange(len(values)) <= 1621)).astype(int)
                raw = pd.concat([pd.DataFrame({"date": np.arange(len(values)), "data": values[column], "cols": column})
                                 for column in values], ignore_index=True)
                raw.to_csv(root / "data/tiny.csv", index=False)
                pd.DataFrame({"file_name": ["tiny.csv"], "train_lens": [1600], "size": ["large"]}) \
                    .to_csv(root / "DETECT_META.csv", index=False)
                backend = ParallelBackend()
                backend.init(backend="sequential", n_cpus=1, gpu_devices=[])
                try:
                    for mode, metric_count, rows in [("score", 6, 1), ("label", 16, 1)]:
                        with self.subTest(mode=mode):
                            config_path = Path(__file__).resolve().parents[1] / f"config/unfixed_detect_{mode}_multi_config.json"
                            config = json.loads(config_path.read_text())
                            model_config = {"models": [{"model_name": "self_impl.ASEI", "model_hyper_params":
                                dict(SMALL, device="cpu", checkpoint_dir=str(root / "checkpoints"))}]}
                            with patch("ts_benchmark.data.data_source.ANOMALY_DETECT_DATASET_PATH", directory):
                                paths = pipeline({"data_name_list": ["tiny.csv"]}, model_config,
                                                 config["evaluation_config"], str(root / mode))
                            records = load_record_data(paths)
                            self.assertEqual(len(records), rows)
                            self.assertTrue(records["log_info"].fillna("").eq("").all(), records["log_info"].tolist())
                            metrics = config["report_config"]["report_metrics"]
                            self.assertEqual(len(metrics), metric_count)
                            self.assertTrue(np.isfinite(records[metrics].to_numpy()).all())
                            report_csv.report(dict(config["report_config"], log_files_list=records,
                                                   save_path=str(root / f"{mode}_report"), leaderboard_file_name="metrics.csv"))
                            report = pd.read_csv(root / f"{mode}_report/metrics.csv")
                            self.assertEqual(set(report["metric_name"]), set(metrics))
                    self.assertEqual(len(list((root / "checkpoints").glob("*.pt"))), 1)
                finally:
                    backend.close()
        finally:
            DataPool().pool = previous_pool


@unittest.skipUnless(torch.cuda.is_available(), "Requires one CUDA GPU; execute on the server")
class SingleGpuTests(unittest.TestCase):
    def test_single_device_fit_tail_checkpoint_and_equivalent_recurrence(self):
        detector = ASEI(**dict(SMALL, predictor_block_size=16), device="cuda:0").detect_fit(data())
        self.assertEqual({parameter.device.index for module in [detector.predictor, detector.corrector, detector.scorer]
                          for parameter in module.parameters()}, {0})
        self.assertIsInstance(detector.predictor, Predictor)
        labels, scores = detector.detect_label(data(35))
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(all(len(value) == 35 for value in labels.values()))
        values = detector.scaler.transform(data(150).values).astype(np.float32)
        np.testing.assert_allclose(detector._features(values), detector._features_reference(values), rtol=1e-5, atol=1e-5)
        features = torch.randn(2, 2, 50, 2, 4, device="cuda:0")
        optimized = copy.deepcopy(detector.scorer).requires_grad_(True)
        reference = copy.deepcopy(optimized)
        reference.recurrence_backend = "reference"
        for model in [optimized, reference]:
            outputs = model(features)
            sum(value.square().mean() for value in outputs).backward()
        for actual, expected in zip(optimized.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-4, atol=1e-5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "single.pt"
            detector.save(path)
            restored = ASEI.load(path, device="cpu")
            self.assertEqual(detector.thresholds_, restored.thresholds_)
            self.assertTrue(np.isfinite(restored.detect_score(data(35))[0]).all())


if __name__ == "__main__":
    unittest.main()
