"""ASEI protocol, device-local evidence, adjoint and metric-cache regressions."""

from dataclasses import asdict
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ts_benchmark.baselines.self_impl.ASEI.asei import ASEI, Config
from ts_benchmark.baselines.self_impl.ASEI.auxiliary import IneffectivePerturbation, Task, TaskSampler
from ts_benchmark.baselines.self_impl.ASEI.cache import ValidationCache
from ts_benchmark.baselines.self_impl.ASEI.calibration import EvidenceReference, error_history, local_features, local_features_tensor
from ts_benchmark.baselines.self_impl.ASEI.model import EvidenceScorer, Predictor, lag_graph
from ts_benchmark.baselines.self_impl.ASEI.recurrence import integrate
from ts_benchmark.evaluation.evaluator import Evaluator
from ts_benchmark.evaluation.metrics import classification_metrics_score as score_metrics
from test_asei import SMALL, data


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.values = np.zeros((1200, 2), dtype=np.float32)
        self.values[600:] = data(600).values
        self.config = Config(**SMALL)
        self.task = Task(0, 64, [(39, 44)], 2, 0, 123, 1, "short")

    def test_local_flat_base_is_relocated_without_changing_the_task_family(self):
        sampler = TaskSampler(self.values, 7, 8, self.config)
        with self.assertRaises(IneffectivePerturbation):
            sampler._materialize_fixed(self.task)
        resolved, normal, changed = sampler.resolve(self.task)
        self.assertGreater(resolved.start, 0)
        self.assertGreater(resolved.base_attempts, 0)
        self.assertEqual(resolved.family, 2)
        self.assertEqual(resolved.events, self.task.events)
        begin, end = resolved.events[0]
        self.assertTrue(np.any(changed[begin] != normal[begin]))
        self.assertTrue(np.any(changed[end - 1] != normal[end - 1]))
        np.testing.assert_array_equal(changed[:begin], normal[:begin])
        np.testing.assert_array_equal(changed[end:], normal[end:])
        self.assertEqual(self.task.start, 0)
        again, original_again, changed_again = sampler.resolve(copy.deepcopy(self.task))
        self.assertEqual(asdict(again), asdict(resolved))
        serialized = Task(**json.loads(json.dumps(asdict(self.task))))
        self.assertEqual(asdict(sampler.resolve(serialized)[0]), asdict(resolved))
        np.testing.assert_array_equal(original_again, normal)
        np.testing.assert_array_equal(changed_again, changed)

    def test_fixed_validation_tasks_reproduce_after_resampling(self):
        first = TaskSampler(self.values, 7, 8, self.config)
        second = TaskSampler(self.values, 7, 8, self.config)
        tasks = first.validation()
        np.random.default_rng(8).normal(size=100)
        recreated = second.validation()
        self.assertEqual([asdict(task) for task in tasks], [asdict(task) for task in recreated])
        self.assertTrue(any(task.base_attempts > 0 for task in tasks))
        for task in tasks:
            normal, changed = first.materialize(task)
            for begin, end in task.events:
                self.assertTrue(np.any(changed[begin] != normal[begin]))
                self.assertTrue(np.any(changed[end - 1] != normal[end - 1]))
        with self.assertRaisesRegex(IneffectivePerturbation, "No effective auxiliary base"):
            TaskSampler(np.zeros_like(self.values), 7, 8, self.config).resolve(self.task)


class RecurrenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def compare(self, evidence, initial, decay, kappa):
        sources = [value.detach().clone().requires_grad_(True) for value in (evidence, decay, initial)]
        expected_sources = [value.detach().clone().requires_grad_(True) for value in sources]
        actual = integrate(sources[0], sources[1], sources[2], kappa, "scripted")
        expected = integrate(expected_sources[0], expected_sources[1], expected_sources[2], kappa, "reference")
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, rtol=1e-11, atol=1e-12)
        actual_grad = torch.autograd.grad(actual[0].square().sum() + actual[1].sum(), sources)
        expected_grad = torch.autograd.grad(expected[0].square().sum() + expected[1].sum(), expected_sources)
        for left, right in zip(actual_grad, expected_grad):
            torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-10)

    def test_values_and_gradients_random_long_run_ties_and_relu_zero(self):
        generator = torch.Generator().manual_seed(6)
        for length in [1, 13, 1024]:
            self.compare(torch.randn(2, length, 3, generator=generator, dtype=torch.float64),
                         torch.rand(2, 3, generator=generator, dtype=torch.float64),
                         torch.tensor(0.98, dtype=torch.float64), 0.02)
        self.compare(torch.zeros(1, 5, 3, dtype=torch.float64), torch.zeros(1, 3, dtype=torch.float64),
                     torch.tensor(0.9, dtype=torch.float64), 0)
        self.compare(torch.tensor([[[1., 1.], [-0.5, -0.5], [1., 1.]]], dtype=torch.float64),
                     torch.zeros(1, 2, dtype=torch.float64), torch.tensor(0.5, dtype=torch.float64), 0)

    def test_all_scorer_parameter_gradients_and_finite_difference(self):
        scorer = EvidenceScorer(6, 0.05, 0.8, 0.98, 0.9, 0.02, 0.25).double()
        reference = copy.deepcopy(scorer)
        reference.recurrence_backend = "reference"
        features = torch.randn(2, 2, 12, 3, 4, dtype=torch.float64)
        for model in (scorer, reference):
            scores, logits = model(features)
            (scores.square().mean() + torch.nn.functional.softplus(logits).mean()).backward()
        for left, right in zip(scorer.parameters(), reference.parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-10, atol=1e-10)
        evidence = torch.randn(1, 3, 2, dtype=torch.float64, requires_grad=True)
        decay = torch.tensor(0.91, dtype=torch.float64, requires_grad=True)
        initial = torch.rand(1, 2, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda e, d, m: integrate(e, d, m, 0.02), (evidence, decay, initial)))


class EvidenceAndProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.model = ASEI(**SMALL, device="cpu").detect_fit(data())

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_tensor_evidence_matches_reference_including_tail_ties(self):
        rng = np.random.default_rng(12)
        errors = rng.integers(0, 5, size=(40, 3, 2)).astype(np.float32)
        residuals = errors.astype(np.float64) - 2
        reference = EvidenceReference(6).fit(errors, residuals)
        baseline = reference.transform(errors, residuals)
        tensor = reference.transform_tensor(torch.from_numpy(errors), torch.from_numpy(residuals))
        np.testing.assert_allclose(tensor.numpy(), baseline, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(local_features_tensor(tensor, 8).numpy(), local_features(baseline, 8), atol=1e-6)
        model = self.model
        values = model.scaler.transform(data(150).values).astype(np.float32)
        np.testing.assert_allclose(model._features(values), model._features_reference(values), rtol=1e-6, atol=1e-6)
        d = model._errors(values)
        history = torch.from_numpy(error_history(d, model.config.ar_order))
        with torch.no_grad():
            predictions = torch.cat([model.corrector(history[begin:begin + model.config.correction_batch_size])
                                     for begin in range(0, len(history), model.config.correction_batch_size)])
            expected_r = d[model.config.ar_order:] - predictions.numpy()
        np.testing.assert_allclose(model._residual_tensor(torch.from_numpy(d)).numpy(), expected_r, rtol=1e-12, atol=1e-12)

    def test_explicit_validation_preserves_training_parts_and_reference_warmup(self):
        model = ASEI(**SMALL, device="cpu").detect_fit(data().iloc[:1280], valid_data=data().iloc[1280:])
        self.assertTrue(model.metadata_["splits"]["external_validation"])
        self.assertEqual(model.metadata_["splits"]["outer_training_end"], 1280)
        np.testing.assert_array_equal(model.scaler.mean_, self.model.scaler.mean_)
        np.testing.assert_array_equal(model.reference.errors, self.model.reference.errors)
        self.assertEqual(model.metadata_["reference_warmup"], model.history + model.config.ar_order)
        self.assertEqual(model.metadata_["reference_points"], 128 - model.metadata_["reference_warmup"])
        self.assertEqual(model.metadata_["score_warmup"], model.raw_history)
        self.assertLess(model.metadata_["reference_warmup"], model.metadata_["score_warmup"])
        default = ASEI(device="cpu")
        self.assertEqual(default.history + default.config.ar_order, 82)
        self.assertEqual(default.raw_history, 89)

    def test_single_sample_tail_regression_and_fixed_validation_cache(self):
        model = ASEI(**dict(SMALL, predictor_block_size=16, validation_cache_mb=0,
                           score_epochs=2, checkpoint_epochs=[1, 2]), device="cpu")
        with patch.object(model, "_batch", wraps=model._batch) as batch:
            model.detect_fit(data())
        validation_builds = sum(call.args[0][0].scenario != "train" for call in batch.call_args_list)
        self.assertEqual(validation_builds, len(model.metadata_["validation_tasks"]))
        self.assertEqual(model.metadata_["validation_evidence_builds"], validation_builds)
        self.assertEqual(len(model.metadata_["selection_history"]), 2)
        graph = lag_graph(data(40).values, [1, 2], 1)
        predictor = Predictor(2, *graph, hidden=4, embedding=2, dilations=[1])
        for steps in (1, 16):
            predictor.zero_grad(set_to_none=True)
            output = predictor(torch.randn(1, 4 + steps - 1, 2), output_steps=steps)
            self.assertEqual(output.shape, (1, steps, 2, 2))
            output.square().mean().backward()
        self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in predictor.parameters()))
        features = torch.randn(1, 2, 10, 2, 4)
        with ValidationCache(torch.device("cpu"), 0) as cache:
            cache.append(features, [])
            location = cache.entries[0][0]
            torch.testing.assert_close(cache.get(0)[0], features)
            self.assertTrue(Path(location).exists())
        self.assertFalse(Path(location).exists())

    def test_reference_requires_prediction_and_ar_history_but_no_rolling_history(self):
        model = self.model
        minimum = model.history + model.config.ar_order + 1
        model_copy = copy.deepcopy(model)
        values = model.scaler.transform(data(minimum).values).astype(np.float32)
        model_copy._fit_reference(values)
        self.assertEqual(len(model_copy.reference.errors), 1)
        with self.assertRaisesRegex(ValueError, "no valid d/r sample"):
            model_copy._fit_reference(values[:-1])


class VUSReuseTests(unittest.TestCase):
    def test_shared_curve_preserves_both_values_and_does_not_leak_between_calls(self):
        labels = np.array([0] * 5 + [1] * 3 + [0] * 12)
        predicted = np.linspace(0, 1, len(labels))
        baseline = [score_metrics.VUS_ROC(labels, predicted), score_metrics.VUS_PR(labels, predicted)]
        evaluator = Evaluator([{"name": "VUS_ROC"}, {"name": "VUS_PR"}])
        with patch.object(score_metrics, "generate_curve", wraps=score_metrics.generate_curve) as curve:
            actual, log = evaluator.evaluate_with_log(labels, predicted)
            self.assertEqual(log, "")
            self.assertEqual(curve.call_count, 1)
            np.testing.assert_array_equal(actual, baseline)
            evaluator.evaluate(labels, predicted[::-1])
            self.assertEqual(curve.call_count, 2)
