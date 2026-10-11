"""Compare TAB against the local author repository; no benchmark training is run.

Run: conda run -n codex python tests/test_anomaly_transformer_reference.py
The sibling Anomaly-Transformer checkout is only needed for these reference tests.
"""

import ast
import contextlib
import importlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn import metrics
import torch
from torch.utils.data import RandomSampler, SequentialSampler


TAB_ROOT = Path(__file__).resolve().parents[1]
AUTHOR_ROOT = TAB_ROOT.parent / "Anomaly-Transformer"
MODEL_ROOT = TAB_ROOT / "ts_benchmark/baselines/self_impl/Anomaly_trans"
sys.path.insert(0, str(AUTHOR_ROOT))
reference = importlib.import_module("solver")
reference_loaders = importlib.import_module("data_factory.data_loader")
reference_model = importlib.import_module("model.AnomalyTransformer")

# Load this adapter without importing TAB's unrelated optional baseline packages.
package = types.ModuleType("tab_anomaly")
package.__path__ = [str(MODEL_ROOT)]
sys.modules[package.__name__] = package
adapter = importlib.import_module("tab_anomaly.AnomalyTransformer")
tab_model = importlib.import_module("tab_anomaly.AnomalyTransformer_model")
tab_loaders = importlib.import_module("tab_anomaly.data_loader")
torch.set_num_threads(2)


def tiny_reference_model(**kwargs):
    return reference_model.AnomalyTransformer(d_model=16, n_heads=2, d_ff=16, **kwargs)


def tiny_tab_model(**kwargs):
    return tab_model.AnomalyTransformer_model(d_model=16, n_heads=2, d_ff=16, **kwargs)


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        rng = np.random.default_rng(31)
        self.train = rng.normal(size=(48, 3))
        self.test = rng.normal(size=(23, 3)) + 1.5
        self.labels = np.zeros((23, 1), dtype=int)
        self.labels[3:11] = 1
        for dataset in ("SMD", "MSL", "SMAP"):
            np.save(self.directory / (dataset + "_train.npy"), self.train)
            np.save(self.directory / (dataset + "_test.npy"), self.test)
            np.save(self.directory / (dataset + "_test_label.npy"), self.labels)
        for filename, values in (("train.csv", self.train), ("test.csv", self.test),
                                 ("test_label.csv", self.labels)):
            frame = pd.DataFrame(values)
            frame.insert(0, "timestamp", np.arange(len(values)))
            frame.to_csv(self.directory / filename, index=False)

    def test_network_and_kl_match(self):
        # Only neutralize hard-coded .cuda() in the author's code for this CPU test.
        with patch.object(torch.Tensor, "cuda", lambda tensor, *a, **kw: tensor):
            torch.manual_seed(19)
            original = tiny_reference_model(win_size=8, enc_in=3, c_out=3, e_layers=3)
            torch.manual_seed(19)
            current = tiny_tab_model(win_size=8, enc_in=3, c_out=3, e_layers=3)
            self.assertEqual(set(original.state_dict()), set(current.state_dict()))
            for key, tensor in original.state_dict().items():
                torch.testing.assert_close(tensor, current.state_dict()[key], rtol=0, atol=0)
            inputs = torch.randn(2, 8, 3)
            expected = original(inputs)
            actual = current(inputs)
            for reference_part, tab_part in zip(expected, actual):
                if isinstance(reference_part, list):
                    for first, second in zip(reference_part, tab_part):
                        torch.testing.assert_close(first, second, rtol=0, atol=0)
                else:
                    torch.testing.assert_close(reference_part, tab_part, rtol=0, atol=0)
            series, prior = expected[1], expected[2]
            normalized = prior[0] / prior[0].sum(-1, keepdim=True)
            torch.testing.assert_close(
                reference.my_kl_loss(series[0], normalized),
                adapter.my_kl_loss(series[0], normalized), rtol=0, atol=0,
            )

    def test_window_loaders_match_all_supported_datasets(self):
        for dataset, presets in adapter.DATASET_HYPER_PARAMS.items():
            for mode in ("train", "val", "test", "thre"):
                with self.subTest(dataset=dataset, mode=mode):
                    original = reference_loaders.get_loader_segment(
                        str(self.directory), batch_size=2, win_size=8,
                        mode=mode, dataset=dataset,
                    )
                    original_dataset = original.dataset
                    if mode == "train":
                        values = original_dataset.train
                    elif mode == "val":
                        values = original_dataset.val
                    else:
                        values = original_dataset.test
                    current = tab_loaders.get_loader(values, 2, 8, presets["step"], mode)
                    self.assertEqual(len(original_dataset), len(current.dataset))
                    for index in range(len(original_dataset)):
                        np.testing.assert_array_equal(
                            original_dataset[index][0], current.dataset[index][0]
                        )
                    sampler = RandomSampler if mode == "train" else SequentialSampler
                    self.assertIsInstance(current.sampler, sampler)

    def test_training_threshold_and_adjustment_match_author(self):
        config = dict(
            data_path=str(self.directory), dataset="SMD", batch_size=2,
            win_size=8, input_c=3, output_c=3, lr=1e-4, k=3,
            num_epochs=1, anormly_ratio=10.0, model_save_path=str(self.directory),
        )
        with contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(patch.object(torch.Tensor, "cuda", lambda tensor, *a, **kw: tensor))
            stack.enter_context(patch.object(np, "Inf", np.inf, create=True))
            stack.enter_context(patch.object(reference, "AnomalyTransformer", tiny_reference_model))
            stack.enter_context(patch.object(adapter, "AnomalyTransformer_model", tiny_tab_model))
            torch.manual_seed(41)
            original = reference.Solver(config)
            original.train()
            torch.manual_seed(41)
            current = adapter.AnomalyTransformer(
                dataset="SMD", win_size=8, batch_size=2, num_epochs=1,
                anomaly_ratio=10.0,
            )
            current.detect_fit_with_test(pd.DataFrame(self.train), None, pd.DataFrame(self.test))
            self.assertEqual(len(current.train_loader.dataset.data), len(self.train))
            self.assertEqual(len(current.test_loader.dataset.data), len(self.test))
            np.testing.assert_array_equal(current.scaler.mean_, original.train_loader.dataset.scaler.mean_)
            np.testing.assert_array_equal(current.scaler.scale_, original.train_loader.dataset.scaler.scale_)
            for key, tensor in original.model.state_dict().items():
                torch.testing.assert_close(tensor, current.model.state_dict()[key], rtol=1e-6, atol=1e-7)
            # Checkpoint losses must come from the test loader, not the declared val loader.
            expected_loss = original.vali(original.test_loader)
            actual_loss = current.vali(current.test_loader)
            np.testing.assert_allclose(actual_loss, expected_loss, rtol=1e-6, atol=1e-7)
            self.assertAlmostEqual(current.early_stopping.val_loss_min, actual_loss[0])
            self.assertAlmostEqual(current.early_stopping.val_loss2_min, actual_loss[1])

            percentile_calls = []
            metric_calls = []
            percentile = np.percentile
            precision_recall = metrics.precision_recall_fscore_support

            def capture_percentile(values, quantile):
                percentile_calls.append((values.copy(), quantile))
                return percentile(values, quantile)

            def capture_metrics(actual, predicted, **kwargs):
                metric_calls.append((actual.copy(), predicted.copy()))
                return precision_recall(actual, predicted, **kwargs)

            stack.enter_context(patch.object(np, "percentile", side_effect=capture_percentile))
            stack.enter_context(patch.object(metrics, "precision_recall_fscore_support", side_effect=capture_metrics))
            original.test()
            labels, scores = current.detect_label(pd.DataFrame(self.test))
            self.assertEqual(len(percentile_calls), 2)
            first_values, first_quantile = percentile_calls[0]
            second_values, second_quantile = percentile_calls[1]
            self.assertEqual(first_quantile, second_quantile)
            np.testing.assert_allclose(np.sort(first_values), np.sort(second_values), rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(scores, first_values[-len(scores):], rtol=1e-6, atol=1e-7)
            actual, adjusted = current.postprocess_predictions(self.labels.flatten(), labels[10.0])
            np.testing.assert_array_equal(actual, metric_calls[0][0])
            np.testing.assert_array_equal(adjusted, metric_calls[0][1])
            self.assertEqual(len(scores), 16)
            _, score_values = current.detect_score(pd.DataFrame(self.test))
            np.testing.assert_array_equal(scores, score_values)
            _, unchanged = current.postprocess_predictions(self.labels.flatten(), score_values)
            np.testing.assert_array_equal(unchanged, score_values)

    def test_presets_aliases_and_explicit_overrides(self):
        for dataset, presets in adapter.DATASET_HYPER_PARAMS.items():
            model = adapter.AnomalyTransformer()
            model.configure_for_series({"dataset_name": "M_SMD" if dataset == "SMD" else dataset})
            for key, value in presets.items():
                self.assertEqual(getattr(model.config, key), value)
        explicit = adapter.AnomalyTransformer(batch_size=7, num_epochs=2, anormly_ratio=2.0, step=3)
        explicit.configure_for_series({"dataset_name": "MSL"})
        self.assertEqual((explicit.config.batch_size, explicit.config.num_epochs,
                          explicit.config.anomaly_ratio, explicit.config.step), (7, 2, 2.0, 3))
        with self.assertRaisesRegex(ValueError, "test_data"):
            explicit.detect_fit(pd.DataFrame(self.train))
        values = pd.DataFrame([[np.nan, 1.0], [2.0, np.nan]])
        psm = adapter.AnomalyTransformer(dataset="PSM")
        np.testing.assert_array_equal(psm._data_values(values), np.nan_to_num(values.to_numpy()))

    def test_adjustment_including_reference_first_point_behavior(self):
        actual = np.array([1, 1, 1, 0, 1, 1, 1, 0])
        predicted = np.array([0, 0, 1, 0, 0, 1, 0, 0])
        adjusted = adapter.adjust_predictions(actual, predicted)
        # Solver.test's range(i, 0, -1) intentionally excludes point zero.
        np.testing.assert_array_equal(adjusted, [0, 1, 1, 0, 1, 1, 1, 0])
        np.testing.assert_array_equal(predicted, [0, 0, 1, 0, 0, 1, 0, 0])

    def test_strategy_hooks_preserve_other_models(self):
        # Execute the actual strategy method with isolated infrastructure; this
        # avoids requiring unrelated optional metric/backend packages in this test.
        source = ast.parse((TAB_ROOT / "ts_benchmark/evaluation/strategy/anomaly_detect.py").read_text(encoding="utf-8"))
        klass = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "AnomalyDetect")
        method = next(node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == "execute")
        module = ast.Module(body=[method], type_ignores=[])
        namespace = {
            "np": np, "pd": pd, "Any": object, "ModelFactory": object,
            "fix_all_random_seed": lambda seed: None,
            "DataPool": lambda: types.SimpleNamespace(get_pool=lambda: types.SimpleNamespace(
                get_series_meta_info=lambda name: {"dataset_name": "SMD"}
            )),
        }
        import base64
        import pickle
        import time
        import traceback
        namespace.update(base64=base64, pickle=pickle, time=time, traceback=traceback)
        exec(compile(module, "anomaly_detect.py", "exec"), namespace)

        class RecordingEvaluator:
            def evaluate_with_log(self, actual, predicted):
                self.actual, self.predicted = actual, predicted
                return [1.0], ""

        class LegacyModel:
            def detect_fit(self, train, labels):
                self.fit_length = len(train)

        class AlignedModel:
            def configure_for_series(self, metadata):
                self.metadata = metadata

            def detect_fit_with_test(self, train, labels, test):
                self.test_length = len(test)

            def postprocess_predictions(self, actual, predicted):
                return actual[:len(predicted)], predicted

        for model, expected_length in ((LegacyModel(), 23), (AlignedModel(), 16)):
            evaluator = RecordingEvaluator()
            strategy = types.SimpleNamespace(
                strategy_config={"seed": None}, evaluator=evaluator,
                split_data=lambda name: (pd.DataFrame(self.train), None,
                                         pd.DataFrame(self.test), pd.DataFrame(self.labels)),
                detect=lambda test: (np.ones(16), np.ones(16)),
            )
            with contextlib.redirect_stdout(io.StringIO()):
                results = namespace["execute"](strategy, "SMD.csv", lambda: model)
            self.assertEqual(len(results), 1)
            self.assertEqual(len(evaluator.actual), expected_length)
            self.assertEqual(len(evaluator.predicted), expected_length)
            if isinstance(model, LegacyModel):
                np.testing.assert_array_equal(evaluator.predicted[16:], np.zeros(7))
            else:
                self.assertEqual(model.test_length, 23)
                self.assertEqual(model.metadata["dataset_name"], "SMD")

    def test_model_specific_configs(self):
        for path in (TAB_ROOT / "config").glob("anomaly_transformer_*_config.json"):
            config = json.loads(path.read_text(encoding="utf-8"))
            self.assertIsNone(config["evaluation_config"]["strategy_args"]["seed"])
            if "label" in path.name:
                self.assertEqual(config["evaluation_config"]["metrics"],
                                 ["accuracy", "precision", "recall", "f_score"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
