"""Small real training tests; no benchmark dataset or optional model packages."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

from ts_benchmark.baselines.self_impl import AnomalyTransformer
from ts_benchmark.baselines.self_impl.Anomaly_trans.AnomalyTransformer import EarlyStopping
from ts_benchmark.baselines.self_impl.Anomaly_trans.AnomalyTransformer_model import AnomalyTransformer_model
from ts_benchmark.baselines.self_impl.Anomaly_trans.attn import AnomalyAttention


SMALL = dict(win_size=4, d_model=8, n_heads=2, d_ff=8, e_layers=1,
             batch_size=7, num_epochs=1, anomaly_ratio=[10, 30])


def data(length=32):
    t = np.arange(length, dtype=float)
    return pd.DataFrame({"a": np.sin(t / 3), "b": np.cos(t / 5)})


class AnomalyTransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cuda_patch = patch.object(torch.cuda, "is_available", return_value=False)
        cls.cuda_patch.start()
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.detector = AnomalyTransformer(**SMALL)
        cls.training_data = data()
        cls.detector.detect_fit(cls.training_data)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)
        cls.cuda_patch.stop()

    def test_actual_cpu_training_and_configured_architecture(self):
        self.assertEqual(len(self.detector.model.encoder.attn_layers), 1)
        self.assertEqual(self.detector.model.projection.in_features, 8)
        self.assertTrue(np.isfinite(self.detector.early_stopping.val_loss_min))
        self.assertTrue(all(torch.isfinite(p).all() for p in self.detector.model.parameters()))
        expected = self.training_data.iloc[:int(len(self.training_data) * 0.8)].mean()
        np.testing.assert_allclose(self.detector.scaler.mean_, expected)

    def test_all_real_timestamps_receive_finite_scores(self):
        for length in [1, 3, 4, 5, 7, 8, 9]:
            with self.subTest(length=length):
                scores, auxiliary = self.detector.detect_score(data(length))
                self.assertEqual(scores.shape, (length,))
                np.testing.assert_array_equal(scores, auxiliary)
                self.assertTrue(np.isfinite(scores).all())
                labels, energy = self.detector.detect_label(data(length))
                self.assertEqual(energy.shape, (length,))
                for predicted in labels.values():
                    self.assertEqual(predicted.shape, (length,))
                    self.assertTrue(np.isin(predicted, [0, 1]).all())

    def test_validation_does_not_build_gradient_graphs(self):
        grad_modes = []
        hook = self.detector.model.register_forward_pre_hook(
            lambda module, inputs: grad_modes.append(torch.is_grad_enabled())
        )
        try:
            self.detector.vali(self.detector.valid_loader)
        finally:
            hook.remove()
        self.assertTrue(grad_modes)
        self.assertFalse(any(grad_modes))

    def test_threshold_calibration_is_independent_of_test_data(self):
        real_percentile = np.percentile
        with patch("ts_benchmark.baselines.self_impl.Anomaly_trans.AnomalyTransformer.np.percentile",
                   wraps=real_percentile) as percentile:
            self.detector.detect_label(data(5))
            calibration_a = percentile.call_args_list[0].args[0].copy()
            percentile.reset_mock()
            self.detector.detect_label(data(5) * 1000)
            calibration_b = percentile.call_args_list[0].args[0].copy()
        np.testing.assert_array_equal(calibration_a, calibration_b)
        self.assertEqual(len(calibration_a), len(self.detector.valid_loader.dataset.data))

    def test_checkpoint_is_a_detached_cpu_copy_with_portable_keys(self):
        model = nn.DataParallel(nn.Linear(2, 1))
        stopper = EarlyStopping()
        stopper(1.0, 1.0, model)
        snapshot = stopper.check_point["weight"].clone()
        with torch.no_grad():
            model.module.weight.add_(5)
        np.testing.assert_array_equal(stopper.check_point["weight"], snapshot)
        self.assertTrue(all(v.device.type == "cpu" for v in stopper.check_point.values()))
        nn.Linear(2, 1).load_state_dict(stopper.check_point)
        self.assertFalse(any("distances" in key for key in self.detector.model.state_dict()))

    def test_attention_distance_buffer_moves_with_module(self):
        attention = AnomalyAttention(4, mask_flag=False, output_attention=True).to(torch.float64)
        self.assertIn("distances", dict(attention.named_buffers()))
        self.assertEqual(attention.distances.dtype, torch.float64)
        np.testing.assert_array_equal(attention.distances[0], [0, 1, 2, 3])

    def test_invalid_settings_and_too_short_training_fail_clearly(self):
        for kwargs in [dict(num_epochs=0), dict(e_layers=0), dict(batch_size=0),
                       dict(d_model=7), dict(anomaly_ratio=101), dict(lr=0)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AnomalyTransformer(**dict(SMALL, **kwargs))
        with self.assertRaisesRegex(ValueError, "two visible"):
            AnomalyTransformer(**SMALL, multi_gpu=True)
        with self.assertRaisesRegex(ValueError, "20% validation"):
            AnomalyTransformer(**SMALL).detect_fit(data(10))


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.device_count() >= 2,
                     "Requires two CUDA GPUs; run this file separately on the server")
class TwoGpuTests(unittest.TestCase):
    def test_parallel_forward_and_two_objective_gradients_match_single_gpu(self):
        single = AnomalyTransformer_model(win_size=4, enc_in=2, c_out=2,
                                         d_model=8, n_heads=2, d_ff=8, e_layers=1).cuda(0)
        replica = AnomalyTransformer_model(win_size=4, enc_in=2, c_out=2,
                                          d_model=8, n_heads=2, d_ff=8, e_layers=1).cuda(0)
        replica.load_state_dict(single.state_dict())
        parallel = nn.DataParallel(replica, device_ids=[0, 1])
        inputs = torch.randn(5, 4, 2, device="cuda:0")
        helper = AnomalyTransformer(**SMALL)
        outputs = []
        for model in [single, parallel]:
            output, series, prior, _ = model(inputs)
            outputs.append(output.detach())
            series_loss, prior_loss = helper._association_losses(series, prior)
            reconstruction = (output - inputs).square().mean()
            (reconstruction - 3 * series_loss).backward(retain_graph=True)
            (reconstruction + 3 * prior_loss).backward()
        torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-4, atol=1e-5)
        for first, second in zip(single.parameters(), replica.parameters()):
            if first.grad is not None:
                torch.testing.assert_close(first.grad, second.grad, rtol=1e-3, atol=1e-4)
        for name, buffer in replica.named_buffers():
            self.assertEqual(buffer.device.index, 0, name)

    def test_real_parallel_fit_checkpoint_reload_and_partial_tail(self):
        detector = AnomalyTransformer(**SMALL, multi_gpu=True)
        detector.detect_fit(data())
        self.assertIsInstance(detector.model, nn.DataParallel)
        self.assertGreaterEqual(len(detector.model.device_ids), 2)
        predictions, scores = detector.detect_label(data(5))
        self.assertEqual(scores.shape, (5,))
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(all(len(p) == 5 for p in predictions.values()))
        self.assertFalse(any(key.startswith("module.") for key in detector.early_stopping.check_point))


if __name__ == "__main__":
    unittest.main()
