"""Equivalence and ordering checks for A1, A2b and A3; not benchmark evidence."""

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from ts_benchmark.baselines.self_impl.dynamic_neighborhood_stat import DynamicNeighborhoodStat
from ts_benchmark.baselines.self_impl.dynamic_neighborhood_stat.DynamicNeighborhoodStat import WindowDataset
from ts_benchmark.baselines.self_impl.dynamic_neighborhood_stat.config import Config
from ts_benchmark.baselines.self_impl.dynamic_neighborhood_stat.features import training_references
from ts_benchmark.baselines.self_impl.dynamic_neighborhood_stat.model import CausalEncoder, DualPathNetwork


class OptimizationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(3)

    def test_pruned_encoders_match_dense_outputs_inputs_and_parameter_gradients(self):
        cases = [(n, (n - 1,), 32) for n in (1, 2, 4, 8, 16, 32)]
        cases += [(length - 1, (length - length // 8 - 1, length - length // 4 - 1, length - 2), length - 1)
                  for length in (64, 128, 256)]
        for length, readouts, receptive_field in cases:
            with self.subTest(length=length, readouts=readouts):
                torch.manual_seed(length)
                dense = CausalEncoder(5, 8, receptive_field, 0).double()
                pruned = copy.deepcopy(dense)
                pruned.add_readouts(length, readouts)
                self.assertEqual(set(dense.state_dict()), set(pruned.state_dict()))
                x = torch.randn(2, length, 5, dtype=torch.float64, requires_grad=True)
                selected_x = x.detach().clone().requires_grad_()
                expected, actual = dense(x)[:, readouts], pruned(selected_x, readouts=readouts)
                torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-11)
                probe = torch.randn_like(actual)
                (expected * probe).sum().backward()
                (actual * probe).sum().backward()
                torch.testing.assert_close(selected_x.grad, x.grad, rtol=2e-9, atol=2e-10)
                for name, parameter in dense.named_parameters():
                    torch.testing.assert_close(dict(pruned.named_parameters())[name].grad, parameter.grad,
                                               rtol=2e-9, atol=2e-10, msg=lambda message: name + ": " + message)

    def test_static_readout_counts(self):
        config = Config()
        values = np.random.default_rng(42).normal(size=(160, 3))
        model = DualPathNetwork(3, config, training_references(values, config))
        neighbor = list(model.encoder_n.readout_plans.values())
        own = list(model.encoder_s.readout_plans.values())
        self.assertEqual(sum(plan.dense_positions for plan in neighbor), 504)
        self.assertEqual(sum(plan.computed_positions for plan in neighbor), 186)
        self.assertEqual(own[0].dense_positions, 1524)
        self.assertEqual(own[0].computed_positions, 600)

    def test_factorized_routing_matches_double_precision_outputs_and_gradients(self):
        config = Config(dropout=0, target_chunk_size=2)
        values = np.random.default_rng(13).normal(size=(160, 5))
        torch.manual_seed(19)
        dense = DualPathNetwork(5, config, training_references(values, config)).double()
        optimized = copy.deepcopy(dense)
        encoded = torch.randn(2, 5, len(config.scales), config.hidden_n, dtype=torch.float64, requires_grad=True)
        context = torch.randn(2, 5, config.context_dim, dtype=torch.float64, requires_grad=True)
        selected_encoded = encoded.detach().clone().requires_grad_()
        selected_context = context.detach().clone().requires_grad_()
        projections = optimized._route_projections(selected_encoded, selected_context)
        expected_chunks, actual_chunks = [], []
        for start in range(0, 5, 2):
            ids = torch.arange(start, min(start + 2, 5))
            expected_chunks.append(dense._route(encoded, context, ids))
            actual_chunks.append(optimized._route_projected(*projections, ids))
        expected = tuple(torch.cat(items, dim=1) for items in zip(*expected_chunks))
        actual = tuple(torch.cat(items, dim=1) for items in zip(*actual_chunks))
        probes = tuple(torch.randn_like(item) for item in expected)
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, rtol=2e-10, atol=2e-11)
        sum((item * probe).sum() for item, probe in zip(expected, probes)).backward()
        sum((item * probe).sum() for item, probe in zip(actual, probes)).backward()
        for left, right in ((selected_encoded.grad, encoded.grad), (selected_context.grad, context.grad)):
            torch.testing.assert_close(left, right, rtol=2e-9, atol=2e-10)
        for name, parameter in dense.named_parameters():
            actual_grad = dict(optimized.named_parameters())[name].grad
            if parameter.grad is None:
                self.assertIsNone(actual_grad, name)
            else:
                torch.testing.assert_close(actual_grad, parameter.grad, rtol=2e-9, atol=2e-10,
                                           msg=lambda message: name + ": " + message)

    def test_a1_a2b_and_combination_match_full_network_and_gradients(self):
        rng = np.random.default_rng(37)
        for channels, length in ((2, 64), (5, 128)):
            values = rng.normal(size=(length + 70, channels))
            config = Config(win_size=length, dropout=0, target_chunk_size=2,
                            factorized_routing=False, prune_tcn=False)
            refs = training_references(values, config)
            for factorized, pruned in ((True, False), (False, True), (True, True)):
                with self.subTest(channels=channels, length=length, factorized=factorized, pruned=pruned):
                    torch.manual_seed(12)
                    dense = DualPathNetwork(channels, config, refs).train()
                    optimized_config = Config(**dict(config.to_dict(), factorized_routing=factorized, prune_tcn=pruned))
                    optimized = DualPathNetwork(channels, optimized_config, refs).train()
                    optimized.load_state_dict(dense.state_dict(), strict=True)
                    windows = torch.from_numpy(np.stack((values[:length], values[1:length + 1]))).requires_grad_()
                    optimized_windows = windows.detach().clone().requires_grad_()
                    expected, actual = dense(windows), optimized(optimized_windows)
                    for key in expected:
                        torch.testing.assert_close(actual[key], expected[key], rtol=2e-5, atol=2e-6, msg=key)
                    expected_losses, actual_losses = dense.losses(windows, expected), optimized.losses(optimized_windows, actual)
                    for key in expected_losses:
                        torch.testing.assert_close(actual_losses[key], expected_losses[key], rtol=2e-5, atol=2e-6, msg=key)
                    expected_losses["loss"].backward()
                    actual_losses["loss"].backward()
                    torch.testing.assert_close(optimized_windows.grad, windows.grad, rtol=3e-4, atol=3e-6)
                    for name, parameter in dense.named_parameters():
                        actual_grad = dict(optimized.named_parameters())[name].grad
                        # Float32 convolution and compact linear reductions round differently;
                        # double-precision component tests above check structural equivalence.
                        torch.testing.assert_close(actual_grad, parameter.grad, rtol=3e-4, atol=3e-5,
                                                   msg=lambda message: name + ": " + message)
                        error = torch.linalg.vector_norm(actual_grad - parameter.grad).item()
                        scale = torch.linalg.vector_norm(parameter.grad).item()
                        self.assertLessEqual(error, 1e-6 + 1e-5 * scale, name)

    def test_batch_gather_preserves_two_epoch_shuffle_order_and_final_batch(self):
        values = np.asfortranarray(np.arange(403 * 3, dtype=np.float64).reshape(403, 3))
        detector = DynamicNeighborhoodStat(win_size=64, batch_size=17, seed=31, device="cpu")
        detector.device_ = torch.device("cpu")
        for training in (False, True):
            with self.subTest(training=training):
                gathered = detector._loader(values, training)
                original = DataLoader(WindowDataset(values, 64), batch_size=17, shuffle=training,
                                      drop_last=False, generator=torch.Generator().manual_seed(31))
                self.assertTrue(gathered.dataset.values.flags.c_contiguous)
                for _ in range(2):
                    actual, expected = list(gathered), list(original)
                    self.assertEqual(len(actual), len(expected))
                    self.assertEqual(sum(len(batch) for batch in actual), len(values) - 63)
                    for left, right in zip(actual, expected):
                        self.assertTrue(left.is_contiguous())
                        torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_execution_flags_preserve_old_checkpoint_identity(self):
        reference = Config(factorized_routing=False, prune_tcn=False)
        optimized = Config()
        self.assertEqual(reference.model_identity(), optimized.model_identity())
        old = optimized.to_dict()
        old.pop("factorized_routing")
        old.pop("prune_tcn")
        self.assertEqual(Config(**old).model_identity(), optimized.model_identity())


if __name__ == "__main__":
    unittest.main(verbosity=2)
