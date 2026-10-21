"""Exact observations and training-only reference statistics from In1_5."""

import numpy as np
import torch
from torch import nn


class Observations(nn.Module):
    def __init__(self, win_size, scales, spans):
        super().__init__()
        self.win_size, self.scales, self.spans = win_size, tuple(scales), tuple(spans)
        self.register_buffer("time_index", torch.arange(win_size, dtype=torch.float64))
        for length in sorted(set(scales) | set(spans)):
            self.register_buffer("u_{}".format(length), torch.linspace(-1, 1, length, dtype=torch.float64))

    def coarse(self, windows):
        """[B,L,d] -> list of [B,d,L/m,2], ordered oldest to newest."""
        x = windows.to(torch.float64).transpose(1, 2)
        sums = torch.cat((torch.zeros_like(x[..., :1]), x.cumsum(dim=-1)), dim=-1)
        weighted = torch.cat((torch.zeros_like(x[..., :1]), (x * self.time_index).cumsum(dim=-1)), dim=-1)
        result = []
        for m in self.scales:
            u = getattr(self, "u_{}".format(m))
            ends = torch.arange(m, self.win_size + 1, m, device=x.device)
            starts = ends - m
            total = sums[..., ends] - sums[..., starts]
            local_weighted = weighted[..., ends] - weighted[..., starts] - starts * total
            level = total / m
            change = (2 * local_weighted / (m - 1) - total) / u.abs().sum()
            result.append(torch.stack((level, change), dim=-1))
        return result

    def statistics(self, windows, energy_floor):
        """Observed A/C/J; this function is never an input to the predictor."""
        x = windows.to(torch.float64).transpose(1, 2)
        outputs = []
        for index, h in enumerate(self.spans):
            block = x[..., -h:]
            u = getattr(self, "u_{}".format(h))
            slope = (block * u).sum(dim=-1, keepdim=True) / u.square().sum()
            residual = block - block.mean(dim=-1, keepdim=True) - slope * u
            energy = residual.square().sum(dim=-1)
            floor = energy_floor[:, index]
            denominator = energy + (h - 2) * floor
            amplitude = torch.log(energy / (h - 2) + floor)
            correlation = (residual[..., 1:] * residual[..., :-1]).sum(dim=-1) / denominator
            concentration = residual.pow(4).sum(dim=-1) / denominator.square()
            outputs.append(torch.stack((amplitude, correlation, concentration), dim=-1))
        return torch.stack(outputs, dim=2)


def _blocks(series, length, win_size, chunk_size):
    # Only endpoints of FULL L-windows contribute reference targets.
    view = np.lib.stride_tricks.sliding_window_view(series, length)[win_size - length:]
    for start in range(0, len(view), chunk_size):
        yield start, view[start:start + chunk_size]


def _coarse_moments(series, config):
    """Chunk-local prefix sums avoid both O(N*sum(m)) work and large time indices."""
    count = len(series) - config.win_size + 1
    mean, m2, seen = np.zeros((len(config.scales), 2)), np.zeros((len(config.scales), 2)), 0
    for start in range(0, count, config.reference_chunk_size):
        size = min(config.reference_chunk_size, count - start)
        segment = series[start:start + size + config.win_size - 1]
        sums = np.r_[0.0, np.cumsum(segment)]
        weighted = np.r_[0.0, np.cumsum(segment * np.arange(len(segment)))]
        ends = np.arange(config.win_size, config.win_size + size)
        targets = np.empty((size, len(config.scales), 2))
        for index, m in enumerate(config.scales):
            begins = ends - m
            total = sums[ends] - sums[begins]
            local_weighted = weighted[ends] - weighted[begins] - begins * total
            targets[:, index, 0] = total / m
            targets[:, index, 1] = (2 * local_weighted / (m - 1) - total) / np.abs(np.linspace(-1, 1, m)).sum()
        chunk_mean = targets.mean(axis=0)
        delta = chunk_mean - mean
        m2 += np.square(targets - chunk_mean).sum(axis=0) + delta ** 2 * seen * size / (seen + size)
        mean += delta * size / (seen + size)
        seen += size
    return mean, m2 / seen


def training_references(values, config):
    """Full-data exact quantiles with O(N) temporary memory per coordinate.

    values is the standardized TRAIN portion, never validation or test data.
    No overlapping raw windows or all-variable target tensors are materialized.
    """
    count = len(values) - config.win_size + 1
    channels, scales, spans = values.shape[1], config.scales, config.spans
    refs = {
        "mean_n": np.empty((channels, len(scales), 2)),
        "variance_n": np.empty((channels, len(scales), 2)),
        "mean_s": np.empty((channels, len(spans), 3)),
        "variance_s": np.empty((channels, len(spans), 3)),
        "energy_floor": np.empty((channels, len(spans))),
        "a_low": np.empty((channels, len(spans))),
        "a_high": np.empty((channels, len(spans))),
        "variance_p": values.var(axis=0, ddof=0),
    }
    for i in range(channels):
        series = values[:, i]
        refs["mean_n"][i], refs["variance_n"][i] = _coarse_moments(series, config)
        for index, h in enumerate(spans):
            u = np.linspace(-1, 1, h, dtype=np.float64)
            moments = np.empty((count, 3), dtype=np.float64)
            for start, block in _blocks(series, h, config.win_size, config.reference_chunk_size):
                slope = np.einsum("nh,h->n", block, u) / np.square(u).sum()
                residual = block - block.mean(axis=-1, keepdims=True) - slope[:, None] * u
                stop = start + len(block)
                moments[start:stop, 0] = np.square(residual).sum(axis=-1)
                moments[start:stop, 1] = (residual[:, 1:] * residual[:, :-1]).sum(axis=-1)
                moments[start:stop, 2] = np.power(residual, 4).sum(axis=-1)
            variance = moments[:, 0] / (h - 2)
            floor = max(config.epsilon_abs, config.eta_energy * np.quantile(variance, 0.5))
            denominator = moments[:, 0] + (h - 2) * floor
            targets = np.column_stack((np.log(variance + floor),
                                       moments[:, 1] / denominator,
                                       moments[:, 2] / np.square(denominator)))
            refs["energy_floor"][i, index] = floor
            refs["mean_s"][i, index] = targets.mean(axis=0)
            refs["variance_s"][i, index] = targets.var(axis=0, ddof=0)
            refs["a_low"][i, index], refs["a_high"][i, index] = np.quantile(
                targets[:, 0], [config.zeta_a, 1 - config.zeta_a])
    refs["fixed_s"] = refs["variance_s"] <= config.epsilon_v
    refs["fixed_values_s"] = refs["mean_s"].copy()
    refs["fixed_values_s"][..., 0] = np.clip(
        refs["fixed_values_s"][..., 0], refs["a_low"], refs["a_high"])
    return refs
