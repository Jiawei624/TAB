"""Trainable ASEI modules. Dimension zero always denotes independent samples."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .recurrence import integrate


def unwrap(module):
    return module


def cpu_state(module):
    return {key: value.detach().cpu().clone()
            for key, value in unwrap(module).state_dict().items()}


@torch.no_grad()
def project_l1_(values, radius):
    """Euclidean projection onto an L1 ball, independently along the last axis."""
    magnitude = values.abs()
    ordered = magnitude.sort(dim=-1, descending=True).values
    cumulative = ordered.cumsum(-1) - radius
    indices = torch.arange(1, values.shape[-1] + 1, device=values.device)
    active = ordered > cumulative / indices
    count = active.sum(-1, keepdim=True).clamp_min(1)
    threshold = (cumulative.gather(-1, count - 1) / count).clamp_min(0)
    values.copy_(values.sign() * (magnitude - threshold).clamp_min(0))


def lag_graph(values, lags, neighbors):
    """All aligned observations; ties: correlation, smaller lag, smaller source."""
    values = np.asarray(values, dtype=np.float64)
    channels = values.shape[1]
    best = np.zeros((channels, channels), dtype=np.float64)
    selected_lag = np.ones((channels, channels), dtype=np.int64)
    for lag in sorted(lags):
        target, source = values[lag:], values[:-lag]
        target = target - target.mean(0)
        source = source - source.mean(0)
        denominator = np.sqrt((target * target).sum(0)[:, None]
                              * (source * source).sum(0)[None, :])
        correlation = np.divide(target.T @ source, denominator,
                                out=np.zeros_like(best), where=denominator > 0)
        correlation = np.clip(np.abs(correlation), 0, 1)
        np.fill_diagonal(correlation, 0)
        improved = correlation > best
        selected_lag[improved] = lag
        best[improved] = correlation[improved]
    # A padded empty slot keeps one-channel and empty-neighborhood cases valid.
    width = max(1, min(neighbors, channels - 1))
    sources = np.zeros((channels, width), dtype=np.int64)
    delays = np.ones_like(sources)
    mask = np.zeros_like(sources, dtype=bool)
    for target in range(channels):
        candidates = [source for source in range(channels)
                      if source != target and best[target, source] > 0]
        candidates.sort(key=lambda source: (-best[target, source],
                                            selected_lag[target, source], source))
        for slot, source in enumerate(candidates[:neighbors]):
            sources[target, slot] = source
            delays[target, slot] = selected_lag[target, source]
            mask[target, slot] = True
    return sources, delays, mask


class CausalBlock(nn.Module):
    def __init__(self, hidden, kernel, dilation):
        super().__init__()
        self.left = (kernel - 1) * dilation
        self.conv = nn.Conv1d(hidden, hidden, kernel, dilation=dilation)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, values):
        values = self.conv(F.pad(values, (self.left, 0)))
        return F.gelu(self.norm(values.transpose(1, 2))).transpose(1, 2)


def temporal_layout(kernel, dilations, window=None):
    """Keep the base TCN by default; build an exact causal receptive field on request."""
    layout, receptive_field = [], 1
    for dilation in dilations:
        increase = (kernel - 1) * dilation
        if window is not None and receptive_field + increase > window:
            break
        layout.append((kernel, dilation))
        receptive_field += increase
    if window is not None:
        while receptive_field < window:
            # A two-tap convolution expands the field without introducing gaps.
            increase = min(receptive_field, window - receptive_field)
            layout.append((2, increase))
            receptive_field += increase
    return tuple(layout), receptive_field


class Predictor(nn.Module):
    def __init__(self, channels, sources, delays, mask, hidden=32,
                 embedding=8, kernel=3, dilations=(1, 2, 4, 8, 16), win_size=None):
        super().__init__()
        self.channels = channels
        self.embedding = nn.Embedding(channels, embedding)
        self.input_projection = nn.Linear(1 + embedding, hidden)
        self.temporal_layout, self.receptive_field = temporal_layout(kernel, dilations, win_size)
        self.encoder = nn.Sequential(*(CausalBlock(hidden, width, dilation)
                                       for width, dilation in self.temporal_layout))
        self.graph_projection = nn.Linear(hidden, hidden, bias=False)
        self.self_head = nn.Sequential(nn.Linear(hidden + embedding, hidden),
                                       nn.GELU(), nn.Linear(hidden, 1))
        self.rel_head = nn.Sequential(nn.Linear(hidden + embedding, hidden),
                                      nn.GELU(), nn.Linear(hidden, 1))
        self.register_buffer("sources", torch.as_tensor(sources, dtype=torch.long))
        self.register_buffer("delays", torch.as_tensor(delays, dtype=torch.long))
        self.register_buffer("edge_mask", torch.as_tensor(mask, dtype=torch.bool))
        self.edge_logits = nn.Parameter(torch.zeros(sources.shape))

    def forward(self, history, output_steps=1):
        # history contains observations strictly before the last target.
        batch, length, channels = history.shape
        embedding = self.embedding.weight[None, None].expand(batch, length, -1, -1)
        features = self.input_projection(torch.cat((history[..., None], embedding), -1))
        features = features.permute(0, 2, 3, 1).reshape(batch * channels, -1, length)
        encoded = self.encoder(features).reshape(batch, channels, -1, length)
        encoded = encoded.permute(0, 3, 1, 2)  # B, time, channel, hidden
        targets = torch.arange(length - output_steps + 1, length + 1,
                               device=history.device)
        own = encoded[:, targets - 1]
        source_times = targets[:, None, None] - self.delays[None]
        related = encoded[:, source_times, self.sources[None]]
        weights = torch.softmax(self.edge_logits.masked_fill(~self.edge_mask, -1e9), -1)
        weights = weights * self.edge_mask
        pooled = (self.graph_projection(related) * weights[None, None, ..., None]).sum(-2)
        head_embedding = self.embedding.weight[None, None].expand(batch, output_steps, -1, -1)
        own = self.self_head(torch.cat((own, head_embedding), -1)).squeeze(-1)
        related = self.rel_head(torch.cat((pooled, head_embedding), -1)).squeeze(-1)
        return torch.stack((own, related), -1)


class ErrorCorrector(nn.Module):
    def __init__(self, channels, order, rho):
        super().__init__()
        self.rho = rho
        self.bias = nn.Parameter(torch.zeros(channels, 2, dtype=torch.float64))
        self.beta = nn.Parameter(torch.zeros(channels, 2, order, dtype=torch.float64))
        self.mlp = nn.Sequential(nn.Linear(order, 8), nn.GELU(), nn.Linear(8, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.register_buffer("log_mean", torch.zeros(channels, 2, order))
        self.register_buffer("log_scale", torch.ones(channels, 2, order))
        self.register_buffer("eta", torch.zeros(channels, 2))

    def forward(self, history, nonlinear=True):
        predicted = self.bias + (history.to(self.beta.dtype) * self.beta).sum(-1)
        if nonlinear:
            normalized = (torch.log1p(history.to(self.log_mean.dtype)) - self.log_mean) / self.log_scale
            predicted = predicted + self.eta * torch.tanh(self.mlp(normalized).squeeze(-1))
        return predicted

    @torch.no_grad()
    def project_(self):
        project_l1_(self.beta, self.rho)


class EvidenceScorer(nn.Module):
    def __init__(self, bound, omega_min, lambda_min, lambda_max, lambda_init,
                 kappa, temperature, recurrence_backend="scripted"):
        super().__init__()
        self.bound, self.omega_min = bound, omega_min
        self.lambda_min, self.lambda_max = lambda_min, lambda_max
        self.kappa, self.temperature = kappa, temperature
        self.recurrence_backend = recurrence_backend
        self.gate_weight = nn.Parameter(torch.zeros(4))
        self.gate_bias = nn.Parameter(torch.zeros(()))
        probability = (lambda_init - lambda_min) / (lambda_max - lambda_min)
        self.decay_logit = nn.Parameter(torch.tensor(math.log(probability / (1 - probability))))
        self.training_bias = nn.Parameter(torch.zeros(()))

    @property
    def decay(self):
        return self.lambda_min + (self.lambda_max - self.lambda_min) * self.decay_logit.sigmoid()

    def integrate(self, features, state=None):
        # features: independent sequences, time, channels, [ec, es, mean_ec, mean_es].
        omega = self.omega_min + (1 - 2 * self.omega_min) * torch.sigmoid(
            (features / self.bound * self.gate_weight).sum(-1) + self.gate_bias)
        evidence = omega * features[..., 0] + (1 - omega) * features[..., 1]
        if state is None:
            state = evidence.new_zeros((evidence.shape[0], evidence.shape[2]))
        return integrate(evidence, self.decay, state, self.kappa, self.recurrence_backend)

    def forward(self, paired_features):
        # Each original/perturbed pair shares one independently initialized state.
        batch, pair, length, channels, _ = paired_features.shape
        scores, _ = self.integrate(paired_features.reshape(batch * pair, length, channels, 4))
        scores = scores.reshape(batch, pair, length)
        return scores, (scores - self.training_bias) / self.temperature

    @torch.no_grad()
    def project_(self):
        project_l1_(self.gate_weight, 2)
        self.gate_bias.clamp_(-1, 1)
