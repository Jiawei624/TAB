"""Frozen empirical references and exact strict-exceedance thresholds."""

import numpy as np
import torch


def error_history(errors, order):
    return np.stack([errors[order - lag:len(errors) - lag]
                     for lag in range(1, order + 1)], axis=-1)


def upper_tail(values, reference):
    """Equation (9): equal reference values belong to the upper tail."""
    return (1 + len(reference) - np.searchsorted(reference, values, side="left")) / (len(reference) + 1)


class EvidenceReference:
    def __init__(self, bound):
        self.bound = bound

    def _g(self, errors, residuals):
        result = np.empty((*errors.shape[:2], 2), dtype=np.float32)
        for perspective, (values, reference) in enumerate(
                ((residuals, self.residuals), (errors, self.errors))):
            for channel in range(errors.shape[1]):
                branches = [upper_tail(values[:, channel, branch], reference[:, channel, branch]).astype(np.float32)
                            for branch in range(2)]
                result[:, channel, perspective] = np.minimum(
                    self.bound, np.maximum(-np.log(branches[0]), -np.log(branches[1])))
        return result

    def fit(self, errors, residuals, device="cpu"):
        if len(errors) == 0 or errors.shape != residuals.shape:
            raise ValueError("Independent reference requires aligned nonempty d/r arrays")
        if not np.isfinite(errors).all() or not np.isfinite(residuals).all():
            raise ValueError("Independent reference contains nonfinite values")
        self.errors = np.sort(errors.astype(np.float32), axis=0)
        self.residuals = np.sort(residuals.astype(np.float32), axis=0)
        self.center = np.zeros(errors.shape[1:], dtype=np.float32)
        self._tensor_device = None
        # Query the reference itself with the deployment arithmetic, then accumulate in FP64.
        g = self.transform_tensor(torch.as_tensor(errors, dtype=torch.float32, device=device),
                                  torch.as_tensor(residuals, dtype=torch.float32, device=device))
        self.center = g.to(torch.float64).mean(0).to(torch.float32).cpu().numpy()
        self._tensor_device = None
        self.to(device)
        return self

    def transform(self, errors, residuals):
        return self._g(errors, residuals) - self.center

    def to(self, device):
        key = str(device)
        if getattr(self, "_tensor_device", None) != key:
            self._tensor_device = key
            self._tensor_errors = torch.as_tensor(self.errors, device=device).permute(1, 2, 0).contiguous()
            self._tensor_residuals = torch.as_tensor(self.residuals, device=device).permute(1, 2, 0).contiguous()
            self._tensor_center = torch.as_tensor(self.center, device=device)
        return self

    def transform_tensor(self, errors, residuals):
        self.to(errors.device)
        perspectives = []
        for values, reference in ((residuals, self._tensor_residuals), (errors, self._tensor_errors)):
            query = values.to(torch.float32).permute(1, 2, 0).contiguous()
            index = torch.searchsorted(reference, query, right=False)
            probability = (len(self.errors) + 1 - index).to(torch.float32) / (len(self.errors) + 1)
            perspectives.append((-torch.log(probability)).amax(1).clamp_max(self.bound).transpose(0, 1))
        return torch.stack(perspectives, -1) - self._tensor_center


def local_features(evidence, window):
    """Valid causal rolling means, excluding incomplete initial history."""
    if len(evidence) < window:
        return np.empty((0, evidence.shape[1], 4), dtype=np.float32)
    sums = np.concatenate((np.zeros_like(evidence[:1], dtype=np.float64),
                           np.cumsum(evidence, axis=0, dtype=np.float64)), axis=0)
    means = (sums[window:] - sums[:-window]) / window
    return np.concatenate((evidence[window - 1:], means), axis=-1).astype(np.float32)


def local_features_tensor(evidence, window):
    single = evidence.ndim == 3
    values = evidence[None] if single else evidence
    batch, length, channels, _ = values.shape
    if length < window:
        result = evidence.new_empty((batch, 0, channels, 4))
    else:
        flattened = values.permute(0, 2, 3, 1).reshape(batch, channels * 2, length)
        means = torch.nn.functional.avg_pool1d(flattened, window, stride=1)
        means = means.reshape(batch, channels, 2, -1).permute(0, 3, 1, 2)
        result = torch.cat((values[:, window-1:], means), -1)
    return result[0] if single else result


def empirical_threshold(scores, alpha):
    """Sorted score at N-floor(alpha*N)-1; return h and actual exceedance count."""
    scores = np.asarray(scores)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Threshold calibration needs finite nonempty normal scores")
    if not 0 <= alpha < 1:
        raise ValueError("alpha must be in [0, 1)")
    count = len(scores)
    budget = int(np.floor(alpha * count))
    threshold = np.sort(scores)[count - budget - 1]
    return float(threshold), int(np.count_nonzero(scores > threshold))
