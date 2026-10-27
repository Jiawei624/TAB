"""Frozen empirical references and exact strict-exceedance thresholds."""

import numpy as np


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
        result = np.empty((*errors.shape[:2], 2), dtype=np.float64)
        for perspective, (values, reference) in enumerate(
                ((residuals, self.residuals), (errors, self.errors))):
            for channel in range(errors.shape[1]):
                branches = [upper_tail(values[:, channel, branch], reference[:, channel, branch])
                            for branch in range(2)]
                result[:, channel, perspective] = np.minimum(
                    self.bound, np.maximum(-np.log(branches[0]), -np.log(branches[1])))
        return result

    def fit(self, errors, residuals):
        if len(errors) == 0 or errors.shape != residuals.shape:
            raise ValueError("Independent reference requires aligned nonempty d/r arrays")
        self.errors = np.sort(errors.astype(np.float64), axis=0)
        self.residuals = np.sort(residuals.astype(np.float64), axis=0)
        self.center = self._g(errors, residuals).mean(0)
        return self

    def transform(self, errors, residuals):
        return self._g(errors, residuals) - self.center


def local_features(evidence, window):
    """Valid causal rolling means, excluding incomplete initial history."""
    if len(evidence) < window:
        return np.empty((0, evidence.shape[1], 4), dtype=np.float32)
    sums = np.concatenate((np.zeros_like(evidence[:1], dtype=np.float64),
                           np.cumsum(evidence, axis=0, dtype=np.float64)), axis=0)
    means = (sums[window:] - sums[:-window]) / window
    return np.concatenate((evidence[window - 1:], means), axis=-1).astype(np.float32)


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
