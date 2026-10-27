"""Chronological boundaries and frozen causal preprocessing."""

import numpy as np
from torch.utils.data import Dataset
import torch
from torch.nn import functional as F


def chronological_split(values, validation=None):
    n_model = len(values) if validation is not None else int(0.8 * len(values))
    f_end, c_end = int(0.8 * n_model), int(0.9 * n_model)
    valid = values[n_model:] if validation is None else validation
    middle = len(valid) // 2
    parts = dict(F=values[:f_end], Cfit=values[f_end:c_end], Rref=values[c_end:n_model],
                 Vsel=valid[:middle], Vcal=valid[middle:])
    bounds = dict(input_points=len(values), outer_training_end=n_model, predictor_end=f_end,
                  corrector_end=c_end, training_end=n_model, validation_points=len(valid),
                  validation_selection_points=middle, validation_calibration_points=len(valid)-middle,
                  validation_midpoint=middle if validation is not None else n_model+middle,
                  external_validation=validation is not None,
                  segment_lengths={name: len(part) for name, part in parts.items()})
    return parts, bounds


class CausalPreprocessor:
    def fit(self, values):
        finite = np.isfinite(values)
        self.all_missing = ~finite.any(0)
        self.initial = np.array([np.median(values[finite[:, i], i]) if finite[:, i].any() else 0.0
                                 for i in range(values.shape[1])], dtype=np.float64)
        filled, _ = self.fill(values)
        self.mean = filled.mean(0, dtype=np.float64)
        self.scale = filled.std(0, dtype=np.float64)
        self.constant = self.scale == 0
        self.scale[self.constant] = 1
        return self

    def fill(self, values, previous=None):
        """Forward fill each independent segment, or extend an explicit stream."""
        values = np.asarray(values, dtype=np.float64)
        last = self.initial.copy() if previous is None else previous.copy()
        filled = values.copy()
        for col in range(values.shape[1]):
            indices = np.where(np.isfinite(values[:, col]), np.arange(len(values)), -1)
            indices = np.maximum.accumulate(indices)
            if len(indices):
                filled[:, col] = np.where(indices >= 0, values[np.maximum(indices, 0), col], last[col])
                last[col] = filled[-1, col]
        return filled, last

    def transform(self, values, previous=None):
        filled, last = self.fill(values, previous)
        standardized = ((filled - self.mean) / self.scale).astype(np.float32)
        if not np.isfinite(standardized).all():
            raise FloatingPointError("ASEI standardized observations overflowed FP32")
        return standardized, last


class BlockWindowDataset(Dataset):
    """Every valid target appears once; blocks overlap only in left history."""
    def __init__(self, values, history, block_size):
        self.values = torch.as_tensor(np.asarray(values, dtype=np.float32))
        self.history, self.block_size = history, block_size
        self.target_count = len(values) - history
        if self.target_count < 1:
            raise ValueError("Predictor requires valid targets")

    def __len__(self):
        return (self.target_count + self.block_size - 1) // self.block_size

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        b = self.history + index * self.block_size
        q = min(self.block_size, len(self.values)-b)
        return self.values[b-self.history:b+q-1], self.values[b:b+q]


def collate_blocks(batch):
    q = max(len(target) for _, target in batch)
    histories, targets, masks = [], [], []
    for history, target in batch:
        missing = q - len(target)
        histories.append(F.pad(history, (0, 0, 0, missing)))
        targets.append(F.pad(target, (0, 0, 0, missing)))
        masks.append(torch.arange(q) < len(target))
    return torch.stack(histories), torch.stack(targets), torch.stack(masks)
