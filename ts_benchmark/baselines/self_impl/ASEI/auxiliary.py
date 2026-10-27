"""Paired perturbations and event-balanced objectives from the ASEI specification."""

from dataclasses import dataclass, replace

import numpy as np
import torch
from torch.nn import functional as F

from .protocol import make_protocol


def characteristic_scale(values):
    """Median absolute channel autocorrelation; nearest power of two, lower tie."""
    values = np.asarray(values, dtype=np.float64)
    values = values[:, np.std(values, axis=0) > 0]
    if values.shape[1] == 0:
        return 8
    crossing = None
    for lag in range(1, min(128, len(values) - 2) + 1):
        left, right = values[lag:], values[:-lag]
        left, right = left - left.mean(0), right - right.mean(0)
        denominator = np.sqrt((left * left).sum(0) * (right * right).sum(0))
        acf = np.divide((left * right).sum(0), denominator,
                        out=np.zeros(values.shape[1]), where=denominator > 0)
        if np.median(np.abs(acf)) < np.exp(-1):
            crossing = lag
            break
    if crossing is None:
        return 32
    nearest = min([2 ** power for power in range(8)], key=lambda value: (abs(value - crossing), value))
    return int(np.clip(nearest, 4, 32))


@dataclass
class Task:
    start: int
    length: int
    events: list  # Half-open intervals on the complete raw task timeline.
    family: int
    subtype: int
    seed: int
    severity: float = None
    scenario: str = "train"
    base_attempts: int = 0
    hard: bool = False

    def __post_init__(self):
        self.events = [tuple(event) for event in self.events]


class IneffectivePerturbation(ValueError):
    pass


class TaskSampler:
    def __init__(self, values, raw_history, scale, config, protocol=None):
        self.values, self.raw_history, self.scale, self.config = values, raw_history, scale, config
        self.protocol = protocol or make_protocol(config, raw_history, scale, 1.0)
        self.lengths = self.protocol.training_lengths
        self.event_start = self.protocol.event_start
        self._shift_masks = {}
        self._candidate_cache = {}
        self.hardness_prefix = None

    def _make(self, lengths, gaps, family, subtype, rng, severity=None, scenario="train"):
        cursor, events = self.event_start, []
        for index, length in enumerate(lengths):
            events.append((cursor, cursor + length))
            cursor += length
            if index < len(gaps):
                cursor += gaps[index]
        total = cursor + self.protocol.recovery_suffix
        if total > len(self.values):
            raise ValueError(
                f"ASEI {scenario} auxiliary task needs {total} contiguous normal points; "
                f"only {len(self.values)} available for s={self.protocol.s}. "
                "No padding or entity concatenation is allowed."
            )
        return Task(int(rng.integers(len(self.values) - total + 1)), total, events,
                    family, subtype, int(rng.integers(2 ** 31)), severity, scenario)

    def training(self, rng):
        return self._make([int(rng.choice(self.lengths))], [], int(rng.integers(3)),
                          int(rng.integers(2)), rng)

    def validation(self):
        tasks = []
        for seed in self.config.validation_seeds:
            rng = np.random.default_rng(seed)
            for family in range(3):
                amplitudes = ([0.3, 0.8, 2] if family == 0 else
                              [0.3, 0.8, 1.5] if family == 1 else [1, 4, 16])
                for severity_index, amplitude in enumerate(amplitudes):
                    for scenario, lengths, gaps in self.protocol.validation_layouts(severity_index):
                        task = self._make(lengths, gaps, family, severity_index % 2, rng, amplitude, scenario)
                        tasks.append(self.resolve(task)[0])
        return tasks

    def materialize(self, task):
        _, normal, changed = self.resolve(task)
        return normal, changed

    def _shift_mask(self, delay):
        if delay not in self._shift_masks:
            mask = np.zeros(len(self.values), dtype=bool)
            for begin in range(delay, len(self.values), 8192):
                end = min(len(self.values), begin + 8192)
                mask[begin:end] = np.any(self.values[begin:end] != self.values[begin - delay:end - delay], axis=1)
            self._shift_masks[delay] = mask
        return self._shift_masks[delay]

    def _candidate_starts(self, task):
        key = (task.length, tuple((int(begin), int(end)) for begin, end in task.events), task.severity)
        if key not in self._candidate_cache:
            count = len(self.values) - task.length + 1
            possible = np.ones(count, dtype=bool)
            delays = [int(task.severity)] if task.severity is not None else [1, 2, 4, 8, 16]
            changed = np.logical_or.reduce([self._shift_mask(delay) for delay in delays])
            for begin, end in task.events:
                possible &= changed[begin:begin + count]
                possible &= changed[end - 1:end - 1 + count]
            self._candidate_cache[key] = np.flatnonzero(possible)
        return self._candidate_cache[key]

    def resolve(self, task):
        """Resample the base, preserving family, layout and a deterministic task-local RNG."""
        try:
            normal, changed = self._materialize_fixed(task)
            return task, normal, changed
        except IneffectivePerturbation:
            pass
        candidates = (self._candidate_starts(task) if task.family == 2
                      else np.arange(len(self.values) - task.length + 1))
        candidates = candidates[candidates != task.start]
        if not len(candidates):
            raise IneffectivePerturbation(
                f"No effective auxiliary base trajectory: family={task.family}, scenario={task.scenario}, "
                "no eligible start changes every event's first and last observation"
            )
        rng = np.random.default_rng(np.random.SeedSequence([task.seed, 104729]))
        attempts = self.config.perturbation_base_attempts
        if task.hard and self.hardness_prefix is not None:
            pool = rng.choice(candidates, min(len(candidates), 4 * attempts), replace=False)
            prefix = self.hardness_prefix
            difficulty = ((prefix[pool + task.length] - prefix[pool + self.raw_history])
                          / (task.length - self.raw_history))
            candidates = pool[np.argsort(-difficulty, kind="stable")[:max(1, int(np.ceil(len(pool) / 4)))]]
        order = rng.choice(candidates, min(attempts, len(candidates)), replace=False)
        for index, start in enumerate(order, 1):
            resolved = replace(task, start=int(start), seed=int(rng.integers(2 ** 31)),
                               base_attempts=task.base_attempts + index)
            try:
                normal, changed = self._materialize_fixed(resolved)
                return resolved, normal, changed
            except IneffectivePerturbation:
                continue
        raise IneffectivePerturbation(
            f"Cannot create an effective auxiliary perturbation after {len(order)} new base trajectories; "
            f"family={task.family}, scenario={task.scenario}. No invalid task is used."
        )

    def _materialize_fixed(self, task):
        normal = self.values[task.start:task.start + task.length].copy()
        rng = np.random.default_rng(task.seed)
        for _ in range(32):
            changed = normal.copy()
            shared_channels = None
            for event_index, (begin, end) in enumerate(task.events):
                length = end - begin
                if task.scenario == "paired" and event_index % 2 and shared_channels is not None:
                    channels = shared_channels
                else:
                    count = int(rng.integers(1, max(1, int(np.ceil(0.1 * normal.shape[1]))) + 1))
                    channels = rng.choice(normal.shape[1], size=count, replace=False)
                    shared_channels = channels
                onsets = rng.integers(0, min(16, length - 1) + 1, size=len(channels))
                onsets[0] = 0
                if task.family == 0:
                    magnitude = (task.severity if task.severity is not None else
                                 np.exp(rng.uniform(np.log(0.2), np.log(3))))
                    magnitude *= rng.choice([-1, 1])
                elif task.family == 1:
                    magnitude = (task.severity if task.severity is not None else
                                 np.exp(rng.uniform(np.log(0.2), np.log(2))))
                    shared_noise = rng.normal(0, magnitude, length)
                else:
                    delay = int(task.severity if task.severity is not None else rng.choice([1, 2, 4, 8, 16]))
                for channel, onset in zip(channels, onsets):
                    start = begin + int(onset)
                    if task.family == 0:
                        shape = (np.ones(end - start) if task.subtype == 0 else
                                 np.arange(1, end - start + 1) / (end - start))
                        changed[start:end, channel] += magnitude * shape
                    elif task.family == 1:
                        noise = (rng.normal(0, magnitude, end - start) if task.subtype == 0
                                 else shared_noise[int(onset):])
                        changed[start:end, channel] += noise
                    else:
                        # Always copy past observations of this same original channel.
                        indices = np.arange(start, end) - delay
                        if indices[0] < 0:
                            raise ValueError("Auxiliary prefix must contain the largest shift delay (16)")
                        changed[start:end, channel] = normal[indices, channel]
            if all(np.any(changed[begin] != normal[begin]) and
                   np.any(changed[end - 1] != normal[end - 1]) for begin, end in task.events):
                return normal, changed
            if task.family == 2 and not np.any(np.ptp(normal, axis=0) > 0):
                break
        raise IneffectivePerturbation("This base trajectory cannot produce an effective perturbation")


def prepare_objective(task, raw_history, rank_pairs, rng):
    """Precompute masks and rank pairs once, identically for every candidate."""
    length = task.length - raw_history
    positive = np.zeros(length, dtype=np.float32)
    events = [(begin - raw_history, end - raw_history) for begin, end in task.events]
    for begin, end in events:
        positive[begin:end] = 1 / ((end - begin) * len(events))
    negative = np.zeros((2, length), dtype=np.float32)
    negative[0, events[0][0]:] = 1
    negative[1, events[0][0]:] = 2
    for begin, end in events:
        negative[1, begin:end] = 0
    # Everything between events is a recovery segment on the perturbed trajectory.
    probability = negative.reshape(-1).astype(np.float64)
    probability /= probability.sum()
    pos_indices, neg_indices = [], []
    for begin, end in events:
        pos_indices.extend(rng.integers(begin, end, rank_pairs))
        neg_indices.extend(rng.choice(len(probability), rank_pairs, p=probability))
    return dict(positive=positive, negative=negative,
                event_count=len(events),
                rank_positive=np.asarray(pos_indices), rank_negative=np.asarray(neg_indices))


def objective(scores, logits, metadata, margin=1, rank_temperature=1):
    """Each base trajectory has the same total weight regardless of event length."""
    positive_losses, negative_losses, negative_weights, ranking = [], [], [], []
    event_count = 0
    for index, item in enumerate(metadata):
        positive = torch.as_tensor(item["positive"], device=scores.device)
        negative = torch.as_tensor(item["negative"], device=scores.device)
        length = len(positive)
        current, current_logits = scores[index, :, :length], logits[index, :, :length]
        positive_losses.append((F.softplus(-current_logits[1]) * positive).sum() * item["event_count"])
        negative_losses.append((F.softplus(current_logits) * negative).sum())
        negative_weights.append(negative.sum())
        event_count += item["event_count"]
        pos = torch.as_tensor(item["rank_positive"], device=scores.device, dtype=torch.long)
        neg = torch.as_tensor(item["rank_negative"], device=scores.device, dtype=torch.long)
        ranking.append(F.softplus((margin - (current[1, pos] - current.reshape(-1)[neg]))
                                  / rank_temperature))
    detection = (0.5 * torch.stack(positive_losses).sum() / event_count
                 + 0.5 * torch.stack(negative_losses).sum() / torch.stack(negative_weights).sum())
    return detection, torch.cat(ranking).mean()


def objective_to_device(metadata, device):
    return [{key: torch.as_tensor(value, device=device) if isinstance(value, np.ndarray) else value
             for key, value in item.items()} for item in metadata]
