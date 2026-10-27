"""Length-only auxiliary protocol selection, independent of seeds and model selection."""

from dataclasses import dataclass
import math


PROTOCOL_SCALES = (1.0, 0.5, 0.25, 0.125)


def scaled_lengths(lengths, factor):
    """Floor event durations, bound them below by one, then remove duplicates."""
    return tuple(sorted({max(1, math.floor(length * factor)) for length in lengths}))


def _cycle(lengths, count=3):
    # Keep all multiple-event scenarios even when distinct durations collapse.
    return tuple(lengths[index % len(lengths)] for index in range(count))


@dataclass(frozen=True)
class AuxiliaryProtocol:
    s: float
    t0: int
    raw_history: int
    normal_prefix: int
    recovery_suffix: int
    separated_gap: int
    dense_gap: int
    recurrence_gap: int
    lambda_min: float
    lambda_max: float
    lambda_init: float
    training_lengths: tuple
    short_lengths: tuple
    long_lengths: tuple
    paired_lengths: tuple

    @property
    def event_start(self):
        return self.raw_history + self.normal_prefix

    def validation_layouts(self, index):
        short, long = _cycle(self.short_lengths), _cycle(self.long_lengths)
        pair_length = self.paired_lengths[index % len(self.paired_lengths)]
        return (
            ("short", short, (self.separated_gap,) * 2),
            ("long", long, (self.separated_gap,) * 2),
            ("dense", short, (self.dense_gap,) * 2),
            ("paired", (pair_length,) * 4,
             (self.recurrence_gap, self.separated_gap, self.recurrence_gap)),
        )

    def required_lengths(self):
        required = {}
        for index in range(3):
            for scenario, lengths, gaps in self.validation_layouts(index):
                length = self.event_start + sum(lengths) + sum(gaps) + self.recovery_suffix
                required[scenario] = max(required.get(scenario, 0), length)
        return self.event_start + max(self.training_lengths) + self.recovery_suffix, required


def make_protocol(config, raw_history, t0, factor):
    # The official defaults give D=89, 640, D+679 and D+1319.
    # Explicit architectural/smoke configurations keep their own unscaled D.
    recovery_updates = config.recovery_suffix - raw_history
    separation_updates = config.validation_gap - raw_history
    if min(recovery_updates, separation_updates) < 1:
        raise ValueError("recovery_suffix and validation_gap must exceed the unscaled raw history")
    training = config.event_lengths or (1, max(1, t0 // 4), t0, 4 * t0, 16 * t0)
    return AuxiliaryProtocol(
        s=factor, t0=t0, raw_history=raw_history,
        normal_prefix=math.ceil(config.normal_prefix * factor),
        recovery_suffix=raw_history + math.ceil(recovery_updates * factor),
        separated_gap=raw_history + math.ceil(separation_updates * factor),
        dense_gap=max(1, math.floor(2 * t0 * factor)),
        recurrence_gap=max(1, math.floor(max(1, t0 // 4) * factor)),
        lambda_min=config.lambda_min ** (1 / factor),
        lambda_max=config.lambda_max ** (1 / factor),
        lambda_init=config.lambda_init ** (1 / factor),
        training_lengths=scaled_lengths(training, factor),
        short_lengths=scaled_lengths((max(1, t0 // 4), max(1, t0 // 2), t0), factor),
        long_lengths=scaled_lengths((4 * t0, 8 * t0, 16 * t0), factor),
        paired_lengths=scaled_lengths((t0, 2 * t0, 4 * t0), factor),
    )


def select_protocol(config, raw_history, t0, fit_length, validation_length):
    trials = []
    for factor in PROTOCOL_SCALES:
        protocol = make_protocol(config, raw_history, t0, factor)
        train_required, validation_required = protocol.required_lengths()
        feasible = (fit_length >= train_required
                    and validation_length >= max(validation_required.values())
                    and protocol.event_start >= 16)
        trials.append(dict(s=factor, training_required=train_required,
                           validation_required=validation_required, feasible=feasible))
        if feasible:
            return protocol, trials
    raise ValueError(
        "ASEI shortest auxiliary protocol s=0.125 cannot fit all training/validation scenarios: "
        f"fit_length={fit_length}, validation_length={validation_length}, "
        f"minimum_requirements={trials[-1]}. No padding or entity concatenation is allowed."
    )
