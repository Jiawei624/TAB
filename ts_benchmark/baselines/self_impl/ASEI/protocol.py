"""Length-only auxiliary layout; the original observations and decay range are preserved."""
from dataclasses import dataclass


@dataclass
class AuxiliaryProtocol:
    t0: int
    raw_history: int
    normal_prefix: int
    recovery_suffix: int
    recurrence_gap: int
    length_cap: int
    training_lengths: tuple

    @property
    def event_start(self):
        return self.raw_history + self.normal_prefix

    @property
    def maximum_raw_length(self):
        return self.event_start + 2 * max(self.training_lengths) + self.recurrence_gap + self.recovery_suffix


def select_protocol(config, raw_history, scale, fit_length, validation_length):
    available = min(fit_length, validation_length)
    prefix = min(128, (available - raw_history) // 8)
    recovery = raw_history + 32
    gap = config.rolling_window
    cap = min(16 * scale, (available-raw_history-prefix-recovery-gap)//2)
    if prefix < 0 or cap < 1:
        raise ValueError("ASEI F/Vsel cannot contain auxiliary trajectories: "
                         f"M={available} D={raw_history} Npre={prefix} Nrec={recovery} Lcap={cap}")
    lengths = tuple(sorted({min(cap, length) for length in (1, max(1, scale//4), scale, 4*scale, 16*scale)}))
    protocol = AuxiliaryProtocol(scale, raw_history, prefix, recovery, gap, cap, lengths)
    return protocol, dict(available=available, feasible=True, maximum_raw_length=protocol.maximum_raw_length)