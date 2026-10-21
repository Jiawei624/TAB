"""The complete main-method configuration (no ablation modes)."""

import math
from dataclasses import asdict, dataclass


@dataclass
class Config:
    win_size: int = 128
    train_ratio: float = 0.8
    hidden_n: int = 32
    hidden_s: int = 32
    context_dim: int = 32
    message_dim: int = 32
    embedding_dim: int = 8
    mlp_hidden: int = 64
    dropout: float = 0.1
    temperature_scale: float = 1.0
    temperature_relation: float = 1.0
    lambda_n: float = 1.0
    lambda_s: float = 1.0
    lambda_a: float = 0.1
    chi: float = 0.25
    alpha: float = 0.5
    beta: float = 1.0
    epsilon_std: float = 1e-8
    epsilon_abs: float = 1e-8
    eta_energy: float = 1e-4
    zeta_a: float = 0.01
    epsilon_v: float = 1e-6
    epsilon_q: float = 1e-6
    epsilon_g: float = 1e-8
    eta_g: float = 0.05
    zeta_error: float = 0.95
    zeta_threshold: float = 0.99
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 50
    patience: int = 7
    min_delta: float = 1e-4
    gradient_clip: float = 1.0
    batch_size: int = 32
    num_workers: int = 0
    target_chunk_size: int = 16
    gradient_checkpointing: bool = True
    reference_chunk_size: int = 8192
    log_interval_seconds: float = 30.0
    seed: int = 42
    device: str = "auto"
    artifact_dir: str = "result/DynamicNeighborhoodStat/artifacts"
    reuse_checkpoint: bool = True
    save_variable_scores: bool = True
    sample_interval_seconds: float = 0.0

    def __post_init__(self):
        integers = ("win_size", "hidden_n", "hidden_s", "context_dim", "message_dim",
                    "embedding_dim", "mlp_hidden", "num_epochs", "patience", "batch_size",
                    "target_chunk_size", "reference_chunk_size")
        for name in integers:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError("{} must be a positive integer".format(name))
        if self.win_size < 64 or self.win_size % 32:
            raise ValueError("win_size must be a multiple of 32 and at least 64")
        if not isinstance(self.num_workers, int) or self.num_workers < 0:
            raise ValueError("num_workers must be a nonnegative integer")
        for name in ("train_ratio", "chi", "alpha", "eta_g", "zeta_error", "zeta_threshold"):
            if not 0 < getattr(self, name) < 1:
                raise ValueError("{} must lie in (0, 1)".format(name))
        if not 0 < self.zeta_a < 0.5 or not 0 <= self.dropout < 1:
            raise ValueError("Require 0 < zeta_a < 0.5 and 0 <= dropout < 1")
        positive = ("temperature_scale", "temperature_relation", "lambda_n", "lambda_s",
                    "beta", "epsilon_std", "epsilon_abs", "eta_energy", "epsilon_v",
                    "epsilon_q", "epsilon_g", "learning_rate", "gradient_clip")
        for name in positive:
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError("{} must be finite and positive".format(name))
        for name in ("lambda_a", "weight_decay", "min_delta", "sample_interval_seconds", "log_interval_seconds"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError("{} must be finite and nonnegative".format(name))

    @property
    def scales(self):
        return tuple(self.win_size // divisor for divisor in (32, 16, 8, 4, 2, 1))

    @property
    def spans(self):
        return (self.win_size // 8, self.win_size // 4)

    def to_dict(self):
        return asdict(self)

    def model_identity(self):
        """Runtime placement/output choices do not change the fitted detector."""
        ignored = {"device", "artifact_dir", "reuse_checkpoint", "save_variable_scores",
                   "num_workers", "target_chunk_size", "gradient_checkpointing",
                   "reference_chunk_size", "sample_interval_seconds", "log_interval_seconds"}
        return {key: value for key, value in self.to_dict().items() if key not in ignored}
