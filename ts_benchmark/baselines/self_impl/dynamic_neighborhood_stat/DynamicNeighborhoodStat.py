"""TAB adapter, normal-only training, sequential calibration and persistence."""

import hashlib
import json
import logging
import random
import re
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

from ts_benchmark.common.constant import ROOT_PATH
from .config import Config
from .features import training_references
from .model import DualPathNetwork
from .progress import StageProgress

logger = logging.getLogger(__name__)


class WindowDataset(Dataset):
    def __init__(self, values, length):
        if length <= 0 or len(values) < length:
            raise ValueError("A window must fit entirely inside its input segment")
        # Normalize the layout once, rather than copying each overlapping window.
        self.values, self.length = np.ascontiguousarray(values), length
        self.offsets = np.arange(length)

    def __len__(self):
        return len(self.values) - self.length + 1

    def __getitem__(self, index):
        if isinstance(index, (list, tuple, np.ndarray)):
            indices = np.asarray(index)
            if indices.ndim != 1 or not len(indices) or not np.issubdtype(indices.dtype, np.integer):
                raise IndexError("A batch must contain integer window indices")
            if indices.min() < 0 or indices.max() >= len(self):
                raise IndexError("Window index is outside its input segment")
            # One contiguous batch allocation; no per-window copy followed by stack.
            return torch.from_numpy(self.values[indices[:, None] + self.offsets[None]])
        if index < 0 or index >= len(self):
            raise IndexError("Window index is outside its input segment")
        window = self.values[index:index + self.length]
        if len(window) != self.length:
            raise RuntimeError("A window crossed the boundary of its input segment")
        return window


def window_split_metadata(train_length, validation_length, window_length):
    """Half-open window boundaries in the official normal training segment."""
    metadata = {"index_basis": "zero_based_official_training_segment", "window_length": window_length,
                "stride": 1, "window_alignment": "right"}
    for name, start, length in (("train", 0, train_length), ("validation", train_length, validation_length)):
        if length < window_length:
            raise ValueError("Both independent segments must contain a full window")
        stop = start + length
        metadata[name] = {"raw_start": start, "raw_end_exclusive": stop,
                          "window_count": length - window_length + 1,
                          "first_window_start": start, "first_window_end_exclusive": start + window_length,
                          "last_window_start": stop - window_length, "last_window_end_exclusive": stop,
                          "first_score_index": start + window_length - 1, "last_score_index": stop - 1}
    if metadata["train"]["last_window_end_exclusive"] > metadata["validation"]["first_window_start"]:
        raise RuntimeError("Training and validation windows must not share observations")
    return metadata


def _fingerprint(values, columns):
    digest = hashlib.sha256(json.dumps([list(columns), list(values.shape)]).encode("utf-8"))
    for start in range(0, len(values), 8192):
        digest.update(values[start:start + 8192].tobytes(order="C"))
    return digest.hexdigest()


def _json(path, content):
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(content, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _tensor_dict(references):
    return {name: torch.as_tensor(value).cpu() for name, value in references.items()}


class DynamicNeighborhoodStat:
    """Main method from In1_5, using TAB's detect_fit/score/label interfaces.

    A fitted instance is specific to one entity and one ordered set of columns.
    Scores are length T; their first L-1 positions are NaN and explicitly invalid.
    """

    CHECKPOINT_VERSION = 1

    def __init__(self, **kwargs):
        self.config = Config(**kwargs)
        self.model = None
        self.series_name_ = "series"
        self._prediction_cache = None

    @staticmethod
    def required_hyper_params():
        # Do not inherit TAB's global input_chunk_length=12.
        return {}

    def set_run_context(self, series_name, seed):
        self.series_name_ = str(series_name)
        self.config.seed = int(seed)

    def _device(self):
        value = self.config.device
        if value == "auto":
            value = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(value)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but this PyTorch environment has no available CUDA device")
        return device

    @staticmethod
    def _values(frame):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame in chronological row order")
        if frame.columns.has_duplicates or "label" in frame.columns:
            raise ValueError("Feature columns must be unique and must not contain the label column")
        try:
            values = frame.to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("All model inputs must be numeric feature columns") from error
        if values.ndim != 2 or values.shape[1] < 2:
            raise ValueError("DynamicNeighborhoodStat requires at least two variables")
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite input: preprocess missing values causally; no implicit filling is performed")
        return values

    def _loader(self, values, training=False):
        generator = torch.Generator().manual_seed(self.config.seed)
        dataset = WindowDataset(values, self.config.win_size)
        sampler = RandomSampler(dataset, generator=generator) if training else SequentialSampler(dataset)
        batches = BatchSampler(sampler, self.config.batch_size, drop_last=False)
        # Passing whole batches through the sampler disables per-item collate/stack.
        return DataLoader(dataset, batch_size=None, sampler=batches, num_workers=self.config.num_workers,
                          pin_memory=self.device_.type == "cuda", generator=generator)

    def _progress(self, stage, total):
        return StageProgress("{} seed={} {}".format(self.series_name_, self.config.seed, stage), total,
                             self.config.log_interval_seconds, self.run_dir_ / "runtime_progress.json")

    def _log_runtime(self):
        runtime = {"torch": torch.__version__, "device": str(self.device_), "cuda_available": torch.cuda.is_available(),
                   "cudnn_available": torch.backends.cudnn.is_available(),
                   "inherited_cudnn_enabled": torch.backends.cudnn.enabled,
                   "model_cudnn_enabled": True, "cudnn_deterministic": True, "cudnn_benchmark": False,
                   "cpu_threads": torch.get_num_threads(), "channels": len(self.columns_),
                   "batch_size": self.config.batch_size, "win_size": self.config.win_size,
                   "target_chunk_size": self.config.target_chunk_size,
                   "gradient_checkpointing": self.config.gradient_checkpointing,
                   "factorized_routing": self.config.factorized_routing, "prune_tcn": self.config.prune_tcn,
                   "num_workers": self.config.num_workers}
        if self.device_.type == "cuda":
            runtime["gpu"] = torch.cuda.get_device_name(self.device_)
        logger.info("%s runtime: %s; artifacts=%s", self.series_name_, json.dumps(runtime), self.run_dir_)
        if self.config.device == "auto" and self.device_.type == "cpu":
            logger.warning("device=auto selected CPU; use device=cuda:0 to require GPU instead of silently falling back")
        _json(self.run_dir_ / "runtime_environment.json", runtime)

    def detect_fit(self, train_data, train_label=None):
        self._prediction_cache = None
        values = self._values(train_data)
        if train_label is not None:
            labels = np.asarray(train_label).reshape(-1)
            if len(labels) != len(values) or not np.isin(labels, [0, 1]).all():
                raise ValueError("Training labels must be aligned binary labels")
            if np.any(labels != 0):
                raise ValueError("This method requires a clean normal training segment; anomalous rows are not deleted or joined")
        split = int(len(values) * self.config.train_ratio)
        if min(split, len(values) - split) < self.config.win_size:
            raise ValueError("Both chronological train and validation portions must contain at least win_size observations")
        self.split_metadata_ = window_split_metadata(split, len(values) - split, self.config.win_size)
        logger.info("%s: train=[0,%d), validation=[%d,%d), windows=%d/%d, first validation score=%d",
                    self.series_name_, split, split, len(values),
                    self.split_metadata_["train"]["window_count"], self.split_metadata_["validation"]["window_count"],
                    self.split_metadata_["validation"]["first_score_index"])
        self.columns_ = list(train_data.columns)
        self.train_fingerprint_ = _fingerprint(values, self.columns_)
        identity = json.dumps(self.config.model_identity(), sort_keys=True).encode("utf-8")
        identity_hash = hashlib.sha256(identity).hexdigest()[:12]
        series = re.sub(r"[^\w.\-]+", "_", self.series_name_)
        root = Path(self.config.artifact_dir).expanduser()
        if not root.is_absolute():
            root = Path(ROOT_PATH) / root
        self.run_dir_ = root.resolve() / series / ("seed_{}".format(self.config.seed)) / (identity_hash + "_" + self.train_fingerprint_[:12])
        self.run_dir_.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path_ = self.run_dir_ / "checkpoint.pt"
        self.evaluation_artifact_dir_ = str(self.run_dir_)
        self.device_ = self._device()
        self._log_runtime()
        if self.config.reuse_checkpoint and self.checkpoint_path_.is_file():
            self._load(self.checkpoint_path_, expected_fingerprint=self.train_fingerprint_)
            # Refresh derived metadata even for checkpoints saved before these fields existed.
            self._write_calibration(split, len(values) - split)
            logger.info("Reusing calibrated checkpoint: %s", self.checkpoint_path_)
            return self

        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        raw_train = values[:split]
        self.mean_ = raw_train.mean(axis=0)
        standard_deviation = raw_train.std(axis=0, ddof=0)
        self.constant_input_ = standard_deviation <= self.config.epsilon_std
        self.scale_ = np.where(self.constant_input_, 1.0, standard_deviation)
        train = (raw_train - self.mean_) / self.scale_
        validation = (values[split:] - self.mean_) / self.scale_
        logger.info("%s: computing full training references (%d train, %d validation observations)", self.series_name_, len(train), len(validation))
        with self._progress("training references", values.shape[1] * (len(self.config.spans) + 1)) as progress:
            self.references_ = training_references(train, self.config, progress=progress.update)
        self.model = DualPathNetwork(values.shape[1], self.config, self.references_).to(self.device_)
        self.training_history_ = self._train(train, validation)
        self._calibrate(validation)
        self.model.eval()
        self.save(self.checkpoint_path_)
        pd.DataFrame(self.training_history_).to_csv(self.run_dir_ / "training_history.csv", index=False)
        self._write_calibration(len(train), len(validation))
        logger.info("Fit and calibration completed: %s", self.checkpoint_path_)
        return self

    # TAB's fix_all_random_seed() disables cuDNN globally. Scope this override to
    # this model, including backward, while preserving deterministic convolutions.
    @torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True)
    def _epoch(self, loader, optimizer=None, stage=None):
        training = optimizer is not None
        self.model.train(training)
        names = ("loss", "reconstruction", "loss_n", "loss_s", "regularizer",
                 "concentration_mean", "concentration_exceed_fraction", "concentration_excess")
        totals, count = torch.zeros(len(names), dtype=torch.float64, device=self.device_), 0
        with torch.set_grad_enabled(training), self._progress(stage or ("train" if training else "validation"), len(loader)) as progress:
            for batch_index, windows in enumerate(loader, 1):
                windows = windows.to(self.device_, non_blocking=True)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                output = self.model(windows)
                losses = self.model.losses(windows, output)
                if not torch.isfinite(losses["loss"]):
                    raise FloatingPointError("Nonfinite loss; inspect input scaling and configured numerical floors")
                if training:
                    losses["loss"].backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip, error_if_nonfinite=True)
                    optimizer.step()
                concentration = output["concentration"].detach()
                metrics = [losses[name].detach().to(torch.float64) for name in names[:5]]
                metrics.extend(value.to(torch.float64) for value in (
                    concentration.mean(), (concentration > self.model.rho_a).float().mean(),
                    torch.relu(concentration - self.model.rho_a).mean()))
                totals.add_(torch.stack(metrics) * len(windows))
                count += len(windows)
                progress.update(batch_index, lambda: "windows={} loss={:.6g}".format(count, float(losses["loss"].detach())))
        return dict(zip(names, (totals / count).cpu().tolist()))

    def _train(self, train, validation):
        train_loader, validation_loader = self._loader(train, True), self._loader(validation)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        best, best_state, waiting, history = float("inf"), None, 0, []
        logger.info("%s: %d training batches + %d validation batches per epoch, maximum %d epochs",
                    self.series_name_, len(train_loader), len(validation_loader), self.config.num_epochs)
        for epoch in range(1, self.config.num_epochs + 1):
            started = time.perf_counter()
            train_metrics = self._epoch(train_loader, optimizer, "epoch {}/{} train".format(epoch, self.config.num_epochs))
            train_seconds = time.perf_counter() - started
            started = time.perf_counter()
            validation_metrics = self._epoch(validation_loader, stage="epoch {}/{} validation".format(epoch, self.config.num_epochs))
            validation_seconds = time.perf_counter() - started
            history.append(dict(epoch=epoch, **{"train_" + k: v for k, v in train_metrics.items()},
                                **{"validation_" + k: v for k, v in validation_metrics.items()},
                                train_seconds=train_seconds, validation_seconds=validation_seconds))
            pd.DataFrame(history).to_csv(self.run_dir_ / "training_history.csv", index=False)
            monitored = validation_metrics["reconstruction"]
            logger.info("%s seed=%d epoch=%d train=%.6g validation=%.6g concentration=%.4f train_seconds=%.1f validation_seconds=%.1f", self.series_name_,
                        self.config.seed, epoch, train_metrics["loss"], monitored, validation_metrics["concentration_mean"],
                        train_seconds, validation_seconds)
            if monitored < best - self.config.min_delta:
                best, waiting = monitored, 0
                best_state = {name: value.detach().cpu().clone() for name, value in self.model.state_dict().items()}
                self.best_epoch_ = epoch
            else:
                waiting += 1
                if waiting >= self.config.patience:
                    break
        self.model.load_state_dict(best_state)
        self.model.eval()
        return history

    @staticmethod
    def _pack_errors(errors):
        n, s, p = errors
        return torch.cat((n.flatten(start_dim=2), s.flatten(start_dim=2), p[..., None]), dim=-1).detach().cpu().numpy()

    def _score_errors(self, errors):
        n_end = 2 * len(self.config.scales)
        s_end = n_end + 3 * len(self.config.spans)
        n = (errors[..., :n_end] / self.q_n_.reshape(len(self.columns_), -1)).mean(axis=-1)
        s = self.config.chi * errors[..., s_end] / self.q_p_
        s = s + (1 - self.config.chi) * (errors[..., n_end:s_end] / self.q_s_.reshape(len(self.columns_), -1)).mean(axis=-1)
        variable = self.config.alpha * self.support_ * n + (1 - self.config.alpha * self.support_) * s
        maximum = variable.max(axis=-1)
        score = maximum + np.log(np.exp(self.config.beta * (variable - maximum[..., None])).mean(axis=-1)) / self.config.beta
        return score, variable

    @torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True)
    def _calibrate(self, validation):
        """Fixed network -> support -> error quantiles -> full-score quantile."""
        count, channels = len(validation) - self.config.win_size + 1, len(self.columns_)
        coordinates = 2 * len(self.config.scales) + 3 * len(self.config.spans) + 1
        required_bytes = count * channels * (coordinates + 1) * np.dtype(np.float64).itemsize
        free_bytes = shutil.disk_usage(self.run_dir_).free
        logger.info("%s: calibration cache=%.3f GiB, free disk=%.3f GiB, directory=%s",
                    self.series_name_, required_bytes / 2**30, free_bytes / 2**30, self.run_dir_)
        if free_bytes < required_bytes:
            raise OSError("Calibration requires {} bytes of temporary disk space, only {} are free in {}".format(
                required_bytes, free_bytes, self.run_dir_))
        token = uuid.uuid4().hex
        errors_path, c_path = self.run_dir_ / (".errors_" + token), self.run_dir_ / (".concentration_" + token)
        errors = concentrations = None
        model_error = torch.zeros(channels, dtype=torch.float64, device=self.device_)
        mean_error = torch.zeros_like(model_error)
        message_sum = torch.zeros(2, dtype=torch.float64, device=self.device_)
        offset = 0
        self.model.eval()
        try:
            errors = np.memmap(errors_path, dtype=np.float64, mode="w+", shape=(count, channels, coordinates))
            concentrations = np.memmap(c_path, dtype=np.float64, mode="w+", shape=(count, channels))
            loader = self._loader(validation)
            with torch.no_grad(), self._progress("calibration residuals", len(loader)) as progress:
                for batch_index, windows in enumerate(loader, 1):
                    windows = windows.to(self.device_, non_blocking=True)
                    output = self.model(windows)
                    residual_errors = self.model.errors(windows, output)
                    packed = self._pack_errors(residual_errors)
                    stop = offset + len(windows)
                    errors[offset:stop] = packed
                    concentrations[offset:stop] = output["concentration"].cpu().numpy()
                    normalizer = self.model.variance_n.clamp_min(self.config.epsilon_v)
                    model_error.add_((residual_errors[0] / normalizer).mean(dim=(-2, -1)).sum(dim=0))
                    mean_error.add_(((output["target_n"] - self.model.mean_n).square() / normalizer).mean(dim=(-2, -1)).sum(dim=0))
                    message_sum.add_(output["message_diagnostics"].sum(dim=(0, 1)).to(torch.float64))
                    offset = stop
                    progress.update(batch_index, "windows={}".format(offset))
            self.support_model_error_, self.support_mean_error_ = (torch.stack((model_error, mean_error)) / count).cpu().numpy()
            message_sum = message_sum.cpu().numpy()
            self.support_gain_ = (self.support_mean_error_ - self.support_model_error_) / np.maximum(self.support_mean_error_, self.config.epsilon_g)
            self.support_ = (self.support_mean_error_ > self.config.epsilon_g) & (self.support_gain_ >= self.config.eta_g)
            # Each quantile is exact; only a single coordinate is copied to RAM at a time.
            q = np.empty((channels, coordinates))
            with self._progress("calibration error quantiles", channels * coordinates) as progress:
                for i in range(channels):
                    for k in range(coordinates):
                        q[i, k] = max(float(np.quantile(errors[:, i, k], self.config.zeta_error)), self.config.epsilon_q)
                        progress.update(i * coordinates + k + 1, "variable={}/{}".format(i + 1, channels))
            n_end = 2 * len(self.config.scales)
            self.q_n_ = q[:, :n_end].reshape(channels, len(self.config.scales), 2)
            self.q_s_ = q[:, n_end:-1].reshape(channels, len(self.config.spans), 3)
            self.q_p_ = q[:, -1]
            validation_scores = np.empty(count, dtype=np.float64)
            with self._progress("calibration scores and threshold", count) as progress:
                for start in range(0, count, self.config.reference_chunk_size):
                    stop = min(start + self.config.reference_chunk_size, count)
                    validation_scores[start:stop], _ = self._score_errors(errors[start:stop])
                    if stop < count:
                        progress.update(stop)
                self.threshold_ = float(np.quantile(validation_scores, self.config.zeta_threshold))
                progress.update(count, "threshold={:.6g} support={}/{}".format(self.threshold_, int(self.support_.sum()), channels))
            self.calibration_diagnostics_ = {
                "concentration_quantiles": np.quantile(concentrations, [0.5, 0.9, 0.95, 0.99]).tolist(),
                "concentration_exceed_fraction": float(np.mean(concentrations > self.model.rho_a)),
                "concentration_mean_excess": float(np.maximum(concentrations - self.model.rho_a, 0).mean()),
                "mean_message_linf": float(message_sum[0] / (count * channels)),
                "mean_weighted_message_linf": float(message_sum[1] / (count * channels)),
                "validation_windows": count,
            }
        finally:
            # Delete only the two explicitly created files; close mappings first for Windows.
            for mapping, path in ((errors, errors_path), (concentrations, c_path)):
                if mapping is not None:
                    mapping._mmap.close()
                path.unlink(missing_ok=True)

    def freeze_energy_diagnostics(self):
        """Theoretical raw A-error bounds, not alarm or overall-score guarantees."""
        if not hasattr(self, "references_"):
            raise RuntimeError("Training references are required for freeze diagnostics")
        log_floor = np.log(self.references_["energy_floor"])
        margin = self.references_["a_low"] - log_floor
        positive = margin > 0
        return {"columns": self.columns_, "spans": list(self.config.spans),
                "log_energy_floor": log_floor.tolist(), "a_low": self.references_["a_low"].tolist(),
                "margin": margin.tolist(), "has_positive_bound": positive.tolist(),
                "energy_squared_error_lower_bound": np.square(np.maximum(margin, 0)).tolist(),
                "positive_bound_fraction": float(positive.mean())}

    def _write_calibration(self, train_length, validation_length):
        self.split_metadata_ = window_split_metadata(train_length, validation_length, self.config.win_size)
        _json(self.run_dir_ / "calibration.json", {
            "config": self.config.to_dict(), "series_name": self.series_name_, "columns": self.columns_,
            "train_fingerprint": self.train_fingerprint_, "train_length": train_length,
            "validation_length": validation_length, "best_epoch": self.best_epoch_,
            "split_metadata": self.split_metadata_, "freeze_energy": self.freeze_energy_diagnostics(),
            "parameter_count": sum(p.numel() for p in self.model.parameters()),
            "receptive_field_n": self.model.encoder_n.receptive_field,
            "receptive_field_s": self.model.encoder_s.receptive_field,
            "mean": self.mean_.tolist(), "scale": self.scale_.tolist(),
            "constant_input": self.constant_input_.tolist(),
            "references": {key: value.tolist() for key, value in self.references_.items()},
            "support": self.support_.tolist(), "support_gain": self.support_gain_.tolist(),
            "support_model_error": self.support_model_error_.tolist(), "support_mean_error": self.support_mean_error_.tolist(),
            "support_fraction": float(self.support_.mean()), "q_n": self.q_n_.tolist(),
            "q_s": self.q_s_.tolist(), "q_p": self.q_p_.tolist(), "threshold": self.threshold_,
            "top_r": self.model.top_r, "rho_a": self.model.rho_a, "diagnostics": self.calibration_diagnostics_,
        })

    def save(self, path):
        if self.model is None or not hasattr(self, "threshold_"):
            raise RuntimeError("Fit and calibrate the detector before saving")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": self.CHECKPOINT_VERSION, "config": self.config.to_dict(),
                   "series_name": self.series_name_, "columns": self.columns_,
                   "train_fingerprint": self.train_fingerprint_, "references": _tensor_dict(self.references_),
                   "model": {name: value.detach().cpu() for name, value in self.model.state_dict().items()},
                   "threshold": self.threshold_, "best_epoch": self.best_epoch_,
                   "history": self.training_history_, "diagnostics": self.calibration_diagnostics_,
                   "split_metadata": getattr(self, "split_metadata_", None)}
        for name in ("mean_", "scale_", "constant_input_", "support_", "support_gain_", "support_model_error_", "support_mean_error_", "q_n_", "q_s_", "q_p_"):
            payload[name] = torch.as_tensor(getattr(self, name)).cpu()
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def _load(self, path, expected_fingerprint=None):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["version"] != self.CHECKPOINT_VERSION:
            raise ValueError("Unsupported checkpoint format")
        if expected_fingerprint is not None and payload["train_fingerprint"] != expected_fingerprint:
            raise ValueError("Checkpoint was fitted to different training observations")
        if Config(**payload["config"]).model_identity() != self.config.model_identity():
            raise ValueError("Checkpoint configuration does not match this detector")
        self.columns_, self.series_name_ = payload["columns"], payload["series_name"]
        self.train_fingerprint_ = payload["train_fingerprint"]
        self.references_ = {key: value.numpy() for key, value in payload["references"].items()}
        self.model = DualPathNetwork(len(self.columns_), self.config, self.references_).to(self.device_)
        self.model.load_state_dict(payload["model"])
        self.model.eval()
        self.threshold_, self.best_epoch_ = payload["threshold"], payload["best_epoch"]
        self.training_history_, self.calibration_diagnostics_ = payload["history"], payload["diagnostics"]
        self.split_metadata_ = payload.get("split_metadata") or getattr(self, "split_metadata_", None)
        for name in ("mean_", "scale_", "constant_input_", "support_", "support_gain_", "support_model_error_", "support_mean_error_", "q_n_", "q_s_", "q_p_"):
            setattr(self, name, payload[name].numpy())
        self._prediction_cache = None

    @classmethod
    def load(cls, path, device="auto"):
        path = Path(path).resolve()
        payload = torch.load(path, map_location="cpu", weights_only=True)
        config = dict(payload["config"], device=device)
        detector = cls(**config)
        detector.device_, detector.run_dir_, detector.checkpoint_path_ = detector._device(), path.parent, path
        detector.evaluation_artifact_dir_ = str(path.parent)
        detector._load(path)
        detector._log_runtime()
        return detector

    @torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True)
    def _predict(self, test_data):
        if self.model is None or not hasattr(self, "threshold_"):
            raise RuntimeError("Call detect_fit or load a calibrated checkpoint first")
        values = self._values(test_data)
        if list(test_data.columns) != self.columns_:
            raise ValueError("Test feature columns/order differ from the fitted detector")
        if len(values) < self.config.win_size:
            raise ValueError("The test segment must contain at least win_size observations")
        key = _fingerprint(values, self.columns_)
        if self._prediction_cache is not None and self._prediction_cache[0] == key:
            logger.info("%s: reusing cached test scores", self.series_name_)
            return self._prediction_cache[1].copy()
        normalized = (values - self.mean_) / self.scale_
        scores = np.full(len(values), np.nan, dtype=np.float64)
        self.evaluation_valid_mask_ = np.arange(len(values)) >= self.config.win_size - 1
        variable_path = self.run_dir_ / "test_variable_scores.npy"
        variable = None
        if self.config.save_variable_scores:
            variable = np.lib.format.open_memmap(variable_path, mode="w+", dtype=np.float64, shape=values.shape)
            variable[:self.config.win_size - 1] = np.nan
        offset = self.config.win_size - 1
        concentration_totals = torch.zeros(3, dtype=torch.float64, device=self.device_)
        c_count = 0
        self.model.eval()
        try:
            loader = self._loader(normalized)
            with torch.no_grad(), self._progress("test inference", len(loader)) as progress:
                for batch_index, windows in enumerate(loader, 1):
                    windows = windows.to(self.device_, non_blocking=True)
                    output = self.model(windows)
                    batch_scores, variable_scores = self._score_errors(self._pack_errors(self.model.errors(windows, output)))
                    stop = offset + len(windows)
                    scores[offset:stop] = batch_scores
                    if variable is not None:
                        variable[offset:stop] = variable_scores
                    c = output["concentration"]
                    concentration_totals.add_(torch.stack((c.sum().to(torch.float64),
                        (c > self.model.rho_a).sum().to(torch.float64), torch.relu(c - self.model.rho_a).sum().to(torch.float64))))
                    c_count += c.numel()
                    offset = stop
                    progress.update(batch_index, "scored={}/{}".format(offset - self.config.win_size + 1, len(loader.dataset)))
        finally:
            if variable is not None:
                variable.flush()
                variable._mmap.close()
        if not np.isfinite(scores[self.evaluation_valid_mask_]).all():
            raise FloatingPointError("Nonfinite valid test scores")
        c_sum, c_exceed, c_excess = concentration_totals.cpu().tolist()
        alarms = np.zeros(len(values), dtype=np.int8)
        alarms[self.evaluation_valid_mask_] = (scores[self.evaluation_valid_mask_] > self.threshold_).astype(np.int8)
        pd.DataFrame({"position": np.arange(len(values)), "input_index": test_data.index,
                      "valid": self.evaluation_valid_mask_, "score": scores, "alarm": alarms}).to_csv(
                          self.run_dir_ / "test_predictions.csv", index=False)
        _json(self.run_dir_ / "test_diagnostics.json", {
            "test_fingerprint": key, "test_length": len(values),
            "valid_start": self.config.win_size - 1, "valid_points": int(self.evaluation_valid_mask_.sum()),
            "threshold": self.threshold_, "support_fraction": float(self.support_.mean()),
            "concentration_mean": c_sum / c_count, "concentration_exceed_fraction": c_exceed / c_count,
            "concentration_mean_excess": c_excess / c_count,
        })
        self._prediction_cache = (key, scores.copy())
        logger.info("%s: test predictions saved to %s", self.series_name_, self.run_dir_ / "test_predictions.csv")
        return scores

    def detect_score(self, test_data):
        scores = self._predict(test_data)
        return scores, scores.copy()

    def detect_label(self, test_data):
        scores = self._predict(test_data)
        alarms = np.zeros(len(scores), dtype=np.int8)
        alarms[self.evaluation_valid_mask_] = (scores[self.evaluation_valid_mask_] > self.threshold_).astype(np.int8)
        return {"validation_q{}".format(self.config.zeta_threshold): alarms}, scores
