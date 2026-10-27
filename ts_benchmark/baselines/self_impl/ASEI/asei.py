"""TAB adapter and four-stage training for Adaptive Spatiotemporal Evidence Integration."""

import copy
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import logging
from pathlib import Path
import platform
import math
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .auxiliary import TaskSampler, characteristic_scale, objective, prepare_objective
from .calibration import EvidenceReference, empirical_threshold, error_history, local_features
from .model import ErrorCorrector, EvidenceScorer, Predictor, cpu_state, lag_graph, temporal_layout, unwrap
from .optimization import LinearARObjective, fit_projected_ar
from .protocol import AuxiliaryProtocol, select_protocol
from .progress import IMPLEMENTATION_VERSION, Progress


logger = logging.getLogger(__name__)


@dataclass
class Config:
    lr: object = None  # AnomalyTransformer-style alias for predictor learning_rate.
    num_epochs: object = None  # Alias for predictor_epochs; other stages keep their budgets.
    win_size: object = None  # Exact self-history receptive field; None keeps the base TCN.
    hidden_dim: int = 32
    embedding_dim: int = 8
    kernel_size: int = 3
    dilations: tuple = (1, 2, 4, 8, 16)
    graph_lags: tuple = (1, 2, 4, 8, 16)
    graph_neighbors: int = 4
    ar_order: int = 4
    rho: float = 0.95
    evidence_bound: float = 6
    rolling_window: int = 8
    omega_min: float = 0.05
    lambda_min: float = 0.8
    lambda_max: float = 0.98
    lambda_init: float = 0.9
    kappa_values: tuple = (0.01, 0.02, 0.04)
    rank_weights: tuple = (0.5, 1)
    alpha: object = 0.01
    anomaly_ratio: object = None  # Optional TAB-style percentages; overrides alpha.
    predictor_epochs: int = 50
    patience: int = 5
    batch_size: int = 128
    predictor_block_size: int = 1  # Contiguous targets per window; 1 preserves point-window sampling.
    log_interval_seconds: float = 30
    learning_rate: float = 1e-3
    ar_max_iter: int = 1000
    ar_initial_step: float = 0.01
    ar_backtracking_factor: float = 0.5
    ar_tolerance: float = 1e-5
    correction_epochs: int = 20
    correction_batch_size: int = 256
    correction_learning_rate: float = 1e-3
    score_epochs: int = 30
    score_learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    auxiliary_per_epoch: int = 128
    auxiliary_batch_size: int = 4
    normal_prefix: int = 640
    recovery_suffix: int = 768
    validation_gap: int = 1408
    checkpoint_epochs: tuple = (5, 10, 15, 20, 25, 30)
    validation_seeds: tuple = (1001, 1002)
    event_lengths: object = None
    hard_pool_size: int = 16
    rank_pairs: int = 32
    classification_temperature: float = 0.25
    ranking_temperature: float = 1
    ranking_margin: float = 1
    inference_chunk_size: int = 512
    num_workers: int = 0
    seed: object = None
    device: str = "auto"
    multi_gpu: bool = False
    device_ids: tuple = (0, 1)  # Logical IDs after CUDA_VISIBLE_DEVICES.
    checkpoint_dir: object = None


class WindowDataset(Dataset):
    def __init__(self, values, history):
        self.values = torch.as_tensor(np.asarray(values, dtype=np.float32))
        self.history = history

    def __len__(self):
        return len(self.values) - self.history

    def __getitem__(self, index):
        return self.values[index:index + self.history], self.values[index + self.history]


class BlockWindowDataset(Dataset):
    """Partition every valid target into contiguous blocks, with causal left context."""
    def __init__(self, values, history, block_size):
        self.values = torch.as_tensor(np.asarray(values, dtype=np.float32))
        self.history, self.block_size = history, block_size
        self.target_count = len(self.values) - history
        if self.target_count < 1:
            raise ValueError("Block predictor training requires at least one valid target")

    def __len__(self):
        return math.ceil(self.target_count / self.block_size)

    def __getitem__(self, index):
        start = index * self.block_size
        steps = min(self.block_size, self.target_count - start)
        return (self.values[start:start + self.history + steps - 1],
                self.values[start + self.history:start + self.history + steps])


def collate_blocks(batch):
    steps = max(len(target) for _, target in batch)
    histories, targets, masks = [], [], []
    for history, target in batch:
        missing = steps - len(target)
        # Only unused tensor positions are padded; every real target remains in the loss.
        histories.append(F.pad(history, (0, 0, 0, missing)))
        targets.append(F.pad(target, (0, 0, 0, missing)))
        masks.append(torch.arange(steps) < len(target))
    return torch.stack(histories), torch.stack(targets), torch.stack(masks)


class ASEI:
    model_name = "ASEI"

    def __init__(self, **kwargs):
        unknown = set(kwargs) - set(Config.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown ASEI hyperparameters: {sorted(unknown)}")
        options = dict(kwargs)
        for alias, target in [("lr", "learning_rate"), ("num_epochs", "predictor_epochs")]:
            if options.get(alias) is not None:
                if target in options and options[target] != options[alias]:
                    raise ValueError(f"Conflicting ASEI hyperparameters: {alias} and {target}")
                options[target] = options[alias]
        self.config = Config(**options)
        self._validate_config()
        cfg = self.config
        self.temporal_layout_, self.receptive_field = temporal_layout(cfg.kernel_size, cfg.dilations, cfg.win_size)
        self.win_size = self.receptive_field
        self.history = self.receptive_field + max(cfg.graph_lags) - 1
        self.raw_history = self.history + cfg.ar_order + cfg.rolling_window - 1
        base_field = temporal_layout(cfg.kernel_size, cfg.dilations)[1]
        for name in ["recovery_suffix", "validation_gap"]:
            if name not in kwargs:
                # Preserve complete recovery/separation update budgets when W changes.
                setattr(cfg, name, getattr(cfg, name) + self.receptive_field - base_field)
        self.seed = int(torch.initial_seed() % (2 ** 32) if cfg.seed is None else cfg.seed)
        device = ("cuda:0" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
        self.device = torch.device(device)
        self.device_ids = list(cfg.device_ids) if cfg.multi_gpu else []
        if cfg.multi_gpu:
            if (not torch.cuda.is_available() or len(self.device_ids) < 2
                    or max(self.device_ids) >= torch.cuda.device_count() or self.device.type != "cuda"):
                raise ValueError("ASEI multi_gpu=true requires at least two visible CUDA GPUs and valid logical device_ids")
            self.device = torch.device(f"cuda:{self.device_ids[0]}")
            for name in ["batch_size", "correction_batch_size", "auxiliary_batch_size"]:
                if getattr(cfg, name) < len(self.device_ids):
                    raise ValueError(f"{name} must be at least the number of training GPUs")
            if cfg.batch_size // cfg.predictor_block_size < len(self.device_ids):
                raise ValueError("batch_size / predictor_block_size must be at least the number of training GPUs")
        self.scaler = StandardScaler()
        self._fitted = False

    def _validate_config(self):
        cfg = self.config
        if cfg.win_size is not None and (isinstance(cfg.win_size, bool)
                or not isinstance(cfg.win_size, int) or cfg.win_size < 1):
            raise ValueError("win_size must be a positive integer")
        positive_integers = ["hidden_dim", "embedding_dim", "kernel_size", "graph_neighbors", "ar_order",
                             "rolling_window", "predictor_epochs", "patience", "batch_size", "predictor_block_size", "ar_max_iter",
                             "correction_epochs", "correction_batch_size", "score_epochs", "auxiliary_per_epoch",
                             "auxiliary_batch_size", "normal_prefix", "recovery_suffix", "validation_gap",
                             "hard_pool_size", "rank_pairs", "inference_chunk_size"]
        for name in positive_integers:
            value = getattr(cfg, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ["dilations", "graph_lags"]:
            values = getattr(cfg, name)
            if not isinstance(values, (tuple, list)) or not values or any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in values):
                raise ValueError(f"{name} must contain positive integers")
        for name in ["learning_rate", "ar_initial_step", "ar_tolerance", "correction_learning_rate", "score_learning_rate",
                     "evidence_bound", "classification_temperature", "ranking_temperature", "ranking_margin", "log_interval_seconds"]:
            value = getattr(cfg, name)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if cfg.batch_size % cfg.predictor_block_size:
            raise ValueError("batch_size must be divisible by predictor_block_size")
        if not 0 < cfg.rho < 1 or not 0 < cfg.omega_min < 0.5:
            raise ValueError("Require 0 < rho < 1 and 0 < omega_min < 0.5")
        if not 0 < cfg.ar_backtracking_factor < 1:
            raise ValueError("ar_backtracking_factor must be strictly between zero and one")
        if (not cfg.checkpoint_epochs or any(isinstance(epoch, bool) or not isinstance(epoch, int)
                or not 1 <= epoch <= cfg.score_epochs for epoch in cfg.checkpoint_epochs)
                or list(cfg.checkpoint_epochs) != sorted(set(cfg.checkpoint_epochs))):
            raise ValueError("checkpoint_epochs must be unique ascending integers within score_epochs")
        if not 0 < cfg.lambda_min < cfg.lambda_init < cfg.lambda_max < 1:
            raise ValueError("Require 0 < lambda_min < lambda_init < lambda_max < 1")
        if not cfg.kappa_values or any(not np.isfinite(k) or not 0 < k < cfg.evidence_bound for k in cfg.kappa_values):
            raise ValueError("kappa_values must lie strictly between 0 and evidence_bound")
        if not cfg.rank_weights or any(not np.isfinite(w) or w <= 0 for w in cfg.rank_weights):
            raise ValueError("rank_weights must be positive")
        if cfg.num_workers < 0 or not isinstance(cfg.num_workers, int):
            raise ValueError("num_workers must be a nonnegative integer")
        if not np.isfinite(cfg.weight_decay) or cfg.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if cfg.event_lengths is not None and (not cfg.event_lengths or any(
                not isinstance(length, int) or isinstance(length, bool) or length < 1 for length in cfg.event_lengths)):
            raise ValueError("event_lengths must contain positive integers")
        if (not cfg.validation_seeds or any(not isinstance(seed, int) or seed < 0 for seed in cfg.validation_seeds)
                or (cfg.seed is not None and (not isinstance(cfg.seed, int) or cfg.seed < 0))):
            raise ValueError("Seeds must be nonnegative integers")
        if not isinstance(cfg.multi_gpu, bool):
            raise ValueError("multi_gpu must be a boolean")
        if not cfg.device_ids or any(not isinstance(i, int) or i < 0 for i in cfg.device_ids) or len(set(cfg.device_ids)) != len(cfg.device_ids):
            raise ValueError("device_ids must contain unique nonnegative logical GPU IDs")
        ratios = cfg.anomaly_ratio
        alphas = cfg.alpha if isinstance(cfg.alpha, (tuple, list)) else [cfg.alpha]
        if ratios is not None:
            ratios = ratios if isinstance(ratios, (tuple, list)) else [ratios]
            alphas = [ratio / 100 for ratio in ratios]
        if not alphas or any(isinstance(a, bool) or not np.isfinite(a) or not 0 <= a < 1 for a in alphas):
            raise ValueError("alpha must be in [0, 1); anomaly_ratio is a percentage in [0, 100)")
        self.alphas = list(dict.fromkeys(float(a) for a in alphas))

    @staticmethod
    def required_hyper_params():
        return {}

    def __repr__(self):
        return self.model_name

    def _wrap(self, module):
        module = module.to(self.device)
        return nn.DataParallel(module, device_ids=self.device_ids,
                               output_device=self.device_ids[0]) if self.device_ids else module

    @staticmethod
    def _freeze(module):
        module.eval()
        module.requires_grad_(False)

    def _values(self, data, fitting=False):
        if isinstance(data, pd.DataFrame):
            if not data.columns.is_unique:
                raise ValueError("ASEI requires unique feature columns")
            if fitting:
                self.columns_ = list(data.columns)
            elif list(data.columns) != self.columns_:
                raise ValueError("ASEI feature columns and their order must match training")
            data = data.to_numpy()
        elif fitting:
            self.columns_ = list(range(np.asarray(data).shape[1]))
        values = np.asarray(data, dtype=np.float64)
        if values.ndim != 2 or min(values.shape) == 0 or not np.isfinite(values).all():
            raise ValueError("ASEI requires a nonempty finite numeric (time, channels) array")
        if not fitting and values.shape[1] != len(self.columns_):
            raise ValueError("ASEI channel count must match training")
        return values

    def _loader(self, dataset, batch_size, shuffle=False):
        generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False,
                          num_workers=self.config.num_workers, pin_memory=self.device.type == "cuda",
                          generator=generator, persistent_workers=self.config.num_workers > 0,
                          multiprocessing_context="spawn" if self.config.num_workers > 0 else None)

    def _stage(self, name, work):
        started = time.monotonic()
        with Progress(f"stage={name}", interval=self.config.log_interval_seconds) as progress:
            progress.phase("executing")
            result = work()
        self.metadata_.setdefault("stage_seconds", {})[name] = time.monotonic() - started
        return result

    def _predictor_loader(self, values, shuffle=False):
        cfg = self.config
        if cfg.predictor_block_size == 1:
            return self._loader(WindowDataset(values, self.history), cfg.batch_size, shuffle)
        return DataLoader(
            BlockWindowDataset(values, self.history, cfg.predictor_block_size),
            batch_size=cfg.batch_size // cfg.predictor_block_size, shuffle=shuffle,
            drop_last=False, collate_fn=collate_blocks, num_workers=cfg.num_workers,
            pin_memory=self.device.type == "cuda", generator=torch.Generator().manual_seed(self.seed),
            persistent_workers=cfg.num_workers > 0,
            multiprocessing_context="spawn" if cfg.num_workers > 0 else None,
        )

    def _predictor_loss(self, loader, optimizer=None, epoch=None):
        total, count = 0.0, 0
        self.predictor.train(optimizer is not None)
        label = f"predictor epoch={epoch} {'train' if optimizer is not None else 'validation'}"
        with torch.set_grad_enabled(optimizer is not None), Progress(label, len(loader), self.config.log_interval_seconds) as progress:
            progress.phase("data_loading")
            for step, batch in enumerate(loader, 1):
                history, target = batch[:2]
                valid = batch[2] if len(batch) == 3 else None
                points = int(valid.sum().item()) if valid is not None else len(history)
                progress.phase(f"batch={step} transfer")
                history, target = history.to(self.device), target.to(self.device)
                if target.ndim == 2:
                    target = target[:, None]
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                progress.phase(f"batch={step} forward")
                predicted = self.predictor(history, output_steps=target.shape[1])
                actual = target[..., None].expand_as(predicted)
                if valid is None:
                    loss = F.huber_loss(predicted, actual, delta=1)
                else:
                    losses = F.huber_loss(predicted, actual, delta=1, reduction="none").mean(dim=(-1, -2))
                    loss = (losses * valid.to(self.device)).sum() / points
                self._finite(loss, "predictor loss")
                if optimizer is not None:
                    progress.phase(f"batch={step} backward")
                    loss.backward()
                    progress.phase(f"batch={step} optimizer")
                    optimizer.step()
                total += loss.item() * points
                count += points
                progress.update(step, f"mean_loss={total / count:.7f} target_points={count}")
                progress.phase("data_loading")
        return total / count

    @staticmethod
    def _finite(value, name):
        if not torch.isfinite(value).all().item():
            raise FloatingPointError(f"ASEI {name} is not finite")

    def _fit_predictor(self, fit, validation):
        cfg = self.config
        with Progress("lag_graph", interval=cfg.log_interval_seconds) as progress:
            progress.phase("aligned_correlations")
            sources, delays, mask = lag_graph(fit, cfg.graph_lags, cfg.graph_neighbors)
        with Progress("predictor setup", interval=cfg.log_interval_seconds) as progress:
            progress.phase("model_device_initialization")
            self.predictor = self._wrap(Predictor(len(self.columns_), sources, delays, mask,
                                                  cfg.hidden_dim, cfg.embedding_dim,
                                                  cfg.kernel_size, cfg.dilations, win_size=cfg.win_size))
            progress.phase("data_loaders")
            train_loader = self._predictor_loader(fit, True)
            val_loader = self._predictor_loader(validation)
            optimizer = torch.optim.AdamW(self.predictor.parameters(), lr=cfg.learning_rate,
                                          weight_decay=cfg.weight_decay)
        logger.info("ASEI predictor workload train_targets=%d train_batches=%d validation_targets=%d validation_batches=%d block_size=%d",
                    len(fit) - self.history, len(train_loader), len(validation) - self.history,
                    len(val_loader), cfg.predictor_block_size)
        best_loss, checkpoint, stale, best_epoch = float("inf"), None, 0, 0
        for epoch in range(1, cfg.predictor_epochs + 1):
            train_loss = self._predictor_loss(train_loader, optimizer, epoch)
            val_loss = self._predictor_loss(val_loader, epoch=epoch)
            logger.info("ASEI predictor epoch=%d train=%.6f validation=%.6f", epoch, train_loss, val_loss)
            if val_loss < best_loss:
                best_loss, checkpoint, stale, best_epoch = val_loss, cpu_state(self.predictor), 0, epoch
            else:
                stale += 1
                if stale >= cfg.patience:
                    break
        unwrap(self.predictor).load_state_dict(checkpoint)
        self._freeze(self.predictor)
        self.metadata_["predictor"] = {"best_epoch": best_epoch, "validation_huber": best_loss,
                                       "graph_edges": int(mask.sum())}

    @torch.no_grad()
    def _errors(self, values, progress_label=None):
        # Frozen sequence inference reuses causal convolution states within each chunk.
        # The primary GPU processes continuous time; temporal state is never split across replicas.
        predictor = unwrap(self.predictor)
        result = []
        chunks = math.ceil(max(0, len(values) - self.history) / self.config.inference_chunk_size)
        context = Progress(progress_label, chunks, self.config.log_interval_seconds) if progress_label else nullcontext()
        with context as progress:
            for step, begin in enumerate(range(self.history, len(values), self.config.inference_chunk_size), 1):
                if progress:
                    progress.phase(f"chunk={step} prediction")
                end = min(len(values), begin + self.config.inference_chunk_size)
                inputs = torch.as_tensor(values[begin - self.history:end - 1],
                                         dtype=torch.float32, device=self.device)[None]
                predicted = predictor(inputs, output_steps=end - begin)[0]
                actual = torch.as_tensor(values[begin:end], dtype=torch.float32, device=self.device)
                error = (actual[..., None] - predicted).square()
                self._finite(error, "prediction errors")
                result.append(error.cpu().numpy())
                if progress:
                    progress.update(step)
        return np.concatenate(result) if result else np.empty((0, len(self.columns_), 2), dtype=np.float32)

    def _fit_corrector(self, values):
        cfg = self.config
        errors = self._errors(values, "correction prediction errors")
        history = error_history(errors, cfg.ar_order)
        target = errors[cfg.ar_order:]
        corrector = ErrorCorrector(len(self.columns_), cfg.ar_order, cfg.rho)
        logged = np.log1p(history)
        scale = logged.std(axis=0)
        scale[scale == 0] = 1
        with torch.no_grad():
            corrector.bias.copy_(torch.from_numpy(errors.mean(0, dtype=np.float64)))
            corrector.log_mean.copy_(torch.from_numpy(logged.mean(0)))
            corrector.log_scale.copy_(torch.from_numpy(scale))
            corrector.eta.copy_(torch.from_numpy(0.1 * (np.percentile(errors, 75, axis=0)
                                                       - np.percentile(errors, 25, axis=0))))
        self.corrector = self._wrap(corrector)
        # Full-batch float64 sufficient sums are reduced across DataParallel replicas.
        # No optimizer update is made until every fitting sample has contributed.
        linear_objective = self._wrap(LinearARObjective(corrector))
        with Progress("linear_AR", cfg.ar_max_iter, cfg.log_interval_seconds) as progress:
            ar_status = fit_projected_ar(
                linear_objective,
                torch.as_tensor(history, dtype=torch.float64, device=self.device),
                torch.as_tensor(target, dtype=torch.float64, device=self.device),
                radius=cfg.rho, max_iter=cfg.ar_max_iter, initial_step=cfg.ar_initial_step,
                shrink=cfg.ar_backtracking_factor, tolerance=cfg.ar_tolerance, progress=progress,
            )
        corrector.bias.requires_grad_(False)
        corrector.beta.requires_grad_(False)
        corrector.mlp.requires_grad_(True)
        del linear_objective
        loader = self._loader(TensorDataset(torch.from_numpy(history), torch.from_numpy(target)),
                              cfg.correction_batch_size, True)
        optimizer = torch.optim.AdamW(corrector.mlp.parameters(), lr=cfg.correction_learning_rate,
                                      weight_decay=cfg.weight_decay)
        for epoch in range(1, cfg.correction_epochs + 1):
            total, count = 0.0, 0
            self.corrector.train()
            with Progress(f"compensation epoch={epoch}", len(loader), cfg.log_interval_seconds) as progress:
                for step, (previous, current) in enumerate(loader, 1):
                    progress.phase(f"batch={step} forward_backward")
                    previous, current = previous.to(self.device), current.to(self.device, dtype=torch.float64)
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.huber_loss(self.corrector(previous), current, delta=1)
                    self._finite(loss, "correction loss")
                    loss.backward()
                    optimizer.step()
                    total += loss.item() * len(previous)
                    count += len(previous)
                    progress.update(step, f"mean_loss={total / count:.7f}")
            if epoch == cfg.correction_epochs or epoch == 1 or epoch % 10 == 0:
                logger.info("ASEI compensation MLP epoch=%d loss=%.6f", epoch, total / count)
        self._freeze(self.corrector)
        self.metadata_["corrector"] = {"samples": len(target), "rho": cfg.rho,
                                       "maximum_beta_l1": float(corrector.beta.abs().sum(-1).max().item()),
                                       "ar_optimization": ar_status}
        logger.info("ASEI linear AR converged=%s iterations=%d mapping_inf=%.6g maximum_beta_l1=%.12g",
                    ar_status["converged"], ar_status["iterations"], ar_status["gradient_mapping_inf"],
                    self.metadata_["corrector"]["maximum_beta_l1"])

    @torch.no_grad()
    def _residuals(self, errors):
        history = error_history(errors, self.config.ar_order)
        residuals = []
        for begin in range(0, len(history), self.config.correction_batch_size):
            previous = torch.as_tensor(history[begin:begin + self.config.correction_batch_size], device=self.device)
            predicted = unwrap(self.corrector)(previous)
            residuals.append(predicted.cpu().numpy())
        return errors[self.config.ar_order:] - np.concatenate(residuals)

    def _features(self, values):
        if len(values) <= self.raw_history:
            return np.empty((0, len(self.columns_), 4), dtype=np.float32)
        errors = self._errors(values)
        residuals = self._residuals(errors)
        evidence = self.reference.transform(errors[self.config.ar_order:], residuals)
        return local_features(evidence, self.config.rolling_window)

    def _feature_chunks(self, values):
        """Bound frozen feature memory independently of the full sequence length."""
        for begin in range(self.raw_history, len(values), self.config.inference_chunk_size):
            end = min(len(values), begin + self.config.inference_chunk_size)
            yield self._features(values[begin - self.raw_history:end])

    def _new_scorer(self, kappa):
        cfg = self.config
        protocol = self.protocol_
        return EvidenceScorer(cfg.evidence_bound, cfg.omega_min, protocol.lambda_min,
                              protocol.lambda_max, protocol.lambda_init, kappa, cfg.classification_temperature)

    def _fit_reference(self, reference):
        errors = self._errors(reference, "independent reference prediction errors")
        residuals = self._residuals(errors)
        self.reference = EvidenceReference(self.config.evidence_bound).fit(errors[self.config.ar_order:], residuals)

    @torch.no_grad()
    def _calibrate(self, validation):
        state, score_chunks = None, []
        chunks = math.ceil((len(validation) - self.raw_history) / self.config.inference_chunk_size)
        with Progress("normal validation calibration", chunks, self.config.log_interval_seconds) as progress:
            for step, features in enumerate(self._feature_chunks(validation), 1):
                progress.phase(f"chunk={step} state_updates")
                current, state = self.scorer.integrate(torch.as_tensor(features, device=self.device)[None], state)
                score_chunks.append(current[0].cpu().numpy())
                progress.update(step)
        self.validation_scores_ = np.concatenate(score_chunks)
        self.thresholds_, calibration = {}, []
        for alpha in self.alphas:
            threshold, false_alarms = empirical_threshold(self.validation_scores_, alpha)
            self.thresholds_[alpha] = threshold
            count = len(self.validation_scores_)
            calibration.append(dict(alpha=alpha, N=count, K=int(np.floor(alpha * count)),
                                    threshold=threshold, false_alarm_count=false_alarms,
                                    empirical_fpr=false_alarms / count, normal_points=count, resolution=1 / count))
        self.metadata_["calibration"] = calibration

    def _batch(self, tasks, sampler):
        arrays, metadata = [], []
        for task in tasks:
            normal, changed = sampler.materialize(task)
            arrays.append(np.stack((self._features(normal), self._features(changed))))
            metadata.append(prepare_objective(task, self.raw_history, self.config.rank_pairs,
                                              np.random.default_rng(task.seed + 11)))
        longest = max(array.shape[1] for array in arrays)
        features = np.zeros((len(arrays), 2, longest, len(self.columns_), 4), dtype=np.float32)
        for index, array in enumerate(arrays):
            features[index, :, :array.shape[1]] = array
        return torch.as_tensor(features, device=self.device), metadata

    def _loss(self, candidate, features, metadata):
        scores, logits = candidate(features)
        return objective(scores, logits, metadata, self.config.ranking_margin,
                         self.config.ranking_temperature)

    @torch.no_grad()
    def _validate_scorers(self, candidates, sampler, tasks):
        grouped = [{} for _ in candidates]
        with Progress("scoring validation", len(tasks), self.config.log_interval_seconds) as progress:
            for task_index, task in enumerate(tasks):
                progress.phase(f"task={task_index + 1} evidence")
                features, metadata = self._batch([task], sampler)
                for candidate_index, candidate in enumerate(candidates):
                    progress.phase(f"task={task_index + 1} candidate={candidate_index + 1}/{len(candidates)}")
                    candidate["model"].eval()
                    detection, ranking = self._loss(candidate["model"], features, metadata)
                    loss = (detection + ranking).item()
                    if not np.isfinite(loss):
                        raise FloatingPointError("ASEI validation auxiliary objective is not finite")
                    grouped[candidate_index].setdefault((task.family, task.scenario), []).append(loss)
                progress.update(task_index + 1)
        self._last_validation_groups_ = [
            {f"family{family}/{scenario}": dict(loss=float(np.mean(values)), count=len(values))
             for (family, scenario), values in groups.items()}
            for groups in grouped
        ]
        return [float(np.mean([np.mean(values) for values in groups.values()])) for groups in grouped]

    def _fit_scorer(self, fit, validation, scale):
        cfg = self.config
        sampler = TaskSampler(fit, self.raw_history, scale, cfg, self.protocol_)
        valid_sampler = TaskSampler(validation, self.raw_history, scale, cfg, self.protocol_)
        valid_tasks = valid_sampler.validation()  # Fail before optimizing if the protocol cannot fit.
        # All configurations share each newly generated paired batch, avoiding a large evidence cache.
        candidates = []
        for kappa in cfg.kappa_values:
            for gamma in cfg.rank_weights:
                model = self._wrap(self._new_scorer(kappa))
                candidates.append(dict(model=model, kappa=kappa, gamma=gamma,
                                       optimizer=torch.optim.AdamW(model.parameters(), lr=cfg.score_learning_rate,
                                                                    weight_decay=cfg.weight_decay)))
        state_means = []
        chunks = math.ceil((len(fit) - self.raw_history) / cfg.inference_chunk_size)
        with Progress("hard-normal evidence", chunks, cfg.log_interval_seconds) as progress:
            for step, features in enumerate(self._feature_chunks(fit), 1):
                progress.phase(f"chunk={step} evidence")
                state_g = features[..., 1].astype(np.float64) + self.reference.center[None, :, 1]
                state_means.append(state_g.mean(1))
                progress.update(step)
        hardness = np.r_[np.zeros(self.raw_history), np.concatenate(state_means)]
        hardness_sum = np.r_[0.0, hardness.cumsum()]
        del state_means
        best = None
        best_key = (float("inf"), float("inf"), float("inf"))
        selection_log = []
        self.scoring_checkpoints_ = {}
        for epoch in range(1, cfg.score_epochs + 1):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, 7919, epoch]))
            tasks = []
            for index in range(cfg.auxiliary_per_epoch):
                if index % 2 == 0:
                    task = sampler.training(rng)
                else:
                    pool = [sampler.training(rng) for _ in range(cfg.hard_pool_size)]
                    def difficulty(item):
                        begin = item.start + self.raw_history
                        return (hardness_sum[item.start + item.length] - hardness_sum[begin]) / (item.length - self.raw_history)
                    pool.sort(key=difficulty, reverse=True)
                    task = pool[int(rng.integers(max(1, int(np.ceil(len(pool) / 4)))))]
                tasks.append(task)
            batches = math.ceil(len(tasks) / cfg.auxiliary_batch_size)
            with Progress(f"scoring epoch={epoch}", batches, cfg.log_interval_seconds) as progress:
                for step, begin in enumerate(range(0, len(tasks), cfg.auxiliary_batch_size), 1):
                    progress.phase(f"batch={step} evidence")
                    features, metadata = self._batch(tasks[begin:begin + cfg.auxiliary_batch_size], sampler)
                    for candidate_index, candidate in enumerate(candidates):
                        progress.phase(f"batch={step} candidate={candidate_index + 1}/{len(candidates)} forward_backward")
                        model, optimizer = candidate["model"], candidate["optimizer"]
                        model.train()
                        optimizer.zero_grad(set_to_none=True)
                        detection, ranking = self._loss(model, features, metadata)
                        loss = detection + candidate["gamma"] * ranking
                        self._finite(loss, "scoring objective")
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1)
                        optimizer.step()
                        unwrap(model).project_()
                    progress.update(step)
            logger.info("ASEI scoring epoch=%d/%d candidates=%d", epoch, cfg.score_epochs, len(candidates))
            if epoch in cfg.checkpoint_epochs:
                losses = self._validate_scorers(candidates, valid_sampler, valid_tasks)
                selection_log.append(dict(epoch=epoch, losses=losses,
                                          groups=copy.deepcopy(getattr(self, "_last_validation_groups_", []))))
                self.scoring_checkpoints_[epoch] = []
                for index, (candidate, loss) in enumerate(zip(candidates, losses)):
                    checkpoint = dict(config_index=index, kappa=candidate["kappa"], gamma=candidate["gamma"],
                                      epoch=epoch, validation_loss=loss, state=cpu_state(candidate["model"]))
                    self.scoring_checkpoints_[epoch].append(checkpoint)
                    key = (loss, epoch, index)
                    if key < best_key:
                        best_key = key
                        best = dict(kappa=candidate["kappa"], gamma=candidate["gamma"],
                                    config_index=index, epoch=epoch, loss=loss, state=checkpoint["state"])
                logger.info("ASEI scoring selection epoch=%d losses=%s", epoch, losses)
        self.scorer = self._new_scorer(best["kappa"]).to(self.device)
        self.scorer.load_state_dict(best.pop("state"))
        self._freeze(self.scorer)
        self.metadata_["selection"] = best
        self.metadata_["selection_history"] = selection_log
        self.metadata_["scoring_candidates"] = [dict(config_index=index, kappa=candidate["kappa"],
                                                     gamma=candidate["gamma"]) for index, candidate in enumerate(candidates)]
        self.metadata_["saved_scoring_checkpoints"] = sum(map(len, self.scoring_checkpoints_.values()))
        self.metadata_["validation_tasks"] = [asdict(task) for task in valid_tasks]
        self.metadata_["decay"] = float(self.scorer.decay.item())
        del candidates

    def detect_fit(self, train_data, train_label=None, valid_data=None):
        """Labels are not used. valid_data optionally supplies an external normal split."""
        self._fitted = False
        cfg = self.config
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        values = self._values(train_data, fitting=True)
        if valid_data is None:
            boundary = int(len(values) * 0.8)
            training, validation = values[:boundary], values[boundary:]
        else:
            training, validation = values, self._values(valid_data)
            boundary = len(values)
        first, second = int(len(training) * 0.8), int(len(training) * 0.9)
        fit, correction, reference = training[:first], training[first:second], training[second:]
        for name, part, minimum in [("predictor", fit, self.history + 1),
                                     ("corrector", correction, self.history + cfg.ar_order + 1),
                                     ("reference", reference, self.raw_history + 1),
                                     ("validation", validation, self.raw_history + 1)]:
            if len(part) < minimum:
                raise ValueError(f"ASEI {name} split has {len(part)} points; requires at least {minimum}. "
                                 "Splits are chronological: outer 80/20, inner 80/10/10. See docs/ASEI.md.")
        self.scaler.fit(fit)
        fit, correction, reference, validation = [self.scaler.transform(part).astype(np.float32)
                                                  for part in [fit, correction, reference, validation]]
        scale = characteristic_scale(correction)
        # Only deterministic normal lengths and T0 participate; no seed or validation loss.
        self.protocol_, protocol_trials = select_protocol(cfg, self.raw_history, scale, len(fit), len(validation))
        validation_tasks = TaskSampler(validation, self.raw_history, scale, cfg, self.protocol_).validation()
        signature = hashlib.sha256()
        signature.update(repr((values.shape, self.columns_)).encode("utf-8"))
        digest_inputs = [values]
        if valid_data is not None:
            digest_inputs.append(self._values(valid_data))
        for digest_values in digest_inputs:
            for begin in range(0, len(digest_values), 4096):
                signature.update(memoryview(np.ascontiguousarray(digest_values[begin:begin + 4096])).cast("B"))
        self.metadata_ = dict(
            implementation_version=IMPLEMENTATION_VERSION,
            data_sha256=signature.hexdigest(),
            seed=self.seed, splits={"outer_training_end": boundary, "predictor_end": first,
                                   "corrector_end": second, "training_end": len(training),
                                   "validation_points": len(validation), "external_validation": valid_data is not None},
            receptive_field=self.receptive_field, raw_history=self.raw_history, T0=scale,
            temporal_layout=[list(block) for block in self.temporal_layout_],
            predictor_learning_rate=cfg.learning_rate, predictor_max_epochs=cfg.predictor_epochs,
            auxiliary_protocol=asdict(self.protocol_), protocol_feasibility=protocol_trials,
            reference_points=len(reference) - self.history - cfg.ar_order,
            validation_task_count=len(validation_tasks), device=str(self.device),
            data_parallel=bool(self.device_ids), logical_device_ids=self.device_ids,
            versions={"python": platform.python_version(), "numpy": np.__version__,
                      "torch": torch.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
        )
        logger.info("ASEI device=%s logical_gpu_ids=%s W=%d raw_history=%d splits=%s T0=%d s=%g",
                    self.device, self.device_ids, self.receptive_field, self.raw_history,
                    self.metadata_["splits"], scale, self.protocol_.s)
        logger.info("ASEI predictor batch_size=%d lr=%g max_epochs=%d; compensation_epochs=%d scoring_epochs=%d",
                    cfg.batch_size, cfg.learning_rate, cfg.predictor_epochs,
                    cfg.correction_epochs, cfg.score_epochs)
        logger.info("ASEI runtime implementation=%s source=%s channels=%d predictor_block_size=%d",
                    IMPLEMENTATION_VERSION, __file__, len(self.columns_), cfg.predictor_block_size)
        self._stage("1/4 predictor", lambda: self._fit_predictor(fit, validation))
        self._stage("2/4 correction", lambda: self._fit_corrector(correction))
        # d and r use the same timestamps within the independent reference segment.
        self._stage("3/4 independent_reference", lambda: self._fit_reference(reference))
        self._stage("4/4 scoring", lambda: self._fit_scorer(fit, validation, scale))
        self._stage("threshold_calibration", lambda: self._calibrate(validation))
        self.metadata_["parameters"] = {
            name: sum(parameter.numel() for parameter in unwrap(module).parameters())
            for name, module in [("predictor", self.predictor), ("corrector", self.corrector), ("scorer", self.scorer)]}
        self._fitted = True
        self.reset_state()
        logger.info("ASEI selected=%s calibration=%s", self.metadata_["selection"], self.metadata_["calibration"])
        if cfg.checkpoint_dir:
            config_hash = hashlib.sha256(json.dumps(asdict(cfg), sort_keys=True).encode("utf-8")).hexdigest()[:8]
            path = Path(cfg.checkpoint_dir) / f"asei_seed{self.seed}_{signature.hexdigest()[:16]}_{config_hash}.pt"
            self.save(path)
            logger.info("ASEI checkpoint=%s", path)
        return self

    def reset_state(self):
        """Start a new entity/sequence. No observations from a previous call are reused."""
        self._raw_cache = np.empty((0, len(self.columns_)), dtype=np.float32)
        self._stream_state = None

    @torch.no_grad()
    def detect_stream(self, data, reset=False):
        """Return (percentage-keyed labels, raw scores); preserve causal caches across chunks."""
        if not self._fitted:
            raise RuntimeError("Call ASEI.detect_fit before detection")
        if reset:
            self.reset_state()
        values = self.scaler.transform(self._values(data)).astype(np.float32)
        lower = -self.config.evidence_bound - max(self.config.kappa_values)
        scores = np.full(len(values), lower, dtype=np.float32)
        self.last_ready_mask_ = np.zeros(len(values), dtype=bool)
        for begin in range(0, len(values), self.config.inference_chunk_size):
            end = min(len(values), begin + self.config.inference_chunk_size)
            previous = len(self._raw_cache)
            joined = np.concatenate((self._raw_cache, values[begin:end]))
            features = self._features(joined)
            first_ready = begin + min(end - begin, max(0, self.raw_history - previous))
            self.last_ready_mask_[first_ready:end] = True
            if len(features):
                tensor = torch.as_tensor(features, device=self.device)[None]
                current, self._stream_state = self.scorer.integrate(tensor, self._stream_state)
                self._finite(current, "online scores")
                scores[first_ready:end] = current[0].cpu().numpy()
            self._raw_cache = joined[-self.raw_history:].copy()
        predictions = {100 * alpha: ((scores > threshold) & self.last_ready_mask_).astype(np.int64)
                       for alpha, threshold in self.thresholds_.items()}
        return predictions, scores

    def detect_score(self, test_data):
        _, scores = self.detect_stream(test_data, reset=True)
        return scores, scores

    def detect_label(self, test_data):
        return self.detect_stream(test_data, reset=True)

    def save(self, path):
        """Save a complete portable detector without DataParallel key prefixes."""
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted ASEI detector")
        payload = dict(format_version=2, config=asdict(self.config), columns=self.columns_, seed=self.seed,
                       protocol=asdict(self.protocol_), scoring_checkpoints=self.scoring_checkpoints_,
                       predictor=cpu_state(self.predictor), corrector=cpu_state(self.corrector),
                       scorer=cpu_state(self.scorer), kappa=self.scorer.kappa,
                       reference={"errors": torch.from_numpy(self.reference.errors),
                                  "residuals": torch.from_numpy(self.reference.residuals),
                                  "center": torch.from_numpy(self.reference.center)},
                       scaler={name: copy.deepcopy(getattr(self.scaler, name))
                               for name in ["mean_", "scale_", "var_", "n_features_in_", "n_samples_seen_"]},
                       validation_scores=torch.from_numpy(self.validation_scores_),
                       thresholds=self.thresholds_, metadata=self.metadata_)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    @classmethod
    def load(cls, path, device="auto", multi_gpu=False):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format_version") != 2:
            raise ValueError("Unsupported ASEI checkpoint format")
        config = dict(payload["config"], device=device, multi_gpu=multi_gpu)
        detector = cls(**config)
        detector.columns_, detector.seed = payload["columns"], payload["seed"]
        detector.protocol_ = AuxiliaryProtocol(**payload["protocol"])
        detector.scoring_checkpoints_ = payload["scoring_checkpoints"]
        cfg, state = detector.config, payload["predictor"]
        detector.predictor = detector._wrap(Predictor(len(detector.columns_), state["sources"],
                                                       state["delays"], state["edge_mask"], cfg.hidden_dim,
                                                       cfg.embedding_dim, cfg.kernel_size, cfg.dilations,
                                                       win_size=cfg.win_size))
        unwrap(detector.predictor).load_state_dict(state)
        detector.corrector = detector._wrap(ErrorCorrector(len(detector.columns_), cfg.ar_order, cfg.rho))
        unwrap(detector.corrector).load_state_dict(payload["corrector"])
        detector.scorer = detector._new_scorer(payload["kappa"]).to(detector.device)
        detector.scorer.load_state_dict(payload["scorer"])
        for module in [detector.predictor, detector.corrector, detector.scorer]:
            detector._freeze(module)
        for name, value in payload["scaler"].items():
            setattr(detector.scaler, name, value)
        detector.reference = EvidenceReference(cfg.evidence_bound)
        for name, value in payload["reference"].items():
            setattr(detector.reference, name, value.numpy())
        detector.validation_scores_ = payload["validation_scores"].numpy()
        detector.thresholds_, detector.metadata_ = payload["thresholds"], payload["metadata"]
        detector._fitted = True
        detector.reset_state()
        return detector
