import logging
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from ts_benchmark.baselines.self_impl.Anomaly_trans.AnomalyTransformer_model import (
    AnomalyTransformer_model,
)
from ts_benchmark.baselines.utils import anomaly_detection_data_provider, train_val_split

logger = logging.getLogger(__name__)

DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS = {
    "win_size": 100,
    "lr": 0.0001,
    "e_layers": 3,
    "d_model": 512,
    "n_heads": 8,
    "d_ff": 512,
    "pretrained_model": None,
    "num_epochs": 3,
    "batch_size": 256,
    "patience": 3,
    "k": 3,
    "multi_gpu": False,
    "anomaly_ratio": [0.1, 0.5, 1.0, 2, 3, 5.0, 10.0, 15, 20, 25],
}


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr = lr_ * (0.5 ** (epoch - 1))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    logger.info("Updating learning rate to %s", lr)


def _unwrapped(model):
    return model.module if isinstance(model, nn.DataParallel) else model


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name="", delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.val_loss2_min = np.inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model):
        if not np.isfinite([val_loss, val_loss2]).all():
            raise FloatingPointError("AnomalyTransformer validation loss is not finite")
        score, score2 = -val_loss, -val_loss2
        if self.best_score is None or (
            score >= self.best_score + self.delta
            and score2 >= self.best_score2 + self.delta
        ):
            self.best_score, self.best_score2 = score, score2
            self.save_checkpoint(val_loss, val_loss2, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                logger.info("EarlyStopping counter: %d / %d", self.counter, self.patience)
            self.early_stop = self.counter >= self.patience

    def save_checkpoint(self, val_loss, val_loss2, model):
        # The same keys load into either a plain model or a parallel model.
        # CPU copies do not consume an extra model's worth of primary GPU memory.
        self.check_point = {
            key: value.detach().cpu().clone()
            for key, value in _unwrapped(model).state_dict().items()
        }
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class TransformerConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)


class AnomalyTransformer:
    def __init__(self, **kwargs):
        self.config = TransformerConfig(**kwargs)
        self.model_name = "AnomalyTransformer"
        for name in ("win_size", "batch_size", "num_epochs", "e_layers",
                     "d_model", "n_heads", "d_ff", "patience"):
            value = getattr(self.config, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.config.win_size > 5000:
            raise ValueError("win_size exceeds the positional embedding limit (5000)")
        if self.config.d_model % 2 or self.config.d_model % self.config.n_heads:
            raise ValueError("d_model must be even and divisible by n_heads")
        if not np.isfinite(self.config.lr) or self.config.lr <= 0:
            raise ValueError("lr must be finite and positive")
        if not np.isfinite(self.config.k) or self.config.k < 0:
            raise ValueError("k must be finite and nonnegative")
        if not isinstance(self.config.multi_gpu, bool):
            raise ValueError("multi_gpu must be a JSON boolean")
        if self.config.pretrained_model is not None:
            raise ValueError("pretrained_model is not supported by this implementation")

        ratios = self.config.anomaly_ratio
        self.anomaly_ratios = list(ratios) if isinstance(ratios, (list, tuple)) else [ratios]
        if not self.anomaly_ratios or any(
            isinstance(ratio, bool) or not isinstance(ratio, (int, float))
            or not np.isfinite(ratio) or not 0 <= ratio <= 100
            for ratio in self.anomaly_ratios
        ):
            raise ValueError("anomaly_ratio must contain percentages in [0, 100]")

        self.scaler = StandardScaler()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device_ids = []
        if self.config.multi_gpu:
            if self.device.type != "cuda" or torch.cuda.device_count() < 2:
                raise ValueError("multi_gpu=true requires at least two visible CUDA GPUs")
            self.device_ids = list(range(torch.cuda.device_count()))
            if self.config.batch_size < len(self.device_ids):
                raise ValueError("batch_size must be at least the number of GPUs")
        self.criterion = nn.MSELoss()
        self.win_size = self.config.win_size

    @staticmethod
    def required_hyper_params() -> dict:
        return {}

    def __repr__(self):
        return self.model_name

    def _association_losses(self, series, prior):
        series_loss, prior_loss = 0.0, 0.0
        for series_layer, prior_layer in zip(series, prior):
            normalized_prior = prior_layer / prior_layer.sum(dim=-1, keepdim=True)
            series_loss += (
                my_kl_loss(series_layer, normalized_prior.detach()).mean()
                + my_kl_loss(normalized_prior.detach(), series_layer).mean()
            )
            prior_loss += (
                my_kl_loss(normalized_prior, series_layer.detach()).mean()
                + my_kl_loss(series_layer.detach(), normalized_prior).mean()
            )
        return series_loss / len(prior), prior_loss / len(prior)

    @torch.no_grad()
    def vali(self, vali_loader):
        self.model.eval()
        loss_1, loss_2, count = 0.0, 0.0, 0
        for input_data, _ in vali_loader:
            inputs = input_data.float().to(self.device)
            output, series, prior, _ = self.model(inputs)
            series_loss, prior_loss = self._association_losses(series, prior)
            rec_loss = self.criterion(output, inputs)
            batch_count = len(inputs)
            loss_1 += (rec_loss - self.config.k * series_loss).item() * batch_count
            loss_2 += (rec_loss + self.config.k * prior_loss).item() * batch_count
            count += batch_count
        if count == 0:
            raise ValueError("Validation data contains no complete windows")
        return loss_1 / count, loss_2 / count

    def detect_fit(self, train_data: pd.DataFrame, train_label=None):
        self.config.input_c = train_data.shape[1]
        self.config.output_c = train_data.shape[1]
        train_values, valid_values = train_val_split(train_data, 0.8, None)
        if min(len(train_values), len(valid_values)) < self.win_size:
            raise ValueError(
                "Both the 80% training split and the 20% validation split must "
                f"contain at least win_size={self.win_size} samples"
            )
        if not np.isfinite(train_data.to_numpy()).all():
            raise ValueError("Training data must contain only finite numeric values")
        self.scaler.fit(train_values.values)
        train_values = self.scaler.transform(train_values.values)
        valid_values = self.scaler.transform(valid_values.values)
        self.train_loader = anomaly_detection_data_provider(
            train_values, batch_size=self.config.batch_size,
            win_size=self.win_size, step=1, mode="train",
        )
        self.valid_loader = anomaly_detection_data_provider(
            valid_values, batch_size=self.config.batch_size,
            win_size=self.win_size, step=1, mode="val",
        )
        self.model = AnomalyTransformer_model(
            win_size=self.win_size, enc_in=self.config.input_c,
            c_out=self.config.output_c, e_layers=self.config.e_layers,
            d_model=self.config.d_model, n_heads=self.config.n_heads,
            d_ff=self.config.d_ff,
        ).to(self.device)
        if self.config.multi_gpu:
            self.model = nn.DataParallel(
                self.model, device_ids=self.device_ids, output_device=self.device_ids[0],
            )
        logger.info(
            "AnomalyTransformer device=%s, data_parallel=%s, device_ids=%s, "
            "global_batch_size=%d, trainable_parameters=%d",
            self.device, self.config.multi_gpu, self.device_ids,
            self.config.batch_size,
            sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        )
        self.early_stopping = EarlyStopping(patience=self.config.patience, verbose=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        for epoch in range(self.config.num_epochs):
            started = time.time()
            self.model.train()
            train_loss, count = 0.0, 0
            for i, (input_data, _) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                inputs = input_data.float().to(self.device)
                output, series, prior, _ = self.model(inputs)
                series_loss, prior_loss = self._association_losses(series, prior)
                rec_loss = self.criterion(output, inputs)
                loss1 = rec_loss - self.config.k * series_loss
                loss2 = rec_loss + self.config.k * prior_loss
                if not torch.isfinite(loss1).item() or not torch.isfinite(loss2).item():
                    raise FloatingPointError("AnomalyTransformer training loss is not finite")
                # Preserve the original two-objective minimax gradients.
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()
                train_loss += loss1.item() * len(inputs)
                count += len(inputs)
                if (i + 1) % 100 == 0:
                    logger.info("Epoch %d, step %d / %d", epoch + 1, i + 1, len(self.train_loader))
            vali_loss1, vali_loss2 = self.vali(self.valid_loader)
            logger.info(
                "Epoch %d: train_loss=%.7f, validation_losses=(%.7f, %.7f), time=%.2fs",
                epoch + 1, train_loss / count, vali_loss1, vali_loss2, time.time() - started,
            )
            self.early_stopping(vali_loss1, vali_loss2, self.model)
            if self.early_stopping.early_stop:
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.config.lr)
        self._restore_checkpoint()
        return self

    def _restore_checkpoint(self):
        _unwrapped(self.model).load_state_dict(self.early_stopping.check_point)
        self.model.eval()

    @torch.no_grad()
    def _collect_energy(self, scaled_data):
        values = np.asarray(scaled_data)
        length = len(values)
        if length == 0:
            raise ValueError("Detection data must not be empty")
        if not np.isfinite(values).all():
            raise ValueError("Detection data must contain only finite numeric values")
        # Keep every real timestamp, including a final partial window.
        # Edge padding supplies the missing context; padded scores are discarded.
        padding = (-length) % self.win_size
        if padding:
            values = np.pad(values, ((0, padding), (0, 0)), mode="edge")
        loader = anomaly_detection_data_provider(
            values, batch_size=self.config.batch_size, win_size=self.win_size,
            step=self.win_size, mode="thre",
        )
        scores = []
        for input_data, _ in loader:
            inputs = input_data.float().to(self.device)
            output, series, prior, _ = self.model(inputs)
            reconstruction_error = (inputs - output).square().mean(dim=-1)
            series_loss, prior_loss = 0.0, 0.0
            for series_layer, prior_layer in zip(series, prior):
                normalized_prior = prior_layer / prior_layer.sum(dim=-1, keepdim=True)
                series_loss += my_kl_loss(series_layer, normalized_prior) * 50
                prior_loss += my_kl_loss(normalized_prior, series_layer) * 50
            weights = torch.softmax(-series_loss - prior_loss, dim=-1)
            scores.append((weights * reconstruction_error).cpu().numpy())
        energy = np.concatenate(scores, axis=0).reshape(-1)[:length]
        if not np.isfinite(energy).all():
            raise FloatingPointError("AnomalyTransformer anomaly scores are not finite")
        return energy

    def detect_score(self, test_data: pd.DataFrame):
        self._restore_checkpoint()
        energy = self._collect_energy(self.scaler.transform(test_data.values))
        return energy, energy

    def detect_label(self, test_data: pd.DataFrame):
        self._restore_checkpoint()
        # Threshold calibration uses only held-out training data, never test data
        # or test labels. The validation values are already standardized.
        val_energy = self._collect_energy(self.valid_loader.dataset.data)
        test_energy = self._collect_energy(self.scaler.transform(test_data.values))
        preds = {
            ratio: (test_energy > np.percentile(val_energy, 100 - ratio)).astype(int)
            for ratio in self.anomaly_ratios
        }
        return preds, test_energy
