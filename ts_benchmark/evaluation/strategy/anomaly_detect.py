# -*- coding: utf-8 -*-
import logging
import time
import traceback
from typing import List, Any

import numpy as np
import pandas as pd

from ts_benchmark.data.data_pool import DataPool
from ts_benchmark.evaluation.evaluator import Evaluator
from ts_benchmark.evaluation.metrics import classification_metrics_label
from ts_benchmark.evaluation.metrics import classification_metrics_score
from ts_benchmark.evaluation.strategy.constants import FieldNames
from ts_benchmark.evaluation.strategy.strategy import Strategy
from ts_benchmark.models import ModelFactory
from ts_benchmark.utils.data_processing import split_before
from ts_benchmark.utils.random_utils import fix_all_random_seed

logger = logging.getLogger(__name__)


class AnomalyDetect(Strategy):
    """Fit and evaluate a detector, retaining raw point predictions for ADD."""

    REQUIRED_CONFIGS = ["seed"]

    def __init__(self, strategy_config: dict, evaluator: Evaluator):
        super().__init__(strategy_config, evaluator)
        self.model = None
        self.data_lens = None

    def execute(self, series_name: str, model_factory: ModelFactory) -> Any:
        try:
            fix_all_random_seed(self._get_scalar_config_value("seed", series_name))
            self.model = model_factory()
            train_data, train_label, test_data, test_label = self.split_data(series_name)
            start_fit_time = time.time()
            fit_method = (
                self.model.detect_fit if hasattr(self.model, "detect_fit")
                else self.model.fit
            )
            fit_method(train_data, train_label)
            end_fit_time = time.time()
            detected = self.detect(test_data)
            # Some classical detectors expose only the primary prediction array.
            if isinstance(detected, tuple):
                if len(detected) != 2:
                    raise ValueError("Detection must return predictions or (predictions, scores)")
                predict_labels, _ = detected
            else:
                predict_labels = detected
            if not isinstance(predict_labels, dict):
                predict_labels = {"None": predict_labels}
            if not predict_labels:
                raise ValueError("Detector returned no predictions")
            actual_label = test_label.to_numpy().reshape(-1).astype(float)
            end_inference_time = time.time()

            single_series_results_list = []
            for ratio, prediction in predict_labels.items():
                prediction = np.asarray(prediction)
                if prediction.ndim == 2 and prediction.shape[1] == 1:
                    prediction = prediction[:, 0]
                if prediction.ndim != 1:
                    raise ValueError("Predictions must have shape (T,) or (T, 1)")
                if len(prediction) == 0 or len(prediction) > len(actual_label):
                    raise ValueError(
                        f"Invalid prediction length {len(prediction)} for "
                        f"{len(actual_label)} test samples"
                    )
                prediction = prediction.astype(float)
                if not np.isfinite(prediction).all():
                    raise ValueError("Predictions must be finite")
                if self.accepted_metrics() == classification_metrics_label.__all__:
                    if not np.isin(prediction, [0, 1]).all():
                        raise ValueError("Label strategies require binary predictions (0 or 1)")
                alignment_log = ""
                remaining_length = len(actual_label) - len(prediction)
                if remaining_length:
                    # Compatibility for baselines that omit their final partial window.
                    # Keep this independent for each ratio and never mutate shared scores.
                    prediction = np.pad(prediction, (0, remaining_length))
                    alignment_log = (
                        f"Prediction is shorter than test labels; padded "
                        f"{remaining_length} trailing samples with 0.\n"
                    )
                    logger.warning("%s: %s", series_name, alignment_log.strip())
                values, log_info = self.evaluator.evaluate_with_log(
                    actual=actual_label, predicted=prediction,
                )
                values += [
                    series_name,
                    end_fit_time - start_fit_time,
                    end_inference_time - end_fit_time,
                    ratio,
                    "",
                    "",
                    alignment_log + log_info,
                ]
                single_series_results_list.append(values)
        except Exception as error:
            log = f"The error series is: {series_name}\n{traceback.format_exc()}\n{error}"
            single_series_results_list = [
                self.get_default_result(**{
                    FieldNames.FILE_NAME: series_name, FieldNames.LOG_INFO: log,
                })
            ]
        finally:
            # Do not keep a finished model (and its GPU tensors) in the strategy.
            self.model = None
        return single_series_results_list

    def _load_data(self, series_name):
        data = DataPool().get_pool().get_series(series_name)
        if not isinstance(data, pd.DataFrame) or "label" not in data.columns:
            raise ValueError(f"{series_name} must be a DataFrame with a label column")
        if len(data) == 0 or data.shape[1] < 2:
            raise ValueError(f"{series_name} must contain samples and at least one feature")
        if not data.columns.is_unique:
            raise ValueError(f"{series_name} contains duplicate columns")
        if not np.isin(data["label"].to_numpy(), [0, 1]).all():
            raise ValueError(f"{series_name} labels must be binary (0 or 1)")
        self.data_lens = len(data)
        return data

    def _split_at(self, data, train_length):
        if (not np.isfinite(train_length) or int(train_length) != train_length
                or not 0 < train_length < len(data)):
            raise ValueError("train_lens must be an integer strictly between 0 and series length")
        train, test = split_before(data, int(train_length))
        return (
            train.drop(columns="label"), train[["label"]],
            test.drop(columns="label"), test[["label"]],
        )

    def split_data(self, series_name):
        raise NotImplementedError

    def detect(self, test_data: pd.DataFrame):
        raise NotImplementedError

    @staticmethod
    def accepted_metrics():
        raise NotImplementedError

    @property
    def field_names(self) -> List[str]:
        return self.evaluator.metric_names + [
            FieldNames.FILE_NAME,
            FieldNames.FIT_TIME,
            FieldNames.INFERENCE_TIME,
            FieldNames.ANOMALY_RATIO,
            FieldNames.ACTUAL_DATA,
            FieldNames.INFERENCE_DATA,
            FieldNames.LOG_INFO,
        ]


class FixedDetectScore(AnomalyDetect):
    REQUIRED_CONFIGS = ["train_test_split"]

    def split_data(self, series_name):
        data = self._load_data(series_name)
        ratio = self._get_scalar_config_value("train_test_split", series_name)
        if not np.isfinite(ratio) or not 0 < ratio < 1:
            raise ValueError("train_test_split must be strictly between 0 and 1")
        return self._split_at(data, int(ratio * len(data)))

    def detect(self, test_data):
        return self.model.detect_score(test_data)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_score.__all__


class FixedDetectLabel(FixedDetectScore):
    def detect(self, test_data):
        return self.model.detect_label(test_data)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_label.__all__


class UnFixedDetectScore(AnomalyDetect):
    def split_data(self, series_name):
        data = self._load_data(series_name).reset_index(drop=True)
        metadata = DataPool().get_pool().get_series_meta_info(series_name)
        if metadata is None or "train_lens" not in metadata:
            raise ValueError(f"Missing train_lens metadata for {series_name}")
        return self._split_at(data, metadata["train_lens"])

    def detect(self, test_data):
        return self.model.detect_score(test_data)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_score.__all__


class UnFixedDetectLabel(UnFixedDetectScore):
    def detect(self, test_data):
        return self.model.detect_label(test_data)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_label.__all__


class AllDetectScore(AnomalyDetect):
    """Transductive protocol: fit and evaluate on the same unlabeled samples."""

    def split_data(self, series_name):
        data = self._load_data(series_name)
        features = data.drop(columns="label")
        return features, None, features, data[["label"]]

    def detect(self, test_data):
        return self.model.detect_score(test_data)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_score.__all__


class AllDetectLabel(AllDetectScore):
    def detect(self, test_data):
        return self.model.detect_label(test_data)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_label.__all__
