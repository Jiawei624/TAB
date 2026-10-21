# -*- coding: utf-8 -*-
import base64
import pickle
import json
import time
import traceback
from pathlib import Path
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


class AnomalyDetect(Strategy):
    """
    异常检测类，用于在时间序列数据上执行异常检测。
    """

    REQUIRED_CONFIGS = ["seed"]

    def __init__(self, strategy_config: dict, evaluator: Evaluator):
        """
        初始化子类实例。

        :param strategy_config: 模型评估配置。
        """
        super().__init__(strategy_config, evaluator)
        self.model = None
        self.data_lens = None

    def execute(self, series_name: str, model_factory: ModelFactory) -> Any:
        """
        执行异常检测策略。

        :param series_name: 要执行异常检测的序列名称。
        :param model_factory: 模型对象的构造/工厂函数。
        :return: 评估结果。
        """
        # fix_random_seed()
        # fix_random_seed(self.strategy_config.get("seed", 42))
        fix_all_random_seed(self.strategy_config.get("seed", 42))

        self.evaluation_metadata_ = None
        model = model_factory()
        try:
            self.model = model
            if hasattr(model, "set_run_context"):
                model.set_run_context(series_name, self.strategy_config.get("seed", 42))
            train_data, train_label, test_data, test_label = self.split_data(
                series_name
            )
            start_fit_time = time.time()
            if hasattr(model, "detect_fit"):
                self.model.detect_fit(train_data, train_label)  # 在训练数据上拟合模型
            else:
                self.model.fit(train_data, train_label)  # 在训练数据上拟合模型

            end_fit_time = time.time()
            predict_labels, another = self.detect(test_data)
            if not isinstance(predict_labels, dict):
                predict_labels = {"None": predict_labels}

            actual_label = test_label.to_numpy().flatten()
            end_inference_time = time.time()

            # 逐组预测生成结果
            single_series_results_list = []
            for ratio, predict_label in predict_labels.items():
                valid_mask = getattr(model, "evaluation_valid_mask_", None)
                metric_context = {}
                if valid_mask is not None:
                    valid_mask = np.asarray(valid_mask)
                    if valid_mask.dtype != bool or valid_mask.shape != actual_label.shape or not valid_mask.any():
                        raise ValueError("Model evaluation_valid_mask_ must be a nonempty aligned boolean mask")
                    predict_label = np.asarray(predict_label)
                    if predict_label.shape != actual_label.shape or np.asarray(another).shape != actual_label.shape:
                        raise ValueError("Masked detectors must return full-length aligned arrays")
                    if not np.isfinite(predict_label[valid_mask]).all():
                        raise ValueError("Predictions must be finite at all valid evaluation points")
                    metric_context = dict(valid_mask=valid_mask, full_actual=actual_label,
                                          full_predicted=predict_label)
                remaining_length = len(actual_label) - len(predict_label)
                print(remaining_length)
                # Pad the predict_label array with zeros at the end
                if remaining_length > 0:
                    predict_label = np.pad(
                        predict_label,
                        (0, remaining_length),
                        mode="constant",
                        constant_values=0,
                    )
                    another = np.pad(
                        another,
                        (0, remaining_length),
                        mode="constant",
                        constant_values=0,
                    )

                # 计算该组预测的全部配置指标
                single_series_results, log_info = self.evaluator.evaluate_with_log(
                    actual=actual_label.astype(float) if valid_mask is None else actual_label[valid_mask].astype(float),
                    predicted=predict_label.astype(float) if valid_mask is None else predict_label[valid_mask].astype(float),
                    **metric_context,
                )
                artifact_dir = getattr(model, "evaluation_artifact_dir_", None)
                if artifact_dir is not None and valid_mask is not None:
                    self.evaluation_metadata_ = self._save_masked_evaluation(
                        artifact_dir, actual_label, predict_label, valid_mask, single_series_results, log_info)
                print(single_series_results)

                inference_data = [predict_label, another]
                actual_data_pickle = pickle.dumps(test_label)
                actual_data_pickle = base64.b64encode(actual_data_pickle).decode(
                    "utf-8"
                )

                inference_data_pickle = pickle.dumps(inference_data)
                inference_data_pickle = base64.b64encode(inference_data_pickle).decode(
                    "utf-8"
                )
                single_series_results += [
                    series_name,
                    end_fit_time - start_fit_time,
                    end_inference_time - end_fit_time,
                    ratio,
                    "",
                    "",
                    log_info,
                ]
                single_series_results_list.append(single_series_results)
        except Exception as e:
            # log = f"{traceback.format_exc()}\n{e}"
            log = f"The error series is: {series_name}\n{traceback.format_exc()}\n{e}"
            single_series_results_list = [
                self.get_default_result(**{FieldNames.FILE_NAME: series_name, FieldNames.LOG_INFO: log})
            ]
        return single_series_results_list

    def _save_masked_evaluation(self, directory, actual, predicted, valid_mask, results, log_info):
        """Labels stay in the evaluator; model training/inference never receives test labels."""
        directory = Path(directory)
        label_mode = "add" in self.accepted_metrics()
        mode = "label" if label_mode else "score"
        positions = np.flatnonzero(valid_mask)
        info = {"index_basis": "zero_based_test_segment", "test_length": len(actual),
                "valid_start": int(positions[0]), "valid_end_exclusive": int(positions[-1] + 1),
                "valid_points": int(valid_mask.sum()), "excluded_points": int((~valid_mask).sum()),
                "metrics": {name: (float(value) if np.isfinite(value) else None)
                            for name, value in zip(self.evaluator.metric_names, results)}, "log_info": log_info}
        edges = np.diff(np.r_[0, actual.astype(np.int8), 0])
        spans = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
        eligibility = [bool(begin > 0 and end < len(actual) and valid_mask[begin:end].all())
                       for begin, end in spans]
        info["total_events"] = len(spans)
        info["complete_events"] = int(sum(eligibility))
        info["excluded_boundary_events"] = len(spans) - info["complete_events"]
        if label_mode:
            rows = []
            for (begin, end), eligible in zip(spans, eligibility):
                hits = np.flatnonzero((predicted[begin:end] == 1) & valid_mask[begin:end])
                rows.append({"begin": int(begin), "end_exclusive": int(end), "length": int(end - begin),
                             "eligible_for_add": eligible, "detected": bool(len(hits)),
                             "first_hit": int(begin + hits[0]) if len(hits) else None,
                             "delay_samples": int(hits[0]) if eligible and len(hits) else int(end - begin) if eligible else None})
            columns = ["begin", "end_exclusive", "length", "eligible_for_add", "detected", "first_hit", "delay_samples"]
            pd.DataFrame(rows, columns=columns).to_csv(directory / "test_events.csv", index=False)
        with (directory / ("evaluation_" + mode + ".json")).open("w", encoding="utf-8") as stream:
            json.dump(info, stream, ensure_ascii=False, indent=2, allow_nan=False)
        return info

    def split_data(self, data: str):
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


# 按配置中的train_test_split比例划分
class FixedDetectScore(AnomalyDetect):
    REQUIRED_FIELDS = ["train_test_split"]

    def split_data(self, series_name):
        data = DataPool().get_pool().get_series(series_name)
        self.data_lens = len(data)
        train_length = int(self.strategy_config["train_test_split"] * self.data_lens)
        train, test = split_before(data, train_length)
        train_data, train_label = (
            train.loc[:, train.columns != "label"],
            train.loc[:, ["label"]],
        )
        test_data, test_label = (
            test.loc[:, train.columns != "label"],
            test.loc[:, ["label"]],
        )
        return train_data, train_label, test_data, test_label

    def detect(self, test_data):
        return self.model.detect_score(test_data)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_score.__all__


# 按配置中的train_test_split比例划分
class FixedDetectLabel(AnomalyDetect):
    REQUIRED_FIELDS = ["train_test_split"]

    def split_data(self, series_name: str):
        data = DataPool().get_pool().get_series(series_name)
        self.data_lens = len(data)
        train_length = int(self.strategy_config["train_test_split"] * self.data_lens)
        train, test = split_before(data, train_length)
        train_data, train_label = (
            train.loc[:, train.columns != "label"],
            train.loc[:, ["label"]],
        )
        test_data, test_label = (
            test.loc[:, train.columns != "label"],
            test.loc[:, ["label"]],
        )
        return train_data, train_label, test_data, test_label

    def detect(self, test_data):
        return self.model.detect_label(test_data)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_label.__all__


# 按该序列元数据中的train_lens，取前段训练、后段测试
class UnFixedDetectScore(AnomalyDetect):
    def split_data(self, series_name: str):
        data = DataPool().get_pool().get_series(series_name)
        data = data.reset_index(drop=True)
        train_length = int(
            DataPool().get_pool().get_series_meta_info(series_name)["train_lens"].item()
        )
        train, test = split_before(data, train_length)
        train_data, train_label = (
            train.loc[:, train.columns != "label"],
            train.loc[:, ["label"]],
        )

        test_data, test_label = (
            test.loc[:, train.columns != "label"],
            test.loc[:, ["label"]],
        )
        return train_data, train_label, test_data, test_label

    def detect(self, test_data):
        return self.model.detect_score(test_data)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_score.__all__


# 按该序列元数据中的train_lens，取前段训练、后段测试
class UnFixedDetectLabel(AnomalyDetect):
    def split_data(self, series_name):
        data = DataPool().get_pool().get_series(series_name)
        data = data.reset_index(drop=True)
        train_length = int(
            DataPool().get_pool().get_series_meta_info(series_name)["train_lens"].item()
        )
        train, test = split_before(data, train_length)
        train_data, train_label = (
            train.loc[:, train.columns != "label"],
            train.loc[:, ["label"]],
        )
        test_data, test_label = (
            test.loc[:, train.columns != "label"],
            test.loc[:, ["label"]],
        )
        return train_data, train_label, test_data, test_label

    def detect(self, test_data):
        return self.model.detect_label(test_data)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_label.__all__


# 整条序列同时用于训练和测试
class AllDetectScore(AnomalyDetect):
    def split_data(self, series_name):
        data = DataPool().get_pool().get_series(series_name)
        train = data
        test = data
        train_data, train_label = train.loc[:, train.columns != "label"], None
        test_data, test_label = (
            test.loc[:, train.columns != "label"],
            test.loc[:, ["label"]],
        )
        return train_data, None, test_data, test_label

    def detect(self, test_data):
        return self.model.detect_score(test_data)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_score.__all__


# 整条序列同时用于训练和测试
class AllDetectLabel(AnomalyDetect):
    def split_data(self, series_name):
        data = DataPool().get_pool().get_series(series_name)
        train = data
        test = data
        train_data, train_label = train.loc[:, train.columns != "label"], None
        test_data, test_label = (
            test.loc[:, train.columns != "label"],
            test.loc[:, ["label"]],
        )
        return train_data, None, test_data, test_label

    def detect(self, test_data):
        return self.model.detect_label(test_data)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_label.__all__
