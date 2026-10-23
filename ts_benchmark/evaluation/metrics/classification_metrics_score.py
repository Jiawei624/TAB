# -*- coding: utf-8 -*-

import logging
import numpy as np
from sklearn import metrics
from ts_benchmark.evaluation.metrics.vus_metrics import metricor, _RangeScoreIndex
from ts_benchmark.evaluation.metrics.utils import get_list_anomaly, find_length
from sklearn.metrics import precision_recall_curve

__all__ = ["auc_roc", "auc_pr", "R_AUC_ROC", "R_AUC_PR", "VUS_ROC", "VUS_PR"]
# __all__ = ["best_ratio", "best_accuracy", "best_f_score", "best_precision", "best_recall", "auc_roc", "auc_pr", "R_AUC_ROC", "R_AUC_PR", "VUS_ROC", "VUS_PR"]


metricor_grader = metricor()
logger = logging.getLogger(__name__)

def best_ratio(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    precisions, recalls, thresholds = precision_recall_curve(actual, predicted)

    # Calculate F1 scores
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)

    # Find the index of the best F1 score
    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])

    # Get the best F1 score and the corresponding threshold
    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    best_threshold = thresholds[best_f1_score_index]

    # Create binary predicted labels based on the best threshold
    predicted_labels = [1 if p >= best_threshold else 0 for p in predicted]

    # Calculate confusion matrix components
    true_positives = sum((a == 1 and p == 1) for a, p in zip(actual, predicted_labels))
    false_positives = sum((a == 0 and p == 1) for a, p in zip(actual, predicted_labels))
    false_negatives = sum((a == 1 and p == 0) for a, p in zip(actual, predicted_labels))
    true_negatives = sum((a == 0 and p == 0) for a, p in zip(actual, predicted_labels))

    # Calculate precision, recall, and accuracy
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    accuracy = (true_positives + true_negatives) / len(actual)
    anomaly_rate = sum(predicted_labels) / len(actual)

    return anomaly_rate
def best_f_score(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    precisions, recalls, thresholds = precision_recall_curve(actual, predicted)

    # Calculate F1 scores
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)

    # Find the index of the best F1 score
    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])

    # Get the best F1 score and the corresponding threshold
    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    best_threshold = thresholds[best_f1_score_index]

    # Create binary predicted labels based on the best threshold
    predicted_labels = [1 if p >= best_threshold else 0 for p in predicted]

    # Calculate confusion matrix components
    true_positives = sum((a == 1 and p == 1) for a, p in zip(actual, predicted_labels))
    false_positives = sum((a == 0 and p == 1) for a, p in zip(actual, predicted_labels))
    false_negatives = sum((a == 1 and p == 0) for a, p in zip(actual, predicted_labels))
    true_negatives = sum((a == 0 and p == 0) for a, p in zip(actual, predicted_labels))

    # Calculate precision, recall, and accuracy
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    accuracy = (true_positives + true_negatives) / len(actual)
    anomaly_rate = sum(predicted_labels) / len(actual)

    return best_f1_score


def best_accuracy(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    precisions, recalls, thresholds = precision_recall_curve(actual, predicted)

    # Calculate F1 scores
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)

    # Find the index of the best F1 score
    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])

    # Get the best F1 score and the corresponding threshold
    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    best_threshold = thresholds[best_f1_score_index]

    # Create binary predicted labels based on the best threshold
    predicted_labels = [1 if p >= best_threshold else 0 for p in predicted]

    # Calculate confusion matrix components
    true_positives = sum((a == 1 and p == 1) for a, p in zip(actual, predicted_labels))
    false_positives = sum((a == 0 and p == 1) for a, p in zip(actual, predicted_labels))
    false_negatives = sum((a == 1 and p == 0) for a, p in zip(actual, predicted_labels))
    true_negatives = sum((a == 0 and p == 0) for a, p in zip(actual, predicted_labels))

    # Calculate precision, recall, and accuracy
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    accuracy = (true_positives + true_negatives) / len(actual)
    anomaly_rate = sum(predicted_labels) / len(actual)

    return accuracy


def best_recall(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    precisions, recalls, thresholds = precision_recall_curve(actual, predicted)

    # Calculate F1 scores
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)

    # Find the index of the best F1 score
    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])

    # Get the best F1 score and the corresponding threshold
    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    best_threshold = thresholds[best_f1_score_index]

    # Create binary predicted labels based on the best threshold
    predicted_labels = [1 if p >= best_threshold else 0 for p in predicted]

    # Calculate confusion matrix components
    true_positives = sum((a == 1 and p == 1) for a, p in zip(actual, predicted_labels))
    false_positives = sum((a == 0 and p == 1) for a, p in zip(actual, predicted_labels))
    false_negatives = sum((a == 1 and p == 0) for a, p in zip(actual, predicted_labels))
    true_negatives = sum((a == 0 and p == 0) for a, p in zip(actual, predicted_labels))

    # Calculate precision, recall, and accuracy
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    accuracy = (true_positives + true_negatives) / len(actual)
    anomaly_rate = sum(predicted_labels) / len(actual)

    return recall


def best_precision(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    precisions, recalls, thresholds = precision_recall_curve(actual, predicted)

    # Calculate F1 scores
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)

    # Find the index of the best F1 score
    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])

    # Get the best F1 score and the corresponding threshold
    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    best_threshold = thresholds[best_f1_score_index]

    # Create binary predicted labels based on the best threshold
    predicted_labels = [1 if p >= best_threshold else 0 for p in predicted]

    # Calculate confusion matrix components
    true_positives = sum((a == 1 and p == 1) for a, p in zip(actual, predicted_labels))
    false_positives = sum((a == 0 and p == 1) for a, p in zip(actual, predicted_labels))
    false_negatives = sum((a == 1 and p == 0) for a, p in zip(actual, predicted_labels))
    true_negatives = sum((a == 0 and p == 0) for a, p in zip(actual, predicted_labels))

    # Calculate precision, recall, and accuracy
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    accuracy = (true_positives + true_negatives) / len(actual)
    anomaly_rate = sum(predicted_labels) / len(actual)

    return precision

def auc_roc(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    return metrics.roc_auc_score(actual, predicted)


def auc_pr(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    return metrics.average_precision_score(actual, predicted)


def _range_metric_pair(actual, predicted, kwargs, *, volume):
    """Cache only within one Evaluator call; never reuse another series' scores."""
    cache = kwargs.get("_metric_cache")
    if cache is None:
        cache = {}
    key = "vus_pair" if volume else "range_auc_pair"
    if key in cache:
        logger.info("Reusing shared %s ROC/PR calculation", "VUS" if volume else "range AUC")
        return cache[key]
    if "anomaly_window" not in cache:
        lengths = get_list_anomaly(actual)
        if not len(lengths):
            raise ValueError("Range/VUS metrics require at least one anomaly segment")
        cache["anomaly_window"] = int(np.median(lengths))
    if "range_score_index" not in cache:
        cache["range_score_index"] = _RangeScoreIndex(predicted)
    window, score_index = cache["anomaly_window"], cache["range_score_index"]
    if volume:
        # Metric consumers need the two integrals, not all plotting surfaces.
        result = metricor_grader.RangeAUC_volume(
            actual, predicted, 2 * window, return_curves=False, _score_index=score_index)
        pair = result[-2:]
    else:
        result = metricor_grader.RangeAUC(
            actual, predicted, window=window, plot_ROC=True, _score_index=score_index)
        pair = result[:2]
    cache[key] = pair
    return pair


def R_AUC_ROC(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    return _range_metric_pair(actual, predicted, kwargs, volume=False)[0]


def R_AUC_PR(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    return _range_metric_pair(actual, predicted, kwargs, volume=False)[1]


def VUS_ROC(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    return _range_metric_pair(actual, predicted, kwargs, volume=True)[0]


def VUS_PR(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    return _range_metric_pair(actual, predicted, kwargs, volume=True)[1]


# y_test = np.array([1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1])
# pred_labels = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1])
# a = VUS_ROC(y_test, pred_labels)
# b = VUS_PR(y_test, pred_labels)
# # vus_results = get_range_vus_roc(y_test, pred_labels, 100)  # default slidingWindow = 100
# print("VUS_ROC", a)
# print("VUS_PR", b)

#
# import numpy as np
#
#
# score = np.array([0, 0, 0, 0, 1, 1, 1, 1])
# label = np.array([0, 1, 1, 0, 1, 1, 1, 1])
# print("auc_roc:", auc_roc(label, score, label))
# print("auc_pr:", auc_pr(label, score, label))
# print("R_AUC_ROC:", R_AUC_ROC(label, score, label))
# print("R_AUC_PR:", R_AUC_PR(label, score, label))
#
# print("VUS_ROC:", VUS_ROC(label, score, label))
# print("VUS_PR:", VUS_PR(label, score, label))
