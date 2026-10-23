"""Check accelerated TAB range/VUS metrics against the original full-scan formulas.

python validation/validate_score_metrics.py
python validation/validate_score_metrics.py --benchmark
"""

from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from ts_benchmark.evaluation.evaluator import Evaluator
from ts_benchmark.evaluation.metrics import classification_metrics_score as score_metrics
from ts_benchmark.evaluation.metrics.vus_metrics import generate_curve, metricor


def legacy_segments(labels):
    result, start = [], None
    for position, value in enumerate(labels):
        if value != 0 and start is None:
            start = position
        if value == 0 and start is not None:
            result.append((start, position - 1))
            start = None
    if start is not None:
        result.append((start, len(labels) - 1))
    return result


def legacy_curve(actual, score, window, percentage=None, *, descending=None):
    """Independent copy of TAB's original extensions and 250 full-array scans."""
    labels = actual.copy().astype(float)
    for start, end in legacy_segments(actual):
        if percentage is None:
            radius, denominator = window // 2, window
        else:
            radius = int((end - start + 1) * percentage)
            denominator = 2 * radius
        right = np.arange(end, min(end + radius, len(labels)))
        labels[right] += np.sqrt(1 - (right - end) / denominator)
        left = np.arange(max(start - radius, 0), start)
        labels[left] += np.sqrt(1 - (start - left) / denominator)
    labels = np.minimum(np.ones(len(labels)), labels)
    ranges = legacy_segments(labels)
    positive = (actual.sum() + labels.sum()) / 2
    tpr, fpr, precision = [0], [0], [1]
    if descending is None:
        descending = -np.sort(-score)
    for index in np.linspace(0, len(score) - 1, 250).astype(int):
        predicted = score >= descending[index]
        product = labels * predicted
        tp = product.sum()
        existence = sum(product[start:end + 1].sum() > 0 for start, end in ranges) / len(ranges)
        tpr.append(min(tp / positive, 1) * existence)
        fpr.append((predicted.sum() - tp) / (len(labels) - positive))
        precision.append(tp / predicted.sum())
    return np.r_[tpr, 1], np.r_[fpr, 1], np.asarray(precision)


def legacy_volume(actual, score, maximum):
    descending = -np.sort(-score)
    curves = [legacy_curve(actual, score, window, descending=descending) for window in range(maximum + 1)]
    aucs, aps = [], []
    for tpr, fpr, precision in curves:
        aucs.append(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
        aps.append(np.sum((tpr[1:-1] - tpr[:-2]) * (precision[1:] + precision[:-1]) / 2))
    return curves, sum(aucs) / len(aucs), sum(aps) / len(aps)


class ScoreMetricChecks(unittest.TestCase):
    def test_range_curves_include_ties_boundaries_merged_events_and_short_series(self):
        rng = np.random.default_rng(4)
        for size in (7, 61, 387):
            labels = (rng.uniform(size=size) < 0.25).astype(float)
            labels[0], labels[1], labels[-1] = 1, 0, 1
            scores = np.round(rng.normal(size=size), 1)
            for values in (scores, np.full(size, -2.0)):
                for window in (0, 1, 2, 3, 9, 2 * size + 1):
                    with self.subTest(size=size, window=window, constant=values is not scores):
                        expected = legacy_curve(labels, values, window)
                        result = metricor().RangeAUC(labels, values, window=window, plot_ROC=True)
                        for got, wanted in zip((result[3], result[2], result[4]), expected):
                            np.testing.assert_allclose(got, wanted, rtol=1e-12, atol=1e-12)

    def test_percentage_extension_preserves_curves(self):
        labels = np.r_[np.ones(6), np.zeros(4), np.ones(7), np.zeros(9)]
        scores = np.random.default_rng(5).normal(size=len(labels))
        for percentage in (0, 0.2, 0.7, 2):
            expected = legacy_curve(labels, scores, 0, percentage)
            got = metricor().RangeAUC(labels, scores, percentage=percentage,
                                      AUC_type="percentage", plot_ROC=True)
            for current, reference in zip((got[3], got[2], got[4]), expected):
                np.testing.assert_allclose(current, reference, rtol=1e-12, atol=1e-12)

    def test_full_vus_surface_and_scalar_mode_match_legacy(self):
        labels = np.zeros(493)
        labels[:6], labels[21:35], labels[45:59], labels[178:205], labels[-11:] = 1, 1, 1, 1, 1
        scores = np.round(np.random.default_rng(9).normal(size=len(labels)), 1)
        maximum = 41
        curves, auc, ap = legacy_volume(labels, scores, maximum)
        full = metricor().RangeAUC_volume(labels, scores, maximum)
        scalar = metricor().RangeAUC_volume(labels, scores, maximum, return_curves=False)
        self.assertEqual([len(part) for part in scalar[:3]], [0, 0, 0])
        np.testing.assert_allclose(full[-2:], [auc, ap], rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(scalar[-2:], full[-2:])
        for index, expected in enumerate(curves):
            for actual, reference in zip((full[0][index], full[1][index], full[2][index]), expected):
                np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-12)
        flattened = generate_curve(labels, scores, maximum)
        np.testing.assert_allclose(flattened[-2:], [auc, ap], rtol=1e-12, atol=1e-12)
        self.assertEqual([len(values) for values in flattened[:6]],
                         [(maximum + 1) * size for size in (252, 252, 252, 251, 251, 251)])

    def test_long_arrays_preserve_cumulative_weight_numerics(self):
        rng = np.random.default_rng(32)
        labels = np.zeros(50_000)
        for start in range(0, len(labels), 700):
            labels[start:start + 213] = 1
        scores = np.round(rng.normal(size=len(labels)), 2)
        for window in (51, 777):
            wanted = legacy_curve(labels, scores, window)
            got = metricor().RangeAUC(labels, scores, window=window, plot_ROC=True)
            for current, reference in zip((got[3], got[2], got[4]), wanted):
                np.testing.assert_allclose(current, reference, rtol=1e-11, atol=1e-11)

    def test_metric_pairs_share_work_only_within_one_evaluation(self):
        labels = np.zeros(193)
        labels[10:17], labels[28:33], labels[61:70] = 1, 1, 1
        rng = np.random.default_rng(62)
        scores = rng.normal(size=len(labels)) + 2 * labels
        evaluator = Evaluator([{"name": name} for name in score_metrics.__all__])
        grader = score_metrics.metricor_grader
        with patch.object(grader, "RangeAUC", wraps=grader.RangeAUC) as range_call, \
             patch.object(grader, "RangeAUC_volume", wraps=grader.RangeAUC_volume) as volume_call:
            first, error = evaluator.evaluate_with_log(labels, scores)
            self.assertEqual(error, "")
            self.assertTrue(np.isfinite(first).all())
            self.assertEqual((range_call.call_count, volume_call.call_count), (1, 1))
            second, error = evaluator.evaluate_with_log(labels, -scores)
            self.assertEqual(error, "")
            self.assertEqual((range_call.call_count, volume_call.call_count), (2, 2))
            self.assertGreater(first[4], second[4])
            self.assertGreater(first[5], second[5])

    def test_progress_and_undefined_metrics_are_reported(self):
        labels, scores = np.r_[0, 1, 1, 0, 0, 1, 0], np.arange(7, dtype=float)
        with self.assertLogs("ts_benchmark.evaluation.metrics.vus_metrics", level="INFO") as logs, \
             patch("ts_benchmark.evaluation.metrics.vus_metrics.time.perf_counter",
                   side_effect=[0, 1, 32, 33, 34, 35]):
            metricor().RangeAUC_volume(labels, scores, 4, return_curves=False)
        self.assertTrue(any("thresholds=250 windows=5 (0..4)" in line for line in logs.output))
        self.assertTrue(any("windows 2/5" in line for line in logs.output))
        self.assertTrue(any("windows 5/5" in line for line in logs.output))
        evaluator = Evaluator([{"name": "VUS_ROC"}, {"name": "VUS_PR"}])
        with self.assertLogs("ts_benchmark.evaluation.evaluator", level="WARNING"):
            result, errors = evaluator.evaluate_with_log(np.zeros(7), scores)
        self.assertTrue(np.isnan(result).all())
        self.assertIn("at least one anomaly segment", errors)
        with self.assertRaisesRegex(ValueError, "same length"):
            metricor().RangeAUC(np.r_[0, 1, 0, 0], scores[:3])
        for labels in (np.array([]), np.zeros(7), np.ones(7), np.r_[0, 1, 0, 1, 1]):
            self.assertEqual(metricor().range_convers_new(labels), legacy_segments(labels))


def benchmark():
    """Brief CPU comparison; synthetic scores are not a model-quality measurement."""
    rng = np.random.default_rng(83)
    labels = np.zeros(20_000)
    for start in range(500, len(labels) - 500, 1400):
        labels[start:start + 64] = 1
    scores = rng.normal(size=len(labels)) + labels
    started = time.perf_counter()
    _, old_auc, old_ap = legacy_volume(labels, scores, 128)
    old_seconds = time.perf_counter() - started
    started = time.perf_counter()
    result = metricor().RangeAUC_volume(labels, scores, 128, return_curves=False)
    new_seconds = time.perf_counter() - started
    np.testing.assert_allclose(result[-2:], [old_auc, old_ap], rtol=1e-11, atol=1e-11)
    print("CPU synthetic metric check: points=20000 windows=129 thresholds=250 "
          "legacy_seconds={:.3f} optimized_seconds={:.3f} max_error={:.3g}".format(
              old_seconds, new_seconds, np.max(np.abs(np.asarray(result[-2:]) - [old_auc, old_ap]))), flush=True)


if __name__ == "__main__":
    if "--benchmark" in sys.argv:
        sys.argv.remove("--benchmark")
        benchmark()
    unittest.main(verbosity=2)
