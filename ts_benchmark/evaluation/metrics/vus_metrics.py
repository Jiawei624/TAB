"""
@article{paparrizos2022volume,
  title={{Volume Under the Surface: A New Accuracy Evaluation Measure for Time-Series Anomaly Detection}},
  author={Paparrizos, John and Boniol, Paul and Palpanas, Themis and Tsay, Ruey S and Elmore, Aaron and Franklin, Michael J},
  journal={Proceedings of the VLDB Endowment},
  volume={15},
  number={11},
  pages={2774--2787},
  year={2022},
  publisher={VLDB Endowment}
}

删除函数metric_new()中把预测分数转为预测标签的代码，直接使用预测分数

"""
from sklearn import metrics
import logging
import numpy as np
import math
import time


logger = logging.getLogger(__name__)


class _RangeScoreIndex:
    """Evaluate the original 250 thresholds without rescanning N points each time."""

    def __init__(self, score):
        self.score = np.asarray(score)
        if self.score.ndim != 1 or not len(self.score) or not np.isfinite(self.score).all():
            raise ValueError("Range AUC requires a nonempty finite one-dimensional score array")
        self.order = np.argsort(-self.score)
        descending = self.score[self.order]
        self.thresholds = descending[np.linspace(0, len(score) - 1, 250).astype(int)]
        # The legacy comparison is >=, so include every tie at each threshold.
        self.counts = np.searchsorted(-descending, -self.thresholds, side="right")

    def curve(self, labels, positive_count, ranges):
        if len(labels) != len(self.score):
            raise ValueError("Range AUC labels and scores must have the same length")
        if not ranges:
            raise ValueError("Range AUC requires at least one anomaly segment")
        weighted_tp = np.cumsum(labels[self.order], dtype=np.float64)[self.counts - 1]
        positive_mass = (positive_count + np.sum(labels)) / 2
        recall = np.minimum(weighted_tp / positive_mass, 1)
        # All weights inside a nonzero segment are positive. A segment is hit
        # exactly when its maximum score reaches the current >= threshold.
        maxima = np.sort([np.max(self.score[begin:end + 1]) for begin, end in ranges])
        existence = (len(ranges) - np.searchsorted(maxima, self.thresholds, side="left")) / len(ranges)
        tpr = np.r_[0, recall * existence, 1]
        fpr = np.r_[0, (self.counts - weighted_tp) / (len(labels) - positive_mass), 1]
        precision = np.r_[1, weighted_tp / self.counts]
        return tpr, fpr, precision


def _range_areas(tpr, fpr, precision):
    # Retain TAB's trapezoids and its final forced ROC endpoint (1, 1).
    auc = np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2)
    ap = np.sum((tpr[1:-1] - tpr[:-2]) * (precision[1:] + precision[:-1]) / 2)
    return auc, ap


class metricor:
    def __init__(
        self, a=1, probability=True, bias="flat",
    ):
        self.a = a
        self.probability = probability
        self.bias = bias

    def detect_model(
        self, model, label, contamination=0.1, window=100, is_A=False, is_threshold=True
    ):
        if is_threshold:
            score = self.scale_threshold(
                model.decision_scores_, model._mu, model._sigma
            )
        else:
            score = self.scale_contamination(
                model.decision_scores_, contamination=contamination
            )
        if is_A is False:
            scoreX = np.zeros(len(score) + window)
            scoreX[
                math.ceil(window / 2) : len(score) + window - math.floor(window / 2)
            ] = score
        else:
            scoreX = score

        self.score_ = scoreX
        L = self.metric(label, scoreX)
        return L

    def labels_conv(self, preds):
        """return indices of predicted anomaly
        """

        index = np.where(preds >= 0.5)
        return index[0]

    def labels_conv_binary(self, preds):
        """return predicted label
        """
        p = np.zeros(len(preds))
        index = np.where(preds >= 0.5)
        p[index[0]] = 1
        return p

    def w(self, AnomalyRange, p):
        MyValue = 0
        MaxValue = 0
        start = AnomalyRange[0]
        AnomalyLength = AnomalyRange[1] - AnomalyRange[0] + 1
        for i in range(start, start + AnomalyLength):
            bi = self.b(i, AnomalyLength)
            MaxValue += bi
            if i in p:
                MyValue += bi
        return MyValue / MaxValue

    def Cardinality_factor(self, Anomolyrange, Prange):
        score = 0
        start = Anomolyrange[0]
        end = Anomolyrange[1]
        for i in Prange:
            if i[0] >= start and i[0] <= end:
                score += 1
            elif start >= i[0] and start <= i[1]:
                score += 1
            elif end >= i[0] and end <= i[1]:
                score += 1
            elif start >= i[0] and end <= i[1]:
                score += 1
        if score == 0:
            return 0
        else:
            return 1 / score

    def b(self, i, length):
        bias = self.bias
        if bias == "flat":
            return 1
        elif bias == "front-end bias":
            return length - i + 1
        elif bias == "back-end bias":
            return i
        else:
            if i <= length / 2:
                return i
            else:
                return length - i + 1

    def scale_threshold(self, score, score_mu, score_sigma):
        return (score >= (score_mu + 3 * score_sigma)).astype(int)

    def metric_new(self, label, score, plot_ROC=False, alpha=0.2):
        """input:
               Real labels and anomaly score in prediction
            
           output:
               AUC, 
               Precision, 
               Recall, 
               F-score, 
               Range-precision, 
               Range-recall, 
               Range-Fscore, 
               Precison@k, 
             
            k is chosen to be # of outliers in real labels
        """
        if np.sum(label) == 0:
            print(
                "All labels are 0. Label must have groud truth value for calculating AUC score."
            )
            return None

        if np.isnan(score).any() or score is None:
            print("Score must not be none.")
            return None

        # area under curve
        auc = metrics.roc_auc_score(label, score)
        # plor ROC curve
        if plot_ROC:
            fpr, tpr, thresholds = metrics.roc_curve(label, score)
            # display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc)
            # display.plot()

        # precision, recall, F
        preds = score
        # preds = score > (np.mean(score) + 3 * np.std(score))
        Precision, Recall, F, Support = metrics.precision_recall_fscore_support(
            label, preds, zero_division=0
        )
        precision = Precision[1]
        recall = Recall[1]
        f = F[1]

        # range anomaly
        Rrecall, ExistenceReward, OverlapReward = self.range_recall_new(
            label, preds, alpha
        )
        Rprecision = self.range_recall_new(preds, label, 0)[0]

        if Rprecision + Rrecall == 0:
            Rf = 0
        else:
            Rf = 2 * Rrecall * Rprecision / (Rprecision + Rrecall)

        # top-k
        k = int(np.sum(label))
        threshold = np.percentile(score, 100 * (1 - k / len(label)))

        # precision_at_k = metrics.top_k_accuracy_score(label, score, k)
        p_at_k = np.where(preds > threshold)[0]
        TP_at_k = sum(label[p_at_k])
        precision_at_k = TP_at_k / k

        L = [
            auc,
            precision,
            recall,
            f,
            Rrecall,
            ExistenceReward,
            OverlapReward,
            Rprecision,
            Rf,
            precision_at_k,
        ]
        if plot_ROC:
            return L, fpr, tpr
        return L

    def metric_PR(self, label, score):
        precision, recall, thresholds = metrics.precision_recall_curve(label, score)
        # plt.figure()
        # disp = metrics.PrecisionRecallDisplay(precision=precision, recall=recall)
        # disp.plot()
        AP = metrics.average_precision_score(label, score)
        return precision, recall, AP

    def range_recall_new(self, labels, preds, alpha):

        p = np.where(preds == 1)[0]  # positions of predicted label==1
        range_pred = self.range_convers_new(preds)
        range_label = self.range_convers_new(labels)

        Nr = len(range_label)  # total # of real anomaly segments

        ExistenceReward = self.existence_reward(range_label, p)

        OverlapReward = 0
        for i in range_label:
            OverlapReward += self.w(i, p) * self.Cardinality_factor(i, range_pred)

        score = alpha * ExistenceReward + (1 - alpha) * OverlapReward
        if Nr != 0:
            return score / Nr, ExistenceReward / Nr, OverlapReward / Nr
        else:
            return 0, 0, 0

    def range_convers_new(self, label):
        """
        input: arrays of binary values 
        output: list of ordered pair [[a0,b0], [a1,b1]... ] of the inputs
        """
        values = np.asarray(label)
        if values.ndim != 1:
            raise ValueError("Anomaly segments require one-dimensional labels")
        edges = np.flatnonzero(np.diff(np.r_[False, values != 0, False]))
        return list(zip(edges[::2].tolist(), (edges[1::2] - 1).tolist()))

    def existence_reward(self, labels, preds):
        """
        labels: list of ordered pair 
        preds predicted data
        """

        score = 0
        for i in labels:
            if np.sum(np.multiply(preds <= i[1], preds >= i[0])) > 0:
                score += 1
        return score

    def num_nonzero_segments(self, x):
        count = 0
        if x[0] > 0:
            count += 1
        for i in range(1, len(x)):
            if x[i] > 0 and x[i - 1] == 0:
                count += 1
        return count

    def extend_postive_range(self, x, window=5, *, ranges=None):
        label = x.copy().astype(float)
        L = self.range_convers_new(label) if ranges is None else ranges
        length = len(label)
        for k in range(len(L)):
            s = L[k][0]
            e = L[k][1]

            x1 = np.arange(e, min(e + window // 2, length))
            label[x1] += np.sqrt(1 - (x1 - e) / (window))

            x2 = np.arange(max(s - window // 2, 0), s)
            label[x2] += np.sqrt(1 - (s - x2) / (window))

        np.minimum(label, 1, out=label)
        return label

    def extend_postive_range_individual(self, x, percentage=0.2):
        label = x.copy().astype(float)
        L = self.range_convers_new(label)  # index of non-zero segments
        length = len(label)
        for k in range(len(L)):
            s = L[k][0]
            e = L[k][1]

            l0 = int((e - s + 1) * percentage)

            x1 = np.arange(e, min(e + l0, length))
            label[x1] += np.sqrt(1 - (x1 - e) / (2 * l0))

            x2 = np.arange(max(s - l0, 0), s)
            label[x2] += np.sqrt(1 - (s - x2) / (2 * l0))

        label = np.minimum(np.ones(length), label)
        return label

    def TPR_FPR_RangeAUC(self, labels, pred, P, L):
        product = labels * pred 

        TP = np.sum(product)

        # recall = min(TP/P,1)
        P_new = (P + np.sum(labels)) / 2  # so TPR is neither large nor small
        # P_new = np.sum(labels)
        recall = min(TP / P_new, 1)
        # recall = TP/np.sum(labels)
        # print('recall '+str(recall))

        existence = 0
        for seg in L:
            if np.sum(product[seg[0] : (seg[1] + 1)]) > 0:
                existence += 1

        existence_ratio = existence / len(L)
        # print(existence_ratio)

        # TPR_RangeAUC = np.sqrt(recall*existence_ratio)
        # print(existence_ratio)
        TPR_RangeAUC = recall * existence_ratio

        FP = np.sum(pred) - TP
        # TN = np.sum((1-pred) * (1-labels))

        # FPR_RangeAUC = FP/(FP+TN)
        N_new = len(labels) - P_new
        FPR_RangeAUC = FP / N_new

        Precision_RangeAUC = TP / np.sum(pred)

        return TPR_RangeAUC, FPR_RangeAUC, Precision_RangeAUC

    def RangeAUC(
        self, labels, score, window=0, percentage=0, plot_ROC=False, AUC_type="window",
        *, _score_index=None,
    ):
        # AUC_type='window'/'percentage'
        score_index = _RangeScoreIndex(score) if _score_index is None else _score_index

        P = np.sum(labels)
        # print(np.sum(labels))
        if AUC_type == "window":
            labels = self.extend_postive_range(labels, window=window)
        else:
            labels = self.extend_postive_range_individual(labels, percentage=percentage)

        # print(np.sum(labels))
        L = self.range_convers_new(labels)
        tpr, fpr, prec = score_index.curve(labels, P, L)
        AUC_range, AP_range = _range_areas(tpr, fpr, prec)

        if plot_ROC:
            return AUC_range, AP_range, fpr, tpr, prec

        return AUC_range

    # TPR_FPR_window
    def RangeAUC_volume(self, labels_original, score, windowSize, *, return_curves=True, _score_index=None):
        score_index = _RangeScoreIndex(score) if _score_index is None else _score_index

        tpr_3d = []
        fpr_3d = []
        prec_3d = []

        auc_3d = []
        ap_3d = []

        window_3d = np.arange(0, windowSize + 1, 1)
        P = np.sum(labels_original)
        ranges = self.range_convers_new(labels_original)
        started = last_logged = time.perf_counter()
        logger.info("Computing VUS surface: points=%d thresholds=250 windows=%d (0..%d)",
                    len(score), len(window_3d), windowSize)

        for completed, window in enumerate(window_3d, 1):
            labels = self.extend_postive_range(labels_original, window, ranges=ranges)

            # print(np.sum(labels))
            L = self.range_convers_new(labels)
            tpr, fpr, prec = score_index.curve(labels, P, L)
            if return_curves:
                tpr_3d.append(tpr)
                fpr_3d.append(fpr)
                prec_3d.append(prec)

            AUC_range, AP_range = _range_areas(tpr, fpr, prec)
            auc_3d.append(AUC_range)
            ap_3d.append(AP_range)
            now = time.perf_counter()
            if completed == 1 or completed == len(window_3d) or now - last_logged >= 30:
                elapsed = now - started
                eta = elapsed * (len(window_3d) - completed) / completed
                logger.info("VUS surface windows %d/%d (%.1f%%) elapsed=%.1fs ETA=%.1fs",
                            completed, len(window_3d), 100 * completed / len(window_3d), elapsed, eta)
                last_logged = now

        return (
            tpr_3d,
            fpr_3d,
            prec_3d,
            window_3d,
            sum(auc_3d) / len(window_3d),
            sum(ap_3d) / len(window_3d),
        )


def generate_curve(label, score, slidingWindow):
    (
        tpr_3d,
        fpr_3d,
        prec_3d,
        window_3d,
        avg_auc_3d,
        avg_ap_3d,
    ) = metricor().RangeAUC_volume(
        labels_original=label, score=score, windowSize=1 * slidingWindow
    )

    X = np.array(tpr_3d).reshape(1, -1).ravel()
    X_ap = np.array(tpr_3d)[:, :-1].reshape(1, -1).ravel()
    Y = np.array(fpr_3d).reshape(1, -1).ravel()
    W = np.array(prec_3d).reshape(1, -1).ravel()
    Z = np.repeat(window_3d, len(tpr_3d[0]))
    Z_ap = np.repeat(window_3d, len(tpr_3d[0]) - 1)

    return Y, Z, X, X_ap, W, Z_ap, avg_auc_3d, avg_ap_3d
