import numpy as np
from scipy import sparse
from pandas import Series, DataFrame

# Compatibility replacements for missing Snorkel functions
def print_scores(tp, fp, tn, fn, title="Scores"):
    """Compatibility replacement for snorkel.learning.print_scores"""
    total = tp + fp + tn + fn
    if total == 0:
        print(f"{title}: No data")
        return
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"{title}:")
    print(f"  TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

def matrix_coverage(L):
    """Compatibility replacement for snorkel.utils.matrix_coverage"""
    # Coverage = fraction of examples that each LF labels (non-zero)
    if len(L.shape) != 2:
        return np.array([])
    return np.mean(L != 0, axis=1)

def matrix_conflicts(L):
    """Compatibility replacement for snorkel.utils.matrix_conflicts"""
    # Conflicts = fraction of examples where this LF disagrees with others
    if len(L.shape) != 2:
        return np.array([])
    
    conflicts = np.zeros(L.shape[0])
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            if L[i, j] != 0:  # If LF i labels example j
                # Check if any other LF disagrees
                other_labels = L[:, j]
                other_labels = other_labels[other_labels != 0]  # Remove abstains
                if len(other_labels) > 1:  # If multiple LFs label this example
                    if not np.all(other_labels == L[i, j]):  # If there's disagreement
                        conflicts[i] += 1
    
    # Normalize by number of examples each LF labels
    labeled_counts = np.sum(L != 0, axis=1)
    conflicts = conflicts / np.maximum(labeled_counts, 1)  # Avoid division by zero
    
    return conflicts


def get_labeling_functions_matrix(L_fns, cands_loader, priority_tests_func=None):
    """
    Return a labeling matrix of the given labeling functions to the given candidates.
    L[i, j] is the labeling of the ith labeling function to the jth candidate
    Args:
        L_fns:
        cands_loader:
        priority_tests_func: A function that returns a Boolean result of whether the priority tests passed or not

    Returns:

    """
    L = np.zeros((len(L_fns),cands_loader.num_of_data_elements)).astype(int)
    for j, snorkel_data_obj in enumerate(cands_loader):
        if priority_tests_func is not None:
            priority_tests_success = priority_tests_func(snorkel_data_obj)
        else:
            priority_tests_success = True
        for i, L_fn in enumerate(L_fns):
            if priority_tests_success:
                L[i, j]  = L_fn(snorkel_data_obj)
            else:
                L[i, j] = -1
    return L


def matrix_tp(L, labels):
    return np.ravel([
        np.sum(np.ravel((L[:, j] > 0)) * (labels > 0)) for j in range(L.shape[1])
    ])


def matrix_fp(L, labels):
    return np.ravel([
        np.sum(np.ravel((L[:, j] > 0)) * (labels < 0)) for j in range(L.shape[1])
    ])


def matrix_tn(L, labels):
    return np.ravel([
        np.sum(np.ravel((L[:, j] < 0)) * (labels < 0)) for j in range(L.shape[1])
    ])


def matrix_fn(L, labels):
    return np.ravel([
        np.sum(np.ravel((L[:, j] < 0)) * (labels > 0)) for j in range(L.shape[1])
    ])


def matrix_overlaps(L):
    """
    Given an N x M matrix where L_{i,j} is the label given by the jth LF to the ith candidate:
    Return the **fraction of candidates that each LF _overlaps with other LFs on_.**
    """
    L_nonzero = L != 0
    x = np.where(L_nonzero.sum(axis=1) > 1, 1, 0)
    x = np.expand_dims(x, axis=0)
    y = L_nonzero
    return np.ravel(np.matmul(x, y) / float(L.shape[0]))


def get_pos_and_neg_probs_from_human_labels(labels):
    """

    Args:
        labels: A list of labels in the scale [-1, -0.5, 0, 0.5, 1.0] standing for
         * 100% inhumnae,
         * does not seem humane,
         * don't know,
         * seems humane,
         * 100% humane

    Returns: ndaraay P of shape len(labels) X 2 where
    P[i,0] = probability for positive label for sample i
    P[i,1] = probability for negative label for sample i

    """

    result = np.zeros(shape=(len(labels), 2))
    for i, label in enumerate(labels):
        pos = (label + 1) / 2
        neg = 1 - pos
        result[i, 0] = pos
        result[i, 1] = neg

    return result


def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray
    Returns: scalar
    """
    predictions_positive = np.clip(predictions, epsilon, 1. - epsilon)
    predictions_negative = np.clip(1-predictions_positive, epsilon, 1. - epsilon)
    predictions = np.column_stack((predictions_positive, predictions_negative))

    targets = get_pos_and_neg_probs_from_human_labels(targets)

    N = len(predictions)
    ce = -np.sum(targets * np.log(predictions+1e-9))/N
    return ce


class SnorkelStatsObj(object):
    def __init__(self, lfs, L):
        """

        Args:
            lfs: A list of labeling functions
            L: A labeling matrix (e.g. one that is created using get_labeling_functions_matrix())
        """
        self.lfs = lfs
        self.L = L.T

    def lfs_stats(self, labels=None, est_accs=None):
        """Returns a pandas DataFrame with the LFs and various per-LF statistics"""
        lf_names = [lf.__name__ for lf in self.lfs]

        # Default LF stats
        col_names = ['j', 'Coverage', 'Overlaps', 'Conflicts']
        d = {
            'j': list(range(len(self.lfs))),
            'Coverage': Series(data=matrix_coverage(self.L), index=lf_names),
            'Overlaps': Series(data=matrix_overlaps(self.L), index=lf_names),
            'Conflicts': Series(data=matrix_conflicts(self.L), index=lf_names)
        }
        if labels is not None:
            col_names.extend(['TP', 'FP', 'FN', 'TN', 'Empirical Acc.'])
            flattened_labels = np.ravel(labels.todense() if sparse.issparse(labels) else labels)
            tp = matrix_tp(self.L, flattened_labels)
            fp = matrix_fp(self.L, flattened_labels)
            fn = matrix_fn(self.L, flattened_labels)
            tn = matrix_tn(self.L, flattened_labels)
            ac = (tp + tn) / (tp + tn + fp + fn)
            d['Empirical Acc.'] = Series(data=ac, index=lf_names)
            d['TP'] = Series(data=tp, index=lf_names)
            d['FP'] = Series(data=fp, index=lf_names)
            d['FN'] = Series(data=fn, index=lf_names)
            d['TN'] = Series(data=tn, index=lf_names)

        if est_accs is not None:
            col_names.append('Learned Acc.')
            d['Learned Acc.'] = est_accs
            d['Learned Acc.'].index = lf_names
        return DataFrame(data=d, index=lf_names)[col_names]


class LfStatsObj(object):
    """

    A Class for getting statistical properties of labeling functions in terms of true positive, false positives etc.
    """
    def __init__(self, lf, cands, labels, generative_model_prediction=False):
        self.lf = lf
        self.cands = cands
        self.labels = labels
        # A Boolean saying if we used the generative model to get the labels or not
        self.generative_model_prediction = generative_model_prediction

        self._tp = self._fp = self._tn = self._fn = None

    @property
    def tp(self):
        if self._tp is None:
            self.test_LF()
        return self._tp

    @property
    def fp(self):
        if self._fp is None:
            self.test_LF()
        return self._fp

    @property
    def tn(self):
        if self._tn is None:
            self.test_LF()
        return self._tn

    @property
    def fn(self):
        if self._fn is None:
            self.test_LF()
        return self._fn

    def test_LF(self):
        """
        Computes TP, FP, TN, FN for the labeling function in self.lf
        Returns:

        """
        lf, cands, labels = self.lf, self.cands, self.labels

        tp = set()
        fp = set()
        tn = set()
        fn = set()

        lf_scores = [lf(c) for c in cands]
        assert len(cands) == len(labels) == len(lf_scores)

        for i, (cand, label, lf_score) in enumerate(zip(cands, labels, lf_scores)):
            if self.generative_model_prediction:
                # The generative model output values in range [0, 1]  and we want it to be in [-1, 1]
                lf_score = lf_score * 2 - 1

            # Bucket the candidates for error analysis
            if label != 0:
                if lf_score > 0:
                    if label > 0:
                        tp.add(cand)
                    else:
                        fp.add(cand)
                elif lf_score < 0:
                    if label < 0:
                        tn.add(cand)
                    else:
                        fn.add(cand)

        self._tp = tp
        self._fp = fp
        self._tn = tn
        self._fn = fn

        return tp, fp, tn, fn

    def display_lf_stats(self):
        tp, fp, tn, fn = self.test_LF()
        # Calculate scores
        print_scores(len(tp), len(fp), len(tn), len(fn),
            title=f"Scores for L_fn {self.lf.__name__}")

    def display_html_analysis_ui(self):
        raise NotImplementedError


