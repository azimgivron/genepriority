"""
Metrics module
==============

Provides evaluation metrics for gene prioritization models. Includes:
  - BEDROC: emphasizes early retrieval of true positives.
  - AUC: classical ranking metric adapted to partial gene sets.

Both functions accept only the evaluated subset of genes but normalize
scores against the total gene universe.
"""
import math
from typing import Callable, List, Tuple

import numpy as np
from sklearn import metrics


def _rie_helper(labels_sorted: np.ndarray, alpha: float) -> Tuple[float, int]:
    """
    Compute the raw RIE (Robust Initial Enhancement) value for a single column.
    from https://github.com/rdkit/rdkit/blob/master/rdkit/ML/Scoring/Scoring.py#L66

    Args:
        labels_sorted (np.ndarray): Array of binary labels (1 positives, 0 negatives),
            sorted by predicted score descending.
        alpha (float): Early-recognition parameter (must be > 0).

    Raises:
        ValueError: If `labels_sorted` is empty or `alpha` ≤ 0.

    Returns:
        Tuple[float, int]:
            - rie_value: Unnormalized RIE score for this column.
            - num_actives: Number of positive instances in this column.
    """
    num_molecules = len(labels_sorted)
    alpha = float(alpha)

    if num_molecules == 0:
        raise ValueError("Label list is empty")
    if alpha <= 0.0:
        raise ValueError("Alpha must be greater than zero")

    # normalization denominator
    denom = (
        1.0
        / num_molecules
        * (1 - math.exp(-alpha))
        / (math.exp(alpha / num_molecules) - 1)
    )

    # count actives
    num_actives = int(np.sum(labels_sorted))
    if num_actives == 0:
        return 0.0, 0

    # accumulate exponentials for active instances
    sum_exp = 0.0
    for rank, lab in enumerate(labels_sorted, start=1):
        if lab:
            sum_exp += math.exp(-(alpha * rank) / num_molecules)

    rie_value = sum_exp / (num_actives * denom)
    return rie_value, num_actives


def bedroc_score(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 20.0) -> float:
    """
    Compute the BEDROC (Boltzmann-Enhanced Discrimination of ROC) score.

    This metric emphasizes early recognition of positives in a ranked list
    of predictions, as described by Truchon and Bayley.
    from https://github.com/rdkit/rdkit/blob/master/rdkit/ML/Scoring/Scoring.py

    References:
        Truchon, J.-F., & Bayly, C. I. (2007).
        Evaluating virtual screening methods: good and bad metrics for the "early
        recognition" problem. _Journal of Chemical Information and Modeling_, 47(2), 488–508.
        DOI: 10.1021/ci600426e

    Args:
        y_true (np.ndarray): Binary ground-truth labels (1 for positive, 0 for negative).
        y_pred (np.ndarray): Predicted scores (higher = more likely positive).
        alpha (float): Early-recognition parameter (controls weighting).

    Returns:
        float:
            BEDROC score in [0, 1], with higher values indicating better early recognition.
    """
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("Number of predictions must match number of true labels")

    # sort by descending predicted score
    sorted_indices = np.argsort(-y_pred)
    labels_sorted = y_true[sorted_indices]

    rie_value, num_actives = _rie_helper(labels_sorted, alpha)

    if num_actives == 0:
        return 0.0

    num_molecules = len(labels_sorted)
    active_ratio = num_actives / num_molecules

    rie_max = (1 - math.exp(-alpha * active_ratio)) / (
        active_ratio * (1 - math.exp(-alpha))
    )
    rie_min = (1 - math.exp(alpha * active_ratio)) / (
        active_ratio * (1 - math.exp(alpha))
    )

    # normalize to [0,1]
    if rie_max != rie_min:
        score = (rie_value - rie_min) / (rie_max - rie_min)
        score = np.round(score, 10)
        assert 0 <= score <= 1, f"BEDROC must be in [0;1]. Found => {score}"
        return score
    return 1.0


def bedroc_per_disease(
    y_true: List[np.ndarray], y_pred: List[np.ndarray], gene_number: int, alpha: float
) -> List[float]:
    """Compute the BEDROC score for each diseases.

    BEDROC (Boltzmann-Enhanced Discrimination of the ROC) places exponential
    emphasis on ranking true positives early in the prediction list.

    Args:
        y_true (List[np.ndarray]):
            Binary labels for the evaluated genes (1 = positive, 0 = negative).
        y_pred (List[np.ndarray]):
            Model scores for those genes; higher means more likely positive.
        gene_number (int):
            Number of genes.
        alpha (float):
            Early-recognition weight parameter (> 0).

    Returns:
        List[float]: BEDROC in [0,1] (0=random, 1=perfect).
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0; got {alpha}.")
    scores = []
    mask = []
    for labels, scores_pred in zip(y_true, y_pred):
        # number of positives for this disease
        n_pos = int(labels.sum())
        mask.append(n_pos > 0)

        if n_pos in (0, gene_number):
            # undefined if no positives or no negatives
            scores.append(np.nan)
            continue
        bedroc = bedroc_score(labels, scores_pred, alpha=alpha)
        scores.append(bedroc)
    return scores, mask


def auc_per_disease(
    y_true: List[np.ndarray], y_pred: List[np.ndarray], gene_number: int
) -> Tuple[List[float], List[bool]]:
    """Compute the true AUC score for each disease.

    Args:
        y_true: List of 1D arrays of 0/1 labels.
        y_pred: List of 1D arrays of scores (same length as each y_true[i]).
        gene_number: Number of genes.

    Returns:
        scores: List of AUC floats (nan if no positives or no negatives).
        mask:   List of booleans indicating which diseases had ≥1 positive.
    """
    return metric_per_disease(y_true, y_pred, gene_number, metrics.roc_auc_score)


def avg_precision_per_disease(
    y_true: List[np.ndarray], y_pred: List[np.ndarray], gene_number: int
) -> Tuple[List[float], List[bool]]:
    """Compute the average Precision score for each disease.

    Args:
        y_true: List of 1D arrays of 0/1 labels.
        y_pred: List of 1D arrays of scores (same length as each y_true[i]).
        gene_number: Number of genes.

    Returns:
        scores: List of Average Precision floats (nan if no positives or no negatives).
        mask:   List of booleans indicating which diseases had ≥1 positive.
    """
    return metric_per_disease(
        y_true, y_pred, gene_number, metrics.average_precision_score
    )


def metric_per_disease(
    y_true: List[np.ndarray], y_pred: List[np.ndarray], gene_number: int, func: Callable
) -> Tuple[List[float], List[bool]]:
    """Compute the metric score from func for each disease.

    Args:
        y_true: List of 1D arrays of 0/1 labels.
        y_pred: List of 1D arrays of scores (same length as each y_true[i]).
        gene_number: Number of genes.

    Returns:
        scores: List of metric (nan if no positives or no negatives).
        mask:   List of booleans indicating which diseases had ≥1 positive.
    """
    scores = []
    mask = []

    for labels, scores_pred in zip(y_true, y_pred):
        # number of positives for this disease
        n_pos = int(labels.sum())
        mask.append(n_pos > 0)

        if n_pos in (0, gene_number):
            # undefined if no positives or no negatives
            scores.append(np.nan)
            continue

        metric = func(labels, scores_pred)
        scores.append(metric)
    return scores, mask


def roc_per_disease(
    y_true: List[np.ndarray], y_pred: List[np.ndarray], gene_number: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the ROC curve averaged over diseases.

    Args:
        y_true: List of 1D arrays of 0/1 labels.
        y_pred: List of 1D arrays of scores (same length as each y_true[i]).
        gene_number: Number of genes.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - 2D array of interpolated mean Precision/Recall values across diseases
                (shape: [2, n_thresholds]).
            - 1D array of thresholds.
    """
    return build_curves_per_disease(y_true, y_pred, gene_number, metrics.roc_curve)


def pr_per_disease(
    y_true: List[np.ndarray], y_pred: List[np.ndarray], gene_number: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the PR curve averaged over diseases.

    Args:
        y_true: List of 1D arrays of 0/1 labels.
        y_pred: List of 1D arrays of scores (same length as each y_true[i]).
        gene_number: Number of genes.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - 2D array of interpolated mean FPR/TPR values across diseases
                (shape: [2, n_thresholds]).
            - 1D array of thresholds.
    """
    return build_curves_per_disease(
        y_true, y_pred, gene_number, metrics.precision_recall_curve
    )


def build_curves_per_disease(
    y_true: List[np.ndarray],
    y_pred: List[np.ndarray],
    gene_number: int,
    func: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute interpolated metric curves (ROC or PR) average over diseases.

    Args:
        y_true (List[np.ndarray]):
            List of binary label arrays (0/1), one per disease.
        y_pred (List[np.ndarray]):
            List of prediction score arrays, aligned with `y_true`.
        gene_number (int):
            Total number of genes, used to skip degenerate cases.
        func (Callable):
            Metric function (e.g., `roc_curve`, `precision_recall_curve`)
            returning (first_metric, second_metric, thresholds).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - 2D array of interpolated metric values
                (shape: [2, n_thresholds]).
            - 1D array of thresholds.
    """
    scores = []
    thr = np.linspace(0, 1, 1000)
    for labels, scores_pred in zip(y_true, y_pred):
        n_pos = int(labels.sum())
        if n_pos in (0, gene_number):
            continue

        first, second, _ = func(labels, scores_pred)
        scores.append((first, second))

    final = interpolate(scores, thr)
    return final, thr


def interpolate(
    scores: List[Tuple[np.ndarray, np.ndarray]], grid: List[float]
) -> np.ndarray:
    """
    Linearly interpolate metric values (e.g., TPR/FPR or Precision/Recall)
        across a common grid.

    Args:
        scores (List[Tuple[np.ndarray, np.ndarray]]):
            Per-disease tuples of (first_metric, second_metric).
        grid (List[float]): Interpolation grid.

    Returns:
        np.ndarray:
            2D array of shape [2, grid_bins] containing
                interpolated metrics.
    """
    final = []
    for first, second in scores:
        second_interp = np.interp(grid, first, second)
        final.append([grid, second_interp])
    return np.array(final).mean(0)
