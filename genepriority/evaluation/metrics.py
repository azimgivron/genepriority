"""
Metrics module
==============

Provides evaluation metrics for gene prioritization models. Includes:
  - BEDROC: emphasizes early retrieval of true positives.
  - AUC: classical ranking metric adapted to partial gene sets.

Both functions accept only the evaluated subset of genes but normalize
scores against the total gene universe.
"""
from typing import List, Tuple

import numpy as np
from sklearn import metrics


def bedroc_score(
    y_true: np.ndarray, y_pred: np.ndarray, decreasing: bool = True, alpha: float = 20.0
):
    """BEDROC metric implemented according to Truchon and Bayley.

    The Boltzmann Enhanced Descrimination of the Receiver Operator
    Characteristic (BEDROC) score is a modification of the Receiver Operator
    Characteristic (ROC) score that allows for a factor of *early recognition*.

    References:
        The original paper by Truchon et al. is located at `10.1021/ci600426e
        <http://dx.doi.org/10.1021/ci600426e>`_.

    Args:
        y_true (np.ndarray): Binary class labels. 1 for positive class, 0 otherwise.
        y_pred (np.ndarray): Prediction values.
        decreasing (bool): True if high values of ``y_pred`` correlates to positive class.
        alpha (float): Early recognition parameter.

    Returns:
        float:
            Value in interval [0, 1] indicating degree to which the predictive
            technique employed detects (early) the positive class.
    """

    assert len(y_true) == len(
        y_pred
    ), "The number of scores must be equal to the number of labels"

    big_n = len(y_true)
    n = sum(y_true == 1)

    if decreasing:
        order = np.argsort(-y_pred)
    else:
        order = np.argsort(y_pred)

    m_rank = (y_true[order] == 1).nonzero()[0]
    s = np.sum(np.exp(-alpha * m_rank / big_n))
    r_a = n / big_n
    rand_sum = r_a * (1 - np.exp(-alpha)) / (np.exp(alpha / big_n) - 1)
    fac = (
        r_a
        * np.sinh(alpha / 2)
        / (np.cosh(alpha / 2) - np.cosh(alpha / 2 - alpha * r_a))
    )

    cte = 1 / (1 - np.exp(alpha * (1 - r_a)))
    return s * fac / rand_sum + cte


def bedroc_scores(
    y_true: List[np.ndarray], y_pred: List[np.ndarray], gene_number: int, alpha: float
) -> float:
    """Compute the BEDROC score for each diseases.

    BEDROC (Boltzmann-Enhanced Discrimination of the ROC) places exponential
    emphasis on ranking true positives early in the prediction list.

    Args:
        y_true (List[np.ndarray]):
            Binary labels for the evaluated genes (1 = positive, 0 = negative).
        y_pred (List[np.ndarray]):
            Model scores for those genes; higher means more likely positive.
        gene_number (int):
            Total number of genes in the universe. Must be ≥ n_samples.
        alpha (float):
            Early-recognition weight parameter (> 0).

    Returns:
        float: BEDROC in [0,1] (0=random, 1=perfect).
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0; got {alpha}.")
    scores = []
    mask = []
    for labels, scores_pred in zip(y_true, y_pred):
        # number of positives for this disease
        n_pos = int(labels.sum())
        mask.append(n_pos > 0)

        if n_pos == 0 or n_pos == gene_number:
            # undefined if no positives or no negatives
            scores.append(np.nan)
            continue
        bedroc = bedroc_score(labels, scores_pred, alpha=alpha)
        scores.append(bedroc)
    return scores, mask


def auc_scores(
    y_true: List[np.ndarray], y_pred: List[np.ndarray], gene_number: int
) -> Tuple[List[float], List[bool]]:
    """Compute the true AUC score for each disease.

    Args:
        y_true: List of 1D arrays of 0/1 labels.
        y_pred: List of 1D arrays of scores (same length as each y_true[i]).
        gene_number: Total size of the gene universe (N).

    Returns:
        scores: List of AUC floats (nan if no positives or no negatives).
        mask:   List of booleans indicating which diseases had ≥1 positive.
    """
    scores = []
    mask = []

    for labels, scores_pred in zip(y_true, y_pred):
        # number of positives for this disease
        n_pos = int(labels.sum())
        mask.append(n_pos > 0)

        if n_pos == 0 or n_pos == gene_number:
            # undefined if no positives or no negatives
            scores.append(np.nan)
            continue

        auc = metrics.roc_auc_score(labels, scores_pred)
        scores.append(auc)
    return scores, mask
