"""
Metrics module
==============

Provides evaluation metrics for gene prioritization models. Includes:
  - BEDROC: emphasizes early retrieval of true positives.
  - AUC: classical ranking metric adapted to partial gene sets.

Both functions accept only the evaluated subset of genes but normalize
scores against the total gene universe.
"""
from typing import List

import numpy as np


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
    n_diseases = len(y_true)
    scores = []
    mask = []
    for i in range(n_diseases):
        if (y_true[i] == 1).sum() > 0:
            mask.append(True)
            order = np.argsort(-y_pred[i])
            pos_ranks = np.nonzero(y_true[i][order] == 1)[0] + 1
            weighted = np.mean(np.exp(-alpha * (pos_ranks / gene_number)))
            expected = (
                (alpha / gene_number)
                * (1 - np.exp(-alpha))
                / (np.exp(alpha / gene_number) - 1)
            )
            shift = 1.0 / (1 - np.exp(alpha))
            scores.append(weighted / expected + shift)
        else:
            mask.append(False)
            scores.append(np.nan)
    return scores, mask


def auc_scores(
    y_true: List[np.ndarray], y_pred: List[np.ndarray], gene_number: int
) -> float:
    """Compute the AUC‐approximation score for each disease.

    Approximates the area under the accumulation curve (AUAC), which
    converges to AUC when positives are sparse, then normalizes by the
    full gene universe size.

    Args:
        y_true (List[np.ndarray]):
            Binary labels (1 = positive, 0 = negative).
        y_pred (List[np.ndarray]):
            Predicted scores for those genes.
        gene_number (int):
            Total number of genes in the universe. Must be ≥ n_samples.

    Returns:
        float: AUC‐approx score in [0,1] (higher = better).
    """
    n_diseases = len(y_true)
    scores = []
    mask = []
    for i in range(n_diseases):
        if (y_true[i] == 1).sum() > 0:
            mask.append(True)
            order = np.argsort(-y_pred[i])
            pos_ranks = np.nonzero(y_true[i][order] == 1)[0] + 1
            scores.append(1-np.mean(pos_ranks / gene_number))
        else:
            mask.append(False)
            scores.append(np.nan)
    return scores, mask
