"""
Utils module
=============

This module defines the utility functions.
"""
import pickle
from typing import Any

import numpy as np
import scipy.sparse as sp
from sklearn import metrics


def calculate_auc_bedroc(
    labels: np.ndarray, predictions: np.ndarray, alpha: float = 160.9
) -> tuple[float, float, float]:
    """
    Computes the Area Under the Curve (AUC-ROC), Average Precision (AP),
    and the Boltzmann-Enhanced Discrimination of ROC (BEDROC) score for the test data.

    - AUC-ROC measures the model's ability to rank positive instances
    higher than negative ones, capturing its overall classification performance.

    - Average Precision (AP) represents the weighted mean of precisions
    at different thresholds, considering both precision and recall.

    - BEDROC extends AUC-ROC by emphasizing early retrieval, making it
    particularly useful for ranking problems where top predictions matter most.

    Formula for AUC-ROC:
        AUC = ∫ TPR(FPR) d(FPR)

    Formula for AP:
        AP = sum( (R_n - R_{n-1}) * P_n )

    Formula for BEDROC:
        BEDROC = (1 / N) * sum(exp(-alpha * rank(y_i))) * normalization_factor

    Where:
        - TPR: True Positive Rate.
        - FPR: False Positive Rate.
        - N: Number of positive samples.
        - rank(y_i): Rank of the positive sample in the sorted predictions.
        - alpha: Controls the early recognition emphasis.
        - R_n: Recall at threshold n.
        - P_n: Precision at threshold n.

    Args:
        labels (np.ndarray): The labels.
        predictions (np.ndarray): The model predictions.
        alpha (float): The alpha value for the bedroc metric.
            Default to 160.9.

    Returns:
        tuple[float, float, float]: A tuple containing:
            - AUC-ROC score: Measures overall ranking performance.
            - Average Precision (AP) score: Reflects precision-recall trade-off.
            - BEDROC score: Emphasizes early recognition quality.
    """
    # pylint: disable=C0415
    from genepriority.evaluation.metrics import bedroc_score

    # Compute AUC-ROC
    auc = metrics.roc_auc_score(labels, predictions)

    # Compute Average Precision (AP)
    avg_precision = metrics.average_precision_score(labels, predictions)

    # Compute BEDROC
    bedroc = bedroc_score(
        y_true=labels,
        y_pred=predictions,
        decreasing=True,
        alpha=alpha,
    )
    return auc, avg_precision, bedroc


def serialize(object_instance: Any, output_path: str):
    """
    Save object to a file in binary format using pickle.

    Args:
        object_instance (Any): An object to serialize.
        output_path (str): The file path where the object should be saved.
    """
    with open(output_path, "wb") as handler:
        pickle.dump(object_instance, handler)


def mask_sparse_containing_0s(
    matrix: sp.csr_matrix, mask: sp.csr_matrix
) -> sp.csr_matrix:
    """
    Masks the values of a given sparse matrix based on a mask matrix.

    This function modifies the input sparse matrix such that:
    - Non-zero values in the input `matrix` are preserved where the `mask` has non-zero entries.
    - All other values are set to zero.

    Args:
        matrix (sp.csr_matrix): A sparse matrix (CSR format) whose values need to be masked.
        mask (sp.csr_matrix): A sparse matrix (CSR format) with non-zero entries indicating where
        values in `matrix` should be retained.

    Returns:
        sp.csr_matrix: A sparse matrix (CSR format) resulting from applying the `mask` to `matrix`.
    """
    matrix_tmp = matrix.copy()
    # Replace explicit 0s with a temporary marker.
    matrix_tmp.data[matrix_tmp.data == 0] = -1
    result = matrix_tmp.multiply(mask)
    # Restore explicit 0s
    result.data[result.data == -1] = 0
    return result
