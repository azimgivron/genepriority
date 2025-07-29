"""
Utils module
=============

This module defines the utility functions.
"""

import pickle
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import scipy as sp
import scipy.sparse as sp
import tensorflow as tf
from sklearn import metrics


def svd(matrix: np.ndarray, rank: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize low-rank factors from a truncated SVD of the matrix.

    Args:
        matrix (np.ndarray): Matrix of shape (n_rows, n_cols).
        rank (int): Target rank for the approximation.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - left_factor: shape (n_rows, rank)
            - right_factor: shape (rank, n_cols)
    """
    (
        left_singular_vectors,
        singular_values,
        right_singular_vectors_t,
    ) = sp.linalg.svd(matrix, full_matrices=False)

    left_vectors_truncated = left_singular_vectors[:, :rank]
    singular_values_truncated = singular_values[:rank]
    right_vectors_t_truncated = right_singular_vectors_t[:rank, :]

    left_factor = left_vectors_truncated * singular_values_truncated[np.newaxis, :]
    right_factor = right_vectors_t_truncated

    return left_factor, right_factor


def calculate_auroc_auprc(
    labels: np.ndarray, predictions: np.ndarray
) -> tuple[float, float, float]:
    """
    Computes the Area Under the Curve (AUC-ROC) and Average Precision (AP)
    score for the test data.

    - AUC-ROC measures the model's ability to rank positive instances
    higher than negative ones, capturing its overall classification performance.

    - Average Precision (AP) represents the weighted mean of precisions
    at different thresholds, considering both precision and recall.

    Formula for AUC-ROC:
        AUC = ∫ TPR(FPR) d(FPR)

    Formula for AP:
        AP = sum( (R_n - R_{n-1}) * P_n )

    Where:
        - TPR: True Positive Rate.
        - FPR: False Positive Rate.
        - N: Number of positive samples.
        - rank(y_i): Rank of the positive sample in the sorted predictions.
        - R_n: Recall at threshold n.
        - P_n: Precision at threshold n.

    Args:
        labels (np.ndarray): The labels.
        predictions (np.ndarray): The model predictions.

    Returns:
        tuple[float, float, float]: A tuple containing:
            - AUC-ROC score: Measures overall ranking performance.
            - Average Precision (AP) score: Reflects precision-recall trade-off.
    """
    # pylint: disable=C0415
    auc = metrics.roc_auc_score(labels, predictions)
    avg_precision = metrics.average_precision_score(labels, predictions)
    return auc, avg_precision


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


def create_tb_dir(run_log_dir: Path) -> "tf.SummaryWriter":
    """
    Create and return a TensorFlow SummaryWriter for the specified log directory.

    Args:
        run_log_dir (Path): The path to the directory where TensorBoard logs will be stored.

    Returns:
        tf.SummaryWriter: A TensorFlow SummaryWriter configured to write logs to the
            specified directory.
    """
    if run_log_dir.exists() and any(run_log_dir.iterdir()):
        for item in run_log_dir.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()  # Remove the current log
    run_log_dir.mkdir(parents=True, exist_ok=True)
    return tf.summary.create_file_writer(str(run_log_dir))
