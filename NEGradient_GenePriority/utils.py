"""
Utils module
=============

This module defines the utility functions.
"""
import pickle
from typing import Dict

import numpy as np
import scipy.sparse as sp

from NEGradient_GenePriority.evaluation.evaluation import Evaluation


def save_evaluations(results: Dict[str, Evaluation], output_path: str):
    """
    Save evaluation results to a file in binary format using pickle.

    Args:
        results (Dict[str, Evaluation]): A dictionary where keys are descriptive strings
            (e.g., latent dimensions or other identifiers) and values are `Evaluation` objects
            containing evaluation metrics and results.
        output_path (str): The file path where the results should be saved.
    """
    with open(output_path, "wb") as handler:
        pickle.dump(results, handler)


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
    matrix_tmp.data[matrix_tmp.data == 0] = -1
    result = matrix_tmp.multiply(mask)
    result.data[result.data == -1] = 0
    return result


def filter_from_indices(matrix: sp.csr_matrix, row_idx: np.ndarray) -> sp.csr_matrix:
    """
    Filters a sparse matrix by zeroing out specified rows.

    This function modifies the given sparse matrix by setting all the elements
    in the specified rows to zero. The input matrix is first converted to
    LIL (List of Lists) format for efficient row-wise manipulation, and
    the output is returned in CSR (Compressed Sparse Row) format.

    Args:
        matrix (sp.csr_matrix): The input sparse matrix in CSR format to be filtered.
        row_idx (np.ndarray): An array of row indices to be zeroed out. Indices should
            be valid within the range of the matrix's rows.

    Returns:
        sp.csr_matrix: A sparse matrix in CSR format where the specified rows
            have been zeroed out.
    """
    matrix_lil = matrix.copy().tolil()
    for row in row_idx:
        matrix_lil[row, :] = 0
    return matrix_lil.tocsr()
