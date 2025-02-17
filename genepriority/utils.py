"""
Utils module
=============

This module defines the utility functions.
"""
import pickle
from typing import Any

import scipy.sparse as sp


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
    matrix_tmp.data[matrix_tmp.data == 0] = -1
    result = matrix_tmp.multiply(mask)
    result.data[result.data == -1] = 0
    return result
