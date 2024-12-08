"""
Utils module
=============

This module defines the utility functions.
"""
from typing import Set, Tuple

import scipy.sparse as sp


def from_indices(
    dataset_matrix: sp.coo_matrix, indices_set: Set[Tuple[int, int]]
) -> sp.coo_matrix:
    """
    Extracts a submatrix from the given sparse matrix based on specified row-column indices
    while keeping the same shape as the original matrix.

    Args:
        dataset_matrix (sp.coo_matrix): The input sparse matrix from which
            elements are to be extracted.
        indices_set (Set[Tuple[int, int]]): A set of (row, column) tuples specifying
            the elements to extract.

    Returns:
        sp.coo_matrix: A sparse matrix in COO format containing only the elements specified by
                       the indices_set. The submatrix will have the same shape as the original
                       matrix, but only the specified elements will be retained.
    """
    mask = [
        (row, col) in indices_set
        for row, col in zip(dataset_matrix.row, dataset_matrix.col)
    ]
    rows = dataset_matrix.row[mask]
    cols = dataset_matrix.col[mask]
    data = dataset_matrix.data[mask]
    return sp.coo_matrix((data, (rows, cols)), shape=dataset_matrix.shape)
