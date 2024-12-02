"""Indices module"""
from __future__ import annotations

from typing import Set, Tuple

import numpy as np
import scipy as sp

from NEGradient_GenePriority.preprocessing.utils import from_indices


class Indices:
    """
    Encapsulates a set of indices for a dataset.

    This class is designed to manage a collection of row-column indices representing a subset
    of a dataset. It provides methods to retrieve the corresponding data from a given sparse
    matrix, as well as a set representation for efficient operations. Additionally, it supports
    merging multiple indices.

    Attributes:
        indices (np.ndarray): A 2D array of shape (n, 2), where each row represents
                              a (row, column) pair of indices.
    """

    def __init__(self, indices: np.ndarray) -> None:
        """
        Initializes the Indices object with the given array of indices.

        Args:
            indices (np.ndarray): A 2D array of shape (n, 2) containing row-column pairs.
        """
        self.indices = indices

    @property
    def indices_set(self) -> Set[Tuple[int, int]]:
        """
        Converts the indices into a set of (row, column) tuples.

        Returns:
            Set[Tuple[int, int]]: A set of (row, column) tuples for the indices.
        """
        return set(zip(self.indices[:, 0], self.indices[:, 1]))

    def get_data(self, dataset_matrix: sp.coo_matrix) -> sp.csr_matrix:
        """
        Retrieves the subset of the dataset corresponding to the stored indices.

        Args:
            dataset_matrix (sp.coo_matrix): The full dataset represented as a COO sparse matrix.

        Returns:
            sp.csr_matrix: A sparse matrix in CSR format containing only the elements
                           specified by the indices. The shape of the returned matrix
                           matches the shape of the original dataset.
        """
        return from_indices(dataset_matrix, self.indices_set).tocsr()

    def merge(self, indices: Indices) -> Indices:
        """
        Merges another Indices object into the current one.

        Args:
            indices (Indices): Another Indices object to merge.

        Returns:
            Indices: A new instance of Indices with merged indices.
        """
        return Indices(np.vstack((self.indices, indices.indices)))

    def mask(self, data: np.ndarray) -> np.ndarray:
        """Mask over the indices.

        Args:
            data (np.ndarray): The data to mask.

        Returns:
            np.ndarray: Masked data.
        """
        rows, cols = zip(*self.indices.tolist())
        return data[np.array(rows), np.array(cols)]
