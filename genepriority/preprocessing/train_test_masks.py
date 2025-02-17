# pylint: disable=R0903,R0913,R0801
"""
TrainTestMasks module
=======================

This module defines the `TrainTestMasks` class, which encapsulates the creation
of training and testing masks for dataset splitting. It provides functionality
for both random splitting and K-fold cross-validation using sparse matrices.
"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import KFold, train_test_split


class TrainTestMasks:
    """
    Encapsulates the creation and management of training and testing masks for
    dataset splitting.

    Attributes:
        training_masks (List[sp.csr_matrix]): A list of sparse matrices representing
            training masks.
        testing_masks (List[sp.csr_matrix]): A list of sparse matrices representing
            testing masks.
        seed (int): The random seed used for reproducibility.
    """

    def __init__(self, seed: int):
        """
        Initializes the TrainTestMasks object with a specified random seed.

        Args:
            seed (int): The random seed to ensure reproducibility in dataset splits.
        """
        self.training_masks = []
        self.testing_masks = []
        self.seed = seed

    def split(
        self,
        mask: sp.csr_matrix,
        train_size: float,
        num_splits: int,
        validation_size: float = None,
    ):
        """
        Performs random splitting of a given mask into training and testing sets.

        Args:
            mask (sp.csr_matrix): The sparse matrix mask to be split.
            train_size (float): Proportion of the dataset to include in the training set.
            num_splits (int): The number of random splits to generate.
            validation_size (float): Unused, set for API uniformity.
        """
        # pylint: disable=W0613
        row_indices, col_indices, values = sp.find(mask)
        indices = np.arange(len(row_indices))
        self.append_train_test_splits(
            indices,
            num_splits,
            row_indices,
            col_indices,
            values,
            train_size,
            mask.shape,
        )

    def append_train_test_splits(
        self,
        indices: np.ndarray,
        num_splits: int,
        row_indices: np.ndarray,
        col_indices: np.ndarray,
        values: np.ndarray,
        train_size: float,
        shape: Tuple[int, int],
    ):
        """
        Generate and store multiple train-test splits for sparse matrix data.

        This method creates `num_splits` train-test splits using random sampling,
        with a specified training size. The splits are stored as sparse matrices
        in CSR format for efficient storage and computation.

        Args:
            indices (np.ndarray): Indices for splitting the data.
            num_splits (int): Number of train-test splits to generate.
            row_indices (np.ndarray): Row indices of non-zero matrix elements.
            col_indices (np.ndarray): Column indices of non-zero matrix elements.
            values (np.ndarray): Values of the non-zero matrix elements.
            train_size (float): Proportion of data in the training set (0 < train_size < 1).
            shape (Tuple[int, int]): Shape of the sparse matrices.
        """
        if not 0 < train_size < 1:
            raise ValueError("train_size must be between 0 and 1.")

        for i in range(num_splits):
            train_row_indices, test_row_indices = train_test_split(
                indices,
                train_size=train_size,
                random_state=self.seed + i,
                shuffle=True,
            )
            self.append_train_test_masks(
                values,
                row_indices,
                col_indices,
                train_row_indices,
                test_row_indices,
                shape,
            )

    def fold(self, mask: sp.csr_matrix, num_folds: int):
        """
        Performs K-fold cross-validation splitting of a given mask into training and testing sets.

        Args:
            mask (sp.csr_matrix): The sparse matrix mask to be split.
            num_folds (int): The number of folds to create for cross-validation.
        """
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=self.seed)
        row_indices, col_indices, values = sp.find(mask)
        indices = np.arange(len(row_indices))
        for train_row_indices, test_row_indices in kfold.split(indices):
            self.append_train_test_masks(
                values,
                row_indices,
                col_indices,
                train_row_indices,
                test_row_indices,
                mask.shape,
            )

    def append_train_test_masks(
        self,
        values: np.ndarray,
        row_indices: np.ndarray,
        col_indices: np.ndarray,
        train_row_indices: np.ndarray,
        test_row_indices: np.ndarray,
        shape: Tuple[int, int],
    ):
        """
        Append training and testing masks to their respective lists.

        Args:
            values (np.ndarray): Non-zero values of the matrix.
            row_indices (np.ndarray): Row indices of non-zero elements.
            col_indices (np.ndarray): Column indices of non-zero elements.
            train_row_indices (np.ndarray): Row indices for the training set.
            test_row_indices (np.ndarray): Row indices for the testing set.
            shape (Tuple[int, int]): Dimensions of the sparse matrix.
        """
        train_mask = sp.csr_matrix(
            (
                values[train_row_indices],
                (row_indices[train_row_indices], col_indices[train_row_indices]),
            ),
            shape=shape,
        )
        test_mask = sp.csr_matrix(
            (
                values[test_row_indices],
                (row_indices[test_row_indices], col_indices[test_row_indices]),
            ),
            shape=shape,
        )
        self.training_masks.append(train_mask)
        self.testing_masks.append(test_mask)

    def __iter__(self) -> Iterator[Tuple[sp.csr_matrix, sp.csr_matrix]]:
        """
        Provides an iterator over the training and testing masks.

        Yields:
            Iterator[Tuple[sp.csr_matrix, sp.csr_matrix]]: An iterator yielding tuples,
                where each tuple contains one training mask and one testing mask.
        """
        return iter(zip(self.training_masks, self.testing_masks))
