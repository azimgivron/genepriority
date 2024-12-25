# pylint: disable=R0903
"""
TrainTestMasks module
=======================

This module defines the `TrainTestMasks` class, which encapsulates the creation
of training and testing masks for dataset splitting. It provides functionality
for both random splitting and K-fold cross-validation using sparse matrices.
"""

from __future__ import annotations

from typing import Iterator, Tuple

import scipy.sparse as sp
from sklearn.model_selection import KFold, train_test_split

from NEGradient_GenePriority.utils import filter_from_indices


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
    ):
        """
        Performs random splitting of a given mask into training and testing sets.

        Args:
            mask (sp.csr_matrix): The sparse matrix mask to be split.
            train_size (float): Proportion of the dataset to include in the training set.
            num_splits (int): The number of random splits to generate.
        """
        for i in range(num_splits):
            row_indices, col_indices, values = sp.find(mask)
            train_row_indices, test_row_indices = train_test_split(
                row_indices,
                train_size=train_size,
                random_state=self.seed + i,
                shuffle=True,
            )
            train_mask = sp.csr_matrix(
                (
                    values[train_row_indices],
                    (row_indices[train_row_indices], col_indices[train_row_indices]),
                ),
                shape=mask.shape,
            )
            test_mask = sp.csr_matrix(
                (
                    values[test_row_indices],
                    (row_indices[test_row_indices], col_indices[test_row_indices]),
                ),
                shape=mask.shape,
            )
            self.training_masks.append(train_mask)
            self.testing_masks.append(test_mask)

    def fold(self, mask: sp.csr_matrix, num_folds: int):
        """
        Performs K-fold cross-validation splitting of a given mask into training and testing sets.

        Args:
            mask (sp.csr_matrix): The sparse matrix mask to be split.
            num_folds (int): The number of folds to create for cross-validation.
        """
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=self.seed)
        rows_idx = sp.find(mask)[0]
        for train_idx, test_idx in kfold.split(rows_idx):
            train_mask = filter_from_indices(mask, train_idx)
            test_mask = filter_from_indices(mask, test_idx)
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
