# pylint: disable=R0903
"""
TrainValTestMasks module
=======================

This module defines the `TrainValTestMasks` class, which encapsulates the creation
of training, validation, and testing masks for dataset splitting. It provides functionality
for random splitting using sparse matrices.
"""

from __future__ import annotations
from typing import Iterator, Tuple

import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from NEGradient_GenePriority.preprocessing.train_test_masks import TrainTestMasks

class TrainValTestMasks(TrainTestMasks):
    """
    Encapsulates the creation and management of training, validation, and testing masks for
    dataset splitting.

    Attributes:
        training_masks (List[sp.csr_matrix]): A list of sparse matrices representing
            training masks.
        validation_masks (List[sp.csr_matrix]): A list of sparse matrices representing
            validation masks.
        testing_masks (List[sp.csr_matrix]): A list of sparse matrices representing
            testing masks.
        seed (int): The random seed used for reproducibility.
        iter_over_validation (bool): A flag indicating whether to iterate over validation
            masks (True) or testing masks (False) during iteration.
    """

    def __init__(self, seed: int, iter_over_validation: bool = False):
        """
        Initializes the TrainValTestMasks object with a specified random seed.

        Args:
            seed (int): The random seed to ensure reproducibility in dataset splits.
            iter_over_validation (bool, optional): Determines whether iteration yields validation
                masks (if True) or testing masks (if False). Defaults to False.
        """
        super().__init__(seed)
        self.validation_masks = []
        self.iter_over_validation = iter_over_validation

    def split(
        self,
        mask: sp.csr_matrix,
        train_size: float,
        validation_size: float,
        num_splits: int,
    ):
        """
        Performs random splitting of a given mask into training, validation, and testing sets.

        Args:
            mask (sp.csr_matrix): The sparse matrix mask to be split.
            train_size (float): Proportion of the dataset to include in the training set.
            validation_size (float): Proportion of the training set to further split into
                validation data.
            num_splits (int): The number of random splits to generate.
        """
        if not (0 < train_size < 1) or not (0 < validation_size < 1):
            raise ValueError("train_size and validation_size must be between 0 and 1.")

        for i in range(num_splits):
            row_indices, col_indices, values = sp.find(mask)
            train_row_indices, test_row_indices = train_test_split(
                row_indices,
                train_size=train_size,
                random_state=self.seed + i,
                shuffle=True,
            )
            train_row_indices, validation_row_indices = train_test_split(
                train_row_indices,
                train_size=1 - validation_size,
                random_state=self.seed + i,
                shuffle=True,
            )
            validation_mask = sp.csr_matrix(
                (
                    values[validation_row_indices],
                    (
                        row_indices[validation_row_indices],
                        col_indices[validation_row_indices],
                    ),
                ),
                shape=mask.shape,
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
            self.validation_masks.append(validation_mask)
            self.testing_masks.append(test_mask)

    def __iter__(self) -> Iterator[Tuple[sp.csr_matrix, sp.csr_matrix]]:
        """
        Provides an iterator over the training and validation/testing masks.

        Yields:
            Iterator[Tuple[sp.csr_matrix, sp.csr_matrix]]: An iterator yielding tuples,
                where each tuple contains one training mask and one validation/testing mask.
        """
        if self.iter_over_validation:
            iterator = iter(zip(self.training_masks, self.validation_masks))
        else:
            iterator = iter(zip(self.training_masks, self.testing_masks))
        return iterator
