# pylint: disable=R0903,R0801
"""
TrainValTestMasks module
=======================

This module defines the `TrainValTestMasks` class, which encapsulates the creation
of training, validation, and testing masks for dataset splitting. It provides functionality
for random splitting using sparse matrices.
"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import scipy.sparse as sp
from NEGradient_GenePriority.preprocessing.train_test_masks import TrainTestMasks
from sklearn.model_selection import train_test_split


class TrainValTestMasks(TrainTestMasks):
    """
    Encapsulates the creation and management of training, validation, and testing masks for
    dataset splitting.

    Attributes:
        training_masks (List[sp.csr_matrix]): A list of sparse matrices representing
            training masks.
        validation_mask (sp.csr_matrix): A sparse matrix representing the
            validation mask.
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
        self.validation_mask = None
        self.iter_over_validation = iter_over_validation

    def split(
        self,
        mask: sp.csr_matrix,
        train_size: float,
        num_splits: int,
        validation_size: float = None,
    ):
        """
        Performs random splitting of a given mask into training, validation, and testing sets.

        Args:
            mask (sp.csr_matrix): The sparse matrix mask to be split.
            train_size (float): Proportion of the dataset to include in the training set.
            num_splits (int): The number of random splits to generate.
            validation_size (float): Proportion of the training set to further split into
                validation data.
        """
        if not 0 < validation_size < 1:
            raise ValueError("validation_size must be between 0 and 1.")

        row_indices, col_indices, values = sp.find(mask)
        validation_row_indices, train_test_row_indices = train_test_split(
            np.arange(len(row_indices)),
            train_size=validation_size,
            random_state=self.seed,
            shuffle=True,
        )
        self.validation_mask = sp.csr_matrix(
            (
                values[validation_row_indices],
                (
                    row_indices[validation_row_indices],
                    col_indices[validation_row_indices],
                ),
            ),
            shape=mask.shape,
        )
        self.append_train_test_splits(
            train_test_row_indices,
            num_splits,
            row_indices,
            col_indices,
            values,
            train_size,
            mask.shape,
        )

    def __iter__(self) -> Iterator[Tuple[sp.csr_matrix, sp.csr_matrix]]:
        """
        Provides an iterator over the training and validation/testing masks.

        Yields:
            Iterator[Tuple[sp.csr_matrix, sp.csr_matrix]]: An iterator yielding tuples,
                where each tuple contains one training mask and one validation/testing mask.
        """
        if self.iter_over_validation:
            validation_masks = [
                self.validation_mask for _ in range(len(self.training_masks))
            ]
            iterator = iter(zip(self.training_masks, validation_masks))
        else:
            iterator = iter(zip(self.training_masks, self.testing_masks))
        return iterator
