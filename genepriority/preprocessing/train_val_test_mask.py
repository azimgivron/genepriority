# pylint: disable=R0903,R0801,R0913
"""
TrainValTestMasks module
========================

This module defines the `TrainValTestMasks` class, which encapsulates the creation
and management of training, validation, and testing masks for dataset splitting.
The masks are generated from a given non-zero indicator sparse matrix, by first
separating a validation set and then splitting the remaining entries using k-fold
cross validation. This allows reproducible random splits using sparse matrices.
"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import KFold, train_test_split


class TrainValTestMasks:
    """
    Encapsulates the creation and management of training, validation, and testing masks for
    dataset splitting.

    The masks are constructed from a provided non-zero (nnz) sparse matrix.
    First, a portion of the non-zero entries is selected as the validation set. The remaining
    entries are then divided into training and testing sets using k-fold cross validation.

    Attributes:
        training_masks (List[sp.csr_matrix]): A list of sparse matrices, each representing
            a training mask for one fold of the k-fold split.
        validation_mask (sp.csr_matrix): A sparse matrix representing the validation mask
            used for error estimation during training, which is fixed across all folds.
        validation_finetuning_mask (sp.csr_matrix): A sparse matrix representing the validation
            mask used for fine tuning, which is fixed across all folds.
        testing_masks (List[sp.csr_matrix]): A list of sparse matrices, each representing
            a testing mask corresponding to one fold of the k-fold split.
        seed (int): The random seed used to ensure reproducibility of the dataset splits.
    """

    def __init__(
        self,
        seed: int,
        nnz_mask: sp.csr_matrix,
        validation_size: float,
        num_folds: int,
    ):
        """
        Initializes the TrainValTestMasks object and creates the masks for training,
        validation, and testing.

        Args:
            seed (int): The random seed to ensure reproducibility in dataset splits.
            nnz_mask (sp.csr_matrix): A sparse matrix representing the non-zero entries of the
                dataset. The non-zero entries are used as the basis for mask generation.
            validation_size (float): Fraction of non-zero entries to be held out as a validation
                set. Must be between 0 and 1.
            num_folds (int): The number of folds to use for k-fold cross validation, which
                determines the number of training/testing mask pairs.

        Raises:
            ValueError: If `validation_size` is not strictly between 0 and 1.
        """
        if not 0 < validation_size < 1:
            raise ValueError("validation_size must be between 0 and 1.")

        self.seed = seed

        # Extract the row, column, and nonzero values from the input sparse mask.
        row_indices, col_indices, values = sp.find(nnz_mask)

        # Split the nonzero entries into a validation set and the rest.
        validation_row_indices, train_test_row_indices = train_test_split(
            np.arange(len(row_indices)),
            train_size=validation_size,
            random_state=self.seed,
            shuffle=True,
        )
        validation_row_indices, validation_finetuning_row_indices = train_test_split(
            np.arange(len(validation_row_indices)),
            train_size=0.5,
            random_state=self.seed,
            shuffle=True,
        )
        self.validation_finetuning_mask = sp.csr_matrix(
            (
                values[validation_finetuning_row_indices],
                (
                    row_indices[validation_finetuning_row_indices],
                    col_indices[validation_finetuning_row_indices],
                ),
            ),
            shape=nnz_mask.shape,
        )
        self.validation_mask = sp.csr_matrix(
            (
                values[validation_row_indices],
                (
                    row_indices[validation_row_indices],
                    col_indices[validation_row_indices],
                ),
            ),
            shape=nnz_mask.shape,
        )
        self.training_masks = []
        self.testing_masks = []

        # Use KFold on the remaining indices to create training and testing masks.
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=self.seed)
        for train_fold_indices, _ in kfold.split(train_test_row_indices):
            self.training_masks.append(
                sp.csr_matrix(
                    (
                        values[train_fold_indices],
                        (
                            row_indices[train_fold_indices],
                            col_indices[train_fold_indices],
                        ),
                    ),
                    shape=nnz_mask.shape,
                )
            )
            union_mask = (
                (
                    self.training_masks[-1]
                    + self.validation_mask
                    + self.validation_finetuning_mask
                )
                .toarray()
                .astype(bool)
            )
            test_mask = np.ones(nnz_mask.shape, dtype=bool)
            test_mask[union_mask] = False
            self.testing_masks.append(test_mask)

    def __iter__(
        self,
    ) -> Iterator[Tuple[sp.csr_matrix, np.ndarray, sp.csr_matrix, sp.csr_matrix]]:
        """
        Provides an iterator over the generated masks for each k-fold split.

        Yields:
            Iterator[Tuple[sp.csr_matrix, np.ndarray, sp.csr_matrix, sp.csr_matrix]]:
                A tuple containing:
                    - A training mask (sp.csr_matrix) for the current fold.
                    - A testing mask (np.ndarray) for the current fold.
                    - The validation mask (sp.csr_matrix), for error estimation
                    during training, which is identical all folds.
                    - The validation mask (sp.csr_matrix), for finetuning,
                    which is identical for all folds.
        """
        # Create a list of validation masks that matches the number of k-folds.
        validation_masks = [
            self.validation_mask for _ in range(len(self.training_masks))
        ]
        validation_finetuning_mask = [
            self.validation_finetuning_mask for _ in range(len(self.training_masks))
        ]
        return iter(
            zip(
                self.training_masks,
                self.testing_masks,
                validation_masks,
                validation_finetuning_mask,
            )
        )

    def __len__(self) -> int:
        """The iterator length.

        Returns:
            int: The length.
        """
        return len(self.training_masks)
