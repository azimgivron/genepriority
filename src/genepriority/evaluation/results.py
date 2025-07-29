"""
Results module
==============
Encapsulates the results of a prediction task.
"""

from typing import List

import numpy as np


class Results:
    """
    Encapsulates ground-truth and predicted associations for a gene-disease task,
    applying a binary mask and optional threshold filtering by disease prevalence.

    Attributes:
        _y_true (np.ndarray): Full ground-truth matrix of shape (N_samples, N_diseases).
        _y_pred (np.ndarray): Full predicted score matrix of same shape.
        mask (np.ndarray): Boolean mask of shape (N_samples, N_diseases) indicating test entries.
        threshold (int): Minimum number of positive associations required to include a disease.
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        mask: np.ndarray,
        threshold: int = 0,
    ):
        """
        Initialize Results with data matrices and filtering criteria.

        Args:
            y_true (np.ndarray): Ground-truth binary or continuous association matrix.
            y_pred (np.ndarray): Model-predicted association scores.
            mask (np.ndarray): Boolean mask selecting valid test entries.
            threshold (int, optional): Minimum total positives per disease to retain. Defaults to 0.
        """
        self._y_true = y_true
        self._y_pred = y_pred
        self.mask = mask
        self.threshold = threshold

    @property
    def y_true(self) -> List[np.ndarray]:
        """
        Extract masked true values for each disease.

        Returns:
            List[np.ndarray]: Length D list of 1D arrays where each array contains
            y_true[:, i] at positions where mask[:, i] is True.
        """
        return [
            self._y_true[:, i][self.mask[:, i]] for i in range(self._y_true.shape[1])
        ]

    @property
    def y_pred(self) -> List[np.ndarray]:
        """
        Extract masked predicted values for each disease.

        Returns:
            List[np.ndarray]: Length D list of 1D arrays where each array contains
            y_pred[:, i] at positions where mask[:, i] is True.
        """
        return [
            self._y_pred[:, i][self.mask[:, i]] for i in range(self._y_pred.shape[1])
        ]

    @property
    def y_true_filtered(self) -> List[np.ndarray]:
        """
        Extract masked true values for diseases meeting the prevalence threshold.

        Returns:
            List[np.ndarray]: As in `y_true`, but only for diseases where
            total positives (sum of y_true) >= threshold.
        """
        valid = self._y_true.sum(axis=0) >= self.threshold
        return [
            self._y_true[:, i][self.mask[:, i]]
            for i in range(self._y_true.shape[1])
            if valid[i]
        ]

    @property
    def y_pred_filtered(self) -> List[np.ndarray]:
        """
        Extract masked predicted values for diseases meeting the prevalence threshold.

        Returns:
            List[np.ndarray]: As in `y_pred`, but only for diseases where
            total true positives >= threshold.
        """
        valid = self._y_true.sum(axis=0) >= self.threshold
        return [
            self._y_pred[:, i][self.mask[:, i]]
            for i in range(self._y_pred.shape[1])
            if valid[i]
        ]

    @property
    def gene_number(self) -> int:
        """
        Returns:
            int: The number of genes.
        """
        return self.mask.shape[0]

    @property
    def disease_number(self) -> int:
        """
        Returns:
            int: The number of diseases.
        """
        return self.mask.shape[1]
