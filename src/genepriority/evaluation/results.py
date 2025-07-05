"""
Results module
==============
Encapsulates the results of a prediction task.
"""
from typing import List

import numpy as np


class Results:
    """
    Encapsulates the results of a prediction task.

    Attributes:
        _y_true (np.ndarray): Ground truth association matrix.
        _y_pred (np.ndarray): Matrix of gene disease association predicted by the model.
        mask (np.ndarray): Mask of entries for test.
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        mask: np.ndarray,
    ):
        """
        Initializes the Results object.

        Args:
            y_true (np.ndarray): Ground truth association matrix.
            y_pred (np.ndarray): Matrix of gene disease association predicted by the model.
            mask (np.ndarray): Mask of entries for test.
        """
        self.mask = mask
        self._y_true = y_true
        self._y_pred = y_pred

    @property
    def y_true(self) -> List[np.ndarray]:
        """
        Retrieve the ground‐truth association values for each disease.

        Returns:
            List[np.ndarray]:
                A list of length D, where D is the number of diseases in the mask.
                Each entry is a 1-D array containing the ground‐truth
                values for that disease at all samples where mask is True.
        """
        return [
            self._y_true[:, i][self.mask[:, i]] for i in range(self._y_true.shape[1])
        ]

    @property
    def y_pred(self) -> List[np.ndarray]:
        """
        Retrieve the predicted association values for each disease.

        Returns:
            List[np.ndarray]:
                A list of length D, where D is the number of diseases in the mask.
                Each entry is a 1-D array containing the predicted
                values for that disease at all samples where mask is True.
        """
        return [
            self._y_pred[:, i][self.mask[:, i]] for i in range(self._y_pred.shape[1])
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
