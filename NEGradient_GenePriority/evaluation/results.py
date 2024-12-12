"""
Results module
==============

Provides a data structure for storing and processing simulation results
of prediction tasks, including both ground truth values and model-predicted values.
"""

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import scipy.sparse as sp


@dataclass
class Results:
    """
    Encapsulates the results of a prediction task.

    This class provides a structure to store and process prediction results,
    including ground truth values and corresponding predicted values from a model.
    It supports validation of input data and facilitates iteration over
    the results in a convenient stacked format.

    Attributes:
        y_true (sp.csr_matrix): Ground truth sparse matrix, where each entry
            represents the true association (e.g., between a disease and a gene).
        y_pred (np.ndarray): Predicted values as a dense array, where each entry
            represents the likelihood of an association predicted by the model.
    """

    y_true: sp.csr_matrix
    y_pred: np.ndarray

    def __post_init__(self):
        """
        Validates the input data types and shapes after initialization.

        Ensures that `y_true` is a sparse CSR matrix and `y_pred` is a dense
        numpy array. Also checks that the shapes of `y_true` and `y_pred` match.

        Raises:
            TypeError: If `y_true` is not of type `sp.csr_matrix` or
                       `y_pred` is not of type `np.ndarray`.
            ValueError: If the shapes of `y_true` and `y_pred` do not match.
        """
        if not isinstance(self.y_true, sp.csr_matrix):
            raise TypeError(
                "`y_true` must be of type sp.csr_matrix, but "
                f"got type {type(self.y_true)} instead."
            )
        if not isinstance(self.y_pred, np.ndarray):
            raise TypeError(
                "`y_pred` must be of type np.ndarray, but got "
                f"type {type(self.y_pred)} instead."
            )
        if self.y_true.shape != self.y_pred.shape:
            raise ValueError(
                f"Shape mismatch: `y_true` has shape {self.y_true.shape}, "
                f"but `y_pred` has shape {self.y_pred.shape}. Shapes must match."
            )

    def __iter__(self) -> Iterator[np.ndarray]:
        """
        Returns an iterator over a stacked array of predictions and ground truth values.

        The stacked array has shape `(disease, 2, gene)`, where the second dimension contains:
        - The predicted value (`y_pred`) in the first position.
        - The ground truth value (`y_true`) in the second position.

        This representation is useful for tasks requiring simultaneous access
        to both the predictions and the ground truth.

        Returns:
            Iterator[np.ndarray]: An iterator over the stacked array combining
            ground truth values and predictions.
        """
        all_1s = self.y_true.T.toarray() == 1  # shape = (disease, gene)
        disease_mask = all_1s.any(axis=1)  # keep where there is a 1
        pred_truth_mat = np.stack((self.y_true.T.toarray(), self.y_pred.T), axis=2)
        pred_truth_mat = pred_truth_mat[disease_mask]
        pred_truth_mat = np.swapaxes(pred_truth_mat, 1, 2)
        return iter(pred_truth_mat)
