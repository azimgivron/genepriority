"""
Results module
==============

Contains a data structure for storing and processing simulation results.
"""

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import scipy.sparse as sp


@dataclass
class Results:
    """
    Prediction results data structure.

    This class encapsulates the results of a prediction task, including
    the ground truth values, predicted values, and additional filtering
    options for test data.

    Attributes:
        y_true (sp.csr_matrix): Ground truth matrix, where each entry
            represents a true association.
        y_pred (np.ndarray): Predicted values from the trained model,
            representing the likelihood of associations.
        mask_1s (np.ndarray): Binary mask indicating the positions of
            positive associations (1s) in the data.
        on_test_data_only (bool): Whether to filter the predictions to
            retain 1s from test data only.
    """

    y_true: sp.csr_matrix
    y_pred: np.ndarray
    mask_1s: np.ndarray
    on_test_data_only: bool = False

    def __post_init__(self):
        """
        Validates the input data types and shapes after initialization.
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
        if not isinstance(self.mask_1s, np.ndarray):
            raise TypeError(
                "`mask_1s` must be of type np.ndarray, but "
                f"got type {type(self.mask_1s)} instead."
            )
        if self.mask_1s.shape != self.y_true.shape:
            raise ValueError(
                f"Shape mismatch: `mask_1s` has shape {self.mask_1s.shape}, "
                f"but `y_true` has shape {self.y_true.shape}. Shapes must match."
            )

    def filter(self, pred_truth_mat: np.ndarray) -> np.ndarray:
        """
        Filters the prediction matrix to retain only the relevant diseases.

        This method identifies columns (diseases) that contain unselected positive
        associations (1s) based on the ground truth and provided masks, and filters
        the prediction matrix accordingly.

        Args:
            pred_truth_mat (np.ndarray): A 3D array of predictions and ground
                truth values.

        Returns:
            np.ndarray: Filtered prediction matrix with selected columns.
        """
        # Convert to dense matrix for element-wise operations.
        ground_truth = self.y_true.T.toarray()  # shape = (disease, gene)
        all_1s = ground_truth == 1
        mask = (ground_truth == 0) | self.mask_1s.T
        unselected_1s = all_1s & mask
        # Identify columns with unselected 1s.
        disease_mask = unselected_1s.any(axis=1)
        return pred_truth_mat[disease_mask]

    def __iter__(self) -> Iterator[np.ndarray]:
        """
        Returns an iterator over a stacked array of predictions and ground truth values.

        This method stacks the predicted values (`y_pred`) and the ground truth values
        (`y_true`) along a new dimension to create a matrix with shape
        (disease, gene, 2), where the last dimension contains the prediction
        in the first position and the ground truth in the second position. If the
        `on_test_data_only` flag is set to `True`, the matrix is filtered to include only
        relevant test data.

        Returns:
            Iterator[np.ndarray]: An iterator over the stacked array of predictions
                and ground truth values.
        """
        # shape is (disease, gene, 2)
        pred_truth_mat = np.stack((self.y_pred.T, self.y_true.T.toarray()), axis=2)
        if self.on_test_data_only:
            pred_truth_mat = self.filter(pred_truth_mat)
        return iter(pred_truth_mat)
