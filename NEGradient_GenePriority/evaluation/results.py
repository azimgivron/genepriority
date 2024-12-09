"""
Results module
==============

Contains a data structure for simulation results.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Results:
    """
    Prediction results data structure.

    Attributes:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values from the trained model.
    """

    y_true: np.ndarray
    y_pred: np.ndarray

    def __post_init__(self):
        if not isinstance(self.y_true, np.ndarray):
            raise TypeError(
                f"`y_true` must be of type np.ndarray, but got type {type(self.y_true)} instead."
            )
        if not isinstance(self.y_pred, np.ndarray):
            raise TypeError(
                f"`y_pred` must be of type np.ndarray, but got type {type(self.y_pred)} instead."
            )
        if len(self.y_true.shape) != len(self.y_pred.shape):
            raise ValueError(
                f"Mismatched dimensions: `y_true` has {len(self.y_true.shape)} dimensions, "
                f"while `y_pred` has {len(self.y_pred.shape)} dimensions. They must be identical."
            )
        if len(self.y_true) != len(self.y_pred):
            raise ValueError(
                f"Mismatched lengths: `y_true` contains {len(self.y_true)} elements, "
                f"while `y_pred` contains {len(self.y_pred)} elements. They must be identical."
            )

    def __iter__(self):
        """
        Makes the `Results` object iterable to allow unpacking with the `*` operator.

        Returns:
            Iterator: An iterator over the `y_true` and `y_pred` arrays.
                Order is `y_true` followed by `y_pred`.
        """
        return iter([self.y_true, self.y_pred])
