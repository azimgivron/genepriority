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

    def __iter__(self):
        """
        Makes the `Results` object iterable to allow unpacking with the `*` operator.

        Returns:
            Iterator: An iterator over the `y_true` and `y_pred` arrays.
        """
        return iter([self.y_true, self.y_pred])
