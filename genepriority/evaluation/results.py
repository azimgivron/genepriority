"""
Results module
==============
Encapsulates the results of a prediction task.
"""
import numpy as np


class Results:
    """
    Encapsulates the results of a prediction task.

    Attributes:
        y_true (np.ndarray): Ground truth association matrix.
        y_pred (np.ndarray): Matrix of gene disease association predicted by the model.
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
        self.y_true = [y_true[:, i][mask[:, i]] for i in range(y_true.shape[1])]
        self.y_pred = [y_pred[:, i][mask[:, i]] for i in range(y_pred.shape[1])]

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
