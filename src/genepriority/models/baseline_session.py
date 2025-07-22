"""
BaselineSession Module
=======================

This module provides the `BaselineSession` class, a stochastic baseline model for generating
random prediction matrices in the NEGA pipeline. It simulates binary predictions based on a
zero-sampling factor and supports serialization of session outputs.
"""

from pathlib import Path

import numpy as np
import scipy.sparse as sp

from genepriority.models.utils import check_save_name
from genepriority.utils import serialize


class BaselineSession:
    """
    BaselineSession for generating a random prediction matrix.

    This class simulates a baseline model using stochastic predictions.
    The probability of a zero in the predicted matrix is determined by the
    zero_sampling_factor provided at initialization.

    Attributes:
        shape (tuple): Dimensions of the input matrix.
        prob_zeros (float): Probability threshold for zeros.
        save_name (Path): Validated absolute path for saving session outputs.
    """

    def __init__(
        self,
        matrix: sp.csr_matrix,
        zero_sampling_factor: int,
        seed: int,
        save_name: Path,
    ):
        """
        Initialize the BaselineSession.

        Args:
            matrix (sp.csr_matrix): Input matrix in CSR format.
            zero_sampling_factor (int): Factor used to compute the probability of zeros.
            seed (int): The random seed for reproducibility.
            save_name (Path): File path where session details will be saved.
        """
        self.shape = matrix.shape
        self.prob_zeros = zero_sampling_factor / (zero_sampling_factor + 1)
        self.save_name = check_save_name(save_name)
        # Set random seed for reproducibility
        np.random.seed(seed)

    def predict_all(self) -> np.ndarray:
        """
        Generate a prediction matrix with random binary values.

        Each element in the matrix is set to 1 if a random draw exceeds the
        zero probability, otherwise 0.

        Returns:
            np.ndarray: The reconstructed (completed) matrix.
        """
        matrix = (np.random.rand(*self.shape) > self.prob_zeros).astype(np.float64)
        return matrix

    def run(self):
        """
        Save a serialized a model.
        """
        if self.save_name is not None:
            serialize(self, self.save_name)
