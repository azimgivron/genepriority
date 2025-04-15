"""
EarlyStopping Module
======================

Provides tools for monitoring validation loss and early stopping for machine learning models.
The EarlyStopping class tracks recent losses and corresponding model weights. If no improvement
is observed within a specified patience interval, it signals for early termination of training.
"""
from collections import deque
from typing import Any, Tuple

import numpy as np


class EarlyStopping:
    """
    A callable class to determine early stopping during model training based on loss improvement.

    This class tracks the losses and corresponding model weights over a specified
    number of iterations (patience). If the losses do not show a favorable improvement
    over the tracked iterations, the early stopping condition is met.

    Attributes:
        losses (deque): A deque to store recent loss values, with a maximum length equal to
            the patience + 1.
        weights (deque): A deque to store tuples of model weights corresponding to the
            recent losses.
    """

    def __init__(self, patience: int):
        """
        Initializes the EarlyStopping instance with a fixed patience parameter.

        Args:
            patience (int): The number of recent epochs/iterations to consider when evaluating the
                stopping condition.
        """
        self.losses = deque(maxlen=patience + 1)
        self.weights = deque(maxlen=patience)

    def __call__(self, loss: float, h1: np.ndarray, h2: np.ndarray) -> bool:
        """
        Update the early stopping criteria with a new loss and model weights.

        On each call, the new loss and a copy of the provided weights (h1 and h2)
        are stored. If the loss deque is full and the condition indicating no improvement
        is met, the function returns True to signal early stopping. Otherwise, it appends
        the loss and weights for further tracking.

        Note: The current stopping criterion checks if the current sequence of losses in the deque
        does not show improvement by comparing them in an element-wise fashion. Adjust the
        comparison as needed based on your improvement metric.

        Args:
            loss (float): The latest computed loss value.
            h1 (np.ndarray): The first set of model parameters (or weights) to be tracked.
            h2 (np.ndarray): The second set of model parameters (or weights) to be tracked.

        Returns:
            bool: True if early stopping condition is met (i.e., no improvement observed),
                  False otherwise.
        """
        self.losses.append(loss)
        losses_array = np.array(self.losses)
        if (
            len(self.losses) == self.losses.maxlen
            and (losses_array[:-1] - losses_array[1:] <= 0).all()
        ):
            return True
        self.weights.append((h1.copy(), h2.copy()))
        return False

    @property
    def best_weights(self) -> Tuple[Any, Any]:
        """
        Retrieve the best weights stored based on the early stopping criteria.

        Returns the first tuple of weights stored in the deque, which typically corresponds
        to the best performing weights according to the early stopping tracking logic.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The model weights (h1, h2) corresponding to
                the best loss.
        """
        if not self.weights:
            raise ValueError("No weights have been stored yet.")
        return self.weights[0]
