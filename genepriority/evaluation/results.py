"""
Results module
==============
Provides a data structure for storing and processing simulation results
of prediction tasks, including both ground truth values and model-predicted values.
It supports input validation and optional masking of results via a test mask.

"""
import numpy as np
import scipy.sparse as sp


class Results:
    """
    Encapsulates the results of a prediction task.

    This class provides a structure to store and process prediction results,
    including ground truth values and corresponding predicted values from a model.
    It validates the input data and supports optional application of a test maskt.

    Attributes:
        y_true (sp.csr_matrix): Sparse matrix of ground truth values. When the flag
            `apply_mask` is True, only the entries specified by the test mask are returned.
        y_pred (np.ndarray): Dense array of predicted values. When the flag
            `apply_mask` is True, only the entries specified by the test mask are returned.
        test_mask (sp.csr_matrix): Sparse matrix serving as a mask to identify
            test set entries (no 0s) for selective evaluation.
        apply_mask (bool): Flag indicating whether to apply the test mask when
            accessing the results.
    """

    _y_true: sp.csr_matrix
    _y_pred: np.ndarray
    test_mask: sp.csr_matrix
    apply_mask: bool

    def __init__(
        self,
        y_true: sp.csr_matrix,
        y_pred: np.ndarray,
        test_mask: sp.csr_matrix = None,
        apply_mask: bool = False,
    ):
        """
        Initializes the Results object.

        Validates input types and shapes, and sets up internal data structures.

        Args:
            y_true (sp.csr_matrix): Ground truth sparse matrix, where each entry
                represents the true association (e.g., between a disease and a gene).
            y_pred (np.ndarray): Predicted values as a dense array, where each entry
                represents the likelihood of an association predicted by the model.
            test_mask (sp.csr_matrix, optional): Sparse matrix serving as a mask to identify
                the test set entries. Default to None.
            apply_mask (bool, optional): Whether to apply the test mask when accessing the
                results. Default to False.

        Raises:
            TypeError: If y_true is not a sp.csr_matrix, y_pred is not a np.ndarray,
                or test_mask is not a sp.csr_matrix.
            ValueError: If the shapes of y_true and y_pred do not match.
        """
        if not isinstance(y_true, sp.csr_matrix):
            raise TypeError(
                "`y_true` must be of type sp.csr_matrix, but "
                f"got type {type(y_true)} instead."
            )
        if not isinstance(y_pred, np.ndarray):
            raise TypeError(
                "`y_pred` must be of type np.ndarray, but got "
                f"type {type(y_pred)} instead."
            )
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: `y_true` has shape {y_true.shape}, "
                f"but `y_pred` has shape {y_pred.shape}. Shapes must match."
            )
        if test_mask is not None and not isinstance(test_mask, sp.csr_matrix):
            raise TypeError(
                "`test_mask` must be of type sp.csr_matrix, but "
                f"got type {type(test_mask)} instead."
            )
        self._y_true = y_true
        self._y_pred = y_pred
        test_mask = (
            test_mask
            if test_mask is not None
            else sp.coo_matrix(([], ([], [])), shape=y_true.shape)
        )
        mask = y_true.copy()
        mask.data[mask.data != 0] = 1
        self.test_mask = test_mask.multiply(mask)
        self.apply_mask = apply_mask
        
    @property
    def np_mask(self) -> np.ndarray:
        """A boolean numpy mask.

        Returns:
            np.ndarray: The mask.
        """
        return self.test_mask.toarray().astype(bool)
        
    @property
    def y_true(self) -> np.ndarray:
        """
        Returns the ground truth values as a dense numpy array.

        If `apply_mask` is True, the test mask is applied to the ground truth
        sparse matrix before conversion to a dense array.

        Returns:
            np.ndarray: Dense array representation of the ground truth values.
        """
        return (
            self._y_true.toarray()[self.np_mask]
            if self.apply_mask
            else self._y_true.toarray()
        )

    @property
    def y_pred(self) -> np.ndarray:
        """
        Returns the predicted values as a dense numpy array.

        If `apply_mask` is True, the test mask is applied to the predicted values.

        Returns:
            np.ndarray: Dense array representation of the predicted values.
        """
        return self._y_pred[self.np_mask] if self.apply_mask else self._y_pred
