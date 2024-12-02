from __future__ import annotations

import numpy as np

from NEGradient_GenePriority.preprocessing.indices import Indices

class TrainTestIndices:
    """
    Represents training and testing indices for dataset splitting.

    This class encapsulates two sets of indices: one for training and one for testing.
    It provides methods to interact with both subsets and retrieve the associated data
    from a sparse matrix. It also supports merging with another TrainTestIndices object.

    Attributes:
        training_indices (Indices): An Indices object representing the training data indices.
        testing_indices (Indices): An Indices object representing the testing data indices.
    """

    def __init__(self, training_indices: np.ndarray, testing_indices: np.ndarray):
        self.training_indices = Indices(training_indices)
        self.testing_indices = Indices(testing_indices)

    def merge(self, train_test_indices: TrainTestIndices) -> TrainTestIndices:
        """
        Merges another TrainTestIndices object into the current one.

        Args:
            train_test_indices (TrainTestIndices): Another TrainTestIndices object to merge.

        Returns:
            TrainTestIndices: A new TrainTestIndices object with merged training
                and testing indices.
        """
        training_indices = self.training_indices.merge(
            train_test_indices.training_indices
        )
        testing_indices = self.testing_indices.merge(train_test_indices.testing_indices)
        return TrainTestIndices(training_indices, testing_indices)
