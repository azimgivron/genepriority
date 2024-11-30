from typing import List

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils.extmath import cartesian


class Indices:
    """
    Represents the train and test indices for dataset splitting.

    This class encapsulates the indices used for splitting a dataset into
    training and testing subsets. It provides methods to retrieve the corresponding
    subsets from a given sparse matrix.

    Attributes:
        training_indices (np.ndarray): Array of indices representing the training data.
        testing_indices (np.ndarray): Array of indices representing the testing data.

    Methods:
        get_training_data: Retrieves the subset of the dataset corresponding to the training indices.
        get_testing_data: Retrieves the subset of the dataset corresponding to the testing indices.
    """

    def __init__(
        self,
        training_indices: np.ndarray,
        testing_indices: np.ndarray,
    ) -> None:
        """
        Initializes the Indices object with training and testing indices.

        Args:
            training_indices (np.ndarray): Indices for the training subset.
            testing_indices (np.ndarray): Indices for the testing subset.
        """
        self.training_indices = training_indices
        self.testing_indices = testing_indices

    def get_training_data(self, dataset_matrix: sp.coo_matrix) -> sp.coo_matrix:
        """
        Retrieves the training subset of the dataset as a sparse matrix.

        Args:
            dataset_matrix (sp.coo_matrix): The full dataset represented as a COO sparse matrix.

        Returns:
            sp.coo_matrix: Sparse matrix containing only the rows corresponding to the training indices.
        """
        return dataset_matrix[self.training_indices]

    def get_testing_data(self, dataset_matrix: sp.coo_matrix) -> sp.coo_matrix:
        """
        Retrieves the testing subset of the dataset as a sparse matrix.

        Args:
            dataset_matrix (sp.coo_matrix): The full dataset represented as a COO sparse matrix.

        Returns:
            sp.coo_matrix: Sparse matrix containing only the rows corresponding to the testing indices.
        """
        return dataset_matrix[self.testing_indices]


def convert_dataframe_to_sparse_matrix(dataframe: pd.DataFrame) -> sp.coo_matrix:
    """
    Converts a DataFrame of pairwise associations into a COO sparse matrix.

    Args:
        dataframe (pd.DataFrame): DataFrame containing two columns of IDs representing
                           associations.

    Returns:
        sp.coo_matrix: Sparse matrix where rows and columns correspond to the
                       two ID columns in the DataFrame, and non-zero entries
                       indicate associations.
    """
    col1_ids, col2_ids = dataframe.to_numpy().T
    association_data = np.ones(len(col1_ids))
    sparse_matrix = sp.coo_matrix((association_data, (col1_ids, col2_ids)))
    return sparse_matrix


def sample_zeros(sparse_matrix: sp.coo_matrix, sampling_factor: int) -> sp.coo_matrix:
    """
    Randomly samples zero entries from the complement of the sparse matrix's non-zero entries.

    Args:
        sparse_matrix (sp.coo_matrix): The input sparse matrix with existing non-zero entries.
        sampling_factor (int): The number of zero entries to sample, expressed as a multiple
                               of the current number of non-zero entries.

    Returns:
        sp.coo_matrix: Sparse matrix with additional sampled zero entries.
    """
    num_existing_ones = sparse_matrix.count_nonzero()
    num_sampled_zeros = num_existing_ones * sampling_factor

    total_rows, total_cols = sparse_matrix.shape
    full_indices = cartesian([np.arange(total_rows), np.arange(total_cols)])
    non_zero_indices = set(zip(sparse_matrix.row, sparse_matrix.col))
    zero_indices = np.array(
        [idx for idx in full_indices if tuple(idx) not in non_zero_indices]
    )

    sampled_indices = zero_indices[
        np.random.choice(len(zero_indices), size=num_sampled_zeros, replace=False)
    ]

    zero_data = np.zeros(len(sampled_indices))
    sampled_zeros = sp.coo_matrix(
        (zero_data, (sampled_indices[:, 0], sampled_indices[:, 1])),
        shape=sparse_matrix.shape,
    )
    return sampled_zeros


def combine_indices(indices1: Indices, indices2: Indices) -> Indices:
    """
    Combines two sets of train-test indices.

    Args:
        indices1 (Indices): The first set of indices.
        indices2 (Indices): The second set of indices.

    Returns:
        Indices: A new Indices object with combined training and testing indices.
    """
    training_indices = np.vstack((indices1.training_indices, indices2.training_indices))
    testing_indices = np.vstack((indices1.testing_indices, indices2.testing_indices))
    return Indices(training_indices, testing_indices)


def combine_splits(splits1: List[Indices], splits2: List[Indices]) -> List[Indices]:
    """
    Combines corresponding train-test splits from two lists of splits.

    Args:
        splits1 (List[Indices]): The first list of splits.
        splits2 (List[Indices]): The second list of splits.

    Returns:
        List[Indices]: A new list of Indices objects with combined training and testing indices.
    """
    return [combine_indices(split1, split2) for split1, split2 in zip(splits1, splits2)]


def combine_matrices(matrix1: sp.coo_matrix, matrix2: sp.coo_matrix) -> sp.coo_matrix:
    """
    Combines two sparse matrices into a single sparse matrix.

    Args:
        matrix1 (sp.coo_matrix): The first sparse matrix.
        matrix2 (sp.coo_matrix): The second sparse matrix.

    Returns:
        sp.coo_matrix: The combined sparse matrix.
    """
    combined_row = np.hstack([matrix1.row, matrix2.row])
    combined_col = np.hstack([matrix1.col, matrix2.col])
    combined_data = np.hstack([matrix1.data, matrix2.data])

    return sp.coo_matrix(
        (combined_data, (combined_row, combined_col)), shape=matrix1.shape
    )


def filter_by_number_of_association(
    dataframe: pd.DataFrame, threshold: int, col_name: str
) -> pd.DataFrame:
    """
    Filters rows in a DataFrame based on a minimum number of associations.

    Args:
        dataframe (pd.DataFrame): DataFrame containing association data.
        threshold (int): Minimum number of associations required for a row to be retained.
        col_name (str): Name of the column to evaluate for filtering.

    Returns:
        pd.DataFrame: Filtered DataFrame containing only rows that meet the threshold criteria.
    """
    association_counts = dataframe[col_name].value_counts()
    valid_ids = association_counts[association_counts >= threshold].index
    return dataframe[dataframe[col_name].isin(valid_ids)].reset_index(drop=True)


def create_random_splits(indices: np.ndarray, num_splits: int) -> List[Indices]:
    """
    Creates random train-test splits from an indices matrix.

    Args:
        indices (np.ndarray): Indices matrix to be split into random subsets.
        num_splits (int): Number of random splits to create.

    Returns:
        List[Indices]: List of Indices objects containing
                    training and testing subsets for each split.
    """
    splits = []
    for random_state in range(num_splits):
        train_idx, test_idx = train_test_split(
            indices,
            train_size=0.9,
            random_state=random_state,
            shuffle=True,
        )
        splits.append(Indices(train_idx, test_idx))
    return splits


def create_folds(sparse_matrix: sp.coo_matrix, num_folds: int) -> List[Indices]:
    """
    Splits a sparse matrix into K train-test folds using K-Fold cross-validation.

    Args:
        sparse_matrix (sp.coo_matrix): Sparse matrix to be split into folds.
        num_folds (int): Number of folds for the split.

    Returns:
        List[Indices]: List of Indices objects containing training and testing
                    subsets for each fold.
    """
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    indices = np.vstack((sparse_matrix.row, sparse_matrix.col)).T
    return [
        Indices(indices[train_idx], indices[test_idx])
        for train_idx, test_idx in kfold.split(indices)
    ]


def compute_statistics(splits: List[Indices]) -> pd.DataFrame:
    """
    Computes summary statistics (mean, variance) of unique testing entries across splits.

    Args:
        splits (List[Indices]): List of Indices objects.

    Returns:
        pd.DataFrame: DataFrame summarizing the counts and statistics of testing entries
                      across all splits.
    """
    counts = [len(split.testing_indices) for split in splits]
    average_count = np.mean(counts)
    variance_count = np.var(counts)
    return pd.DataFrame(
        [[*counts, average_count, variance_count]],
        columns=(
            [f"Count (Split {i+1})" for i in range(len(counts))]
            + ["Average Count", "Variance of Count"]
        ),
    )
