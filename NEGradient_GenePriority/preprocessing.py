from typing import List, Set, Tuple
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import KFold, train_test_split
from dataclasses import dataclass

class Indices:
    """
    Encapsulates a set of indices for a dataset.

    This class is designed to manage a collection of row-column indices representing a subset
    of a dataset. It provides methods to retrieve the corresponding data from a given sparse
    matrix, as well as a set representation for efficient operations.

    Attributes:
        indices (np.ndarray): A 2D array of shape (n, 2), where each row represents
                              a (row, column) pair of indices.

    Methods:
        indices_set: Converts the indices into a set of (row, column) tuples for efficient lookups.
        get_data: Retrieves the subset of a dataset corresponding to the stored indices.
    """

    def __init__(self, indices: np.ndarray) -> None:
        """
        Initializes the Indices object with the given array of indices.

        Args:
            indices (np.ndarray): A 2D array of shape (n, 2) containing row-column pairs.
        """
        self.indices = indices

    @property
    def indices_set(self) -> Set[Tuple[int, int]]:
        """
        Converts the indices into a set of (row, column) tuples.

        Returns:
            Set[Tuple[int, int]]: A set of (row, column) tuples for the indices.
        """
        return set(zip(self.indices[:, 0], self.indices[:, 1]))

    def get_data(self, dataset_matrix: sp.coo_matrix) -> sp.coo_matrix:
        """
        Retrieves the subset of the dataset corresponding to the stored indices.

        Args:
            dataset_matrix (sp.coo_matrix): The full dataset represented as a COO sparse matrix.

        Returns:
            sp.coo_matrix: A sparse matrix in COO format containing only the elements
                           specified by the indices. The shape of the returned matrix
                           matches the shape of the original dataset.
        """
        return from_indices(dataset_matrix, self.indices_set)


@dataclass
class TrainingTestIndices:
    """
    Represents training and testing indices for dataset splitting.

    This class encapsulates two sets of indices: one for training and one for testing.
    It provides methods to interact with both subsets and retrieve the associated data
    from a sparse matrix.

    Attributes:
        training_indices (Indices): An Indices object representing the training data indices.
        testing_indices (Indices): An Indices object representing the testing data indices.
    """

    training_indices: Indices
    testing_indices: Indices


def from_indices(
    dataset_matrix: sp.coo_matrix, indices_set: Set[Tuple[int, int]]
) -> sp.coo_matrix:
    """
    Extracts a submatrix from the given sparse matrix based on specified row-column indices.

    Args:
        dataset_matrix (sp.coo_matrix): The input sparse matrix from which elements are to be extracted.
        indices (Set[Tuple[int, int]]): A set of indices of the elements
                              to extract. Each element of the set is representing the
                              (row, column) coordinates.

    Returns:
        sp.coo_matrix: A sparse matrix in COO format containing only the elements specified by
                       the indices array. The submatrix will have the same shape as the original
                       matrix, but only the specified elements will be retained.
    """
    # Create a mask for entries in dataset_matrix that match the indices
    mask = [
        (row, col) in indices_set
        for row, col in zip(dataset_matrix.row, dataset_matrix.col)
    ]

    # Filter out only the relevant entries
    rows = dataset_matrix.row[mask]
    cols = dataset_matrix.col[mask]
    data = dataset_matrix.data[mask]

    # Create a new sparse matrix from the filtered data
    submatrix = sp.coo_matrix((data, (rows, cols)), shape=dataset_matrix.shape)
    return submatrix


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


def sample_zeros(sparse_matrix: sp.coo_matrix, sampling_factor: int, seed: int = None) -> sp.coo_matrix:
    """
    Randomly samples zero entries from the complement of the sparse matrix's non-zero entries.

    Args:
        sparse_matrix (sp.coo_matrix): The input sparse matrix with existing non-zero entries.
        sampling_factor (int): The number of zero entries to sample, expressed as a multiple
                               of the current number of non-zero entries.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        sp.coo_matrix: Sparse matrix of sampled zeros.
    """
    # Set the random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Get the number of existing non-zero entries
    num_existing_ones = sparse_matrix.nnz
    num_sampled_zeros = num_existing_ones * sampling_factor

    # Get the matrix dimensions and existing non-zero indices
    total_rows, total_cols = sparse_matrix.shape
    non_zero_indices = set(zip(sparse_matrix.row, sparse_matrix.col))

    # Initialize sampling
    sampled_zeros = []
    sampled_count = 0

    while sampled_count < num_sampled_zeros:
        # Randomly sample row and column indices
        row_indices = np.random.randint(
            0, total_rows, size=num_sampled_zeros - sampled_count
        )
        col_indices = np.random.randint(
            0, total_cols, size=num_sampled_zeros - sampled_count
        )
        new_indices = zip(row_indices, col_indices)

        # Filter out indices that already exist in the sparse matrix
        unique_indices = [idx for idx in new_indices if idx not in non_zero_indices]

        # Add unique indices to the sampled set
        sampled_zeros.extend(unique_indices)
        sampled_count += len(unique_indices)

    # Convert sampled indices to sparse COO matrix
    sampled_zeros = np.array(sampled_zeros[:num_sampled_zeros])
    zero_data = np.zeros(len(sampled_zeros))
    result = sp.coo_matrix(
        (zero_data, (sampled_zeros[:, 0], sampled_zeros[:, 1])),
        shape=sparse_matrix.shape,
    )
    return result


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
        train_test_indices = TrainingTestIndices(Indices(train_idx), Indices(test_idx))
        splits.append(train_test_indices)
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
        TrainingTestIndices(Indices(indices[train_idx]), Indices(indices[test_idx]))
        for train_idx, test_idx in kfold.split(indices)
    ]


def compute_statistics(
    sparse_matrix: sp.coo_matrix, splits: List[Indices]
) -> pd.DataFrame:
    """
    Computes summary statistics (mean, variance) of unique testing entries across splits.

    Args:
        sparse_matrix (sp.coo_matrix): The input sparse matrix.
        splits (List[Indices]): List of Indices objects.

    Returns:
        pd.DataFrame: DataFrame summarizing the counts and statistics of testing entries
                      across all splits.
    """
    counts = []
    for split in splits:
        data = split.testing_indices.get_data(sparse_matrix)
        columns_with_ones = np.unique(data.col[data.data == 1])
        counts.append(len(columns_with_ones))
    average_count = np.mean(counts)
    variance_count = np.std(counts)
    return pd.DataFrame(
        [[*counts, average_count, variance_count]],
        columns=(
            [f"Count (Split {i+1})" for i in range(len(counts))]
            + ["Average Count", "Standard deviation of Count"]
        ),
    )
