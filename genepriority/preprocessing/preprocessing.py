# pylint: disable=R0914
"""
Preprocessing module
=====================

Provides utilities for data preprocessing, including dataset index management, 
train-test splitting, data format conversions, and statistical analysis, 
streamlining data preparation workflows.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from genepriority.preprocessing.train_test_masks import TrainTestMasks


def convert_dataframe_to_sparse_matrix(
    dataframe: pd.DataFrame, shape: Tuple[int, int]
) -> sp.csr_matrix:
    """
    Converts a DataFrame of pairwise associations into a COO sparse matrix.

    Args:
        dataframe (pd.DataFrame): DataFrame containing two columns of IDs representing
            associations.
        shape (Tuple[int, int]): The shape of the output matrix.

    Returns:
        sp.csr_matrix: Sparse matrix where rows and columns correspond to the
            two ID columns in the DataFrame, and non-zero entries indicate
            associations.
    """
    row, col = dataframe.to_numpy().T
    association_data = np.ones(len(row))
    return sp.coo_matrix((association_data, (row, col)), shape=shape).tocsr()


def sample_zeros(
    sparse_matrix: sp.csr_matrix, sampling_factor: int, seed: int = None
) -> sp.csr_matrix:
    """
    Randomly samples zero entries and add them.

    Args:
        sparse_matrix (sp.csr_matrix): The input sparse matrix with existing non-zero entries.
        sampling_factor (int): The number of zero entries to sample, expressed as a multiple
            of the current number of non-zero entries.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        sp.csr_matrix: Sparse matrix with additional sampled zeros.
    """
    if seed is not None:
        np.random.seed(seed)

    nnz = sparse_matrix.nnz * sampling_factor
    current_nnz = 0
    row, col = sparse_matrix.shape
    output = sp.coo_matrix(([], ([], [])), shape=(row, col))
    while current_nnz < nnz:
        row_indices = np.random.randint(0, row, size=nnz - current_nnz)
        col_indices = np.random.randint(0, col, size=nnz - current_nnz)
        output += sp.coo_matrix(
            (-np.ones_like(col_indices), (row_indices, col_indices)),
            shape=(row, col),
        ).tocsr()
        output -= output.multiply(sparse_matrix)
        current_nnz = output.nnz

    output += sparse_matrix
    output.data[output.data <= -1] = 0
    return output.tocsr()


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


def compute_statistics(
    sparse_matrix: sp.coo_matrix, splits: List[TrainTestMasks]
) -> pd.DataFrame:
    """
    Computes summary statistics (mean, variance) of unique testing entries across splits.

    Args:
        sparse_matrix (sp.coo_matrix): The input sparse matrix.
        splits (List[TrainTestMasks]): List of TrainTestMasks objects.

    Returns:
        pd.DataFrame: DataFrame summarizing the counts and statistics of testing entries
            across all splits.
    """
    counts = []
    for _, test_mask in splits:
        data = sp.find(sparse_matrix.multiply(test_mask))[1]
        counts.append(len(np.unique(data)))
    average_count = np.mean(counts)
    variance_count = np.std(counts)
    return pd.DataFrame(
        [[*counts, average_count, variance_count]],
        columns=(
            [f"Split {i+1}" for i in range(len(counts))]
            + ["Average", "Standard Deviation"]
        ),
        index=["Counts"],
    ).T
