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

from genepriority.preprocessing.train_val_test_mask import TrainValTestMasks


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
    Randomly samples zero entries from the input sparse matrix and explicitly adds them.

    Args:
        sparse_matrix (sp.csr_matrix): The input sparse matrix containing non-zero entries.
        sampling_factor (int): Factor that determines the number of zero entries to sample.
            The total number of zeros to sample will be `sparse_matrix.nnz * sampling_factor`.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        sp.csr_matrix: A new sparse matrix with the same shape as the input matrix, where the
            sampled zero entries are explicitly added along with the existing non-zero entries.
    """
    if seed is not None:
        np.random.seed(seed)

    nb_samples = sparse_matrix.nnz * sampling_factor
    dense_matrix = sparse_matrix.toarray()

    zero_indices = np.argwhere(dense_matrix == 0)
    if nb_samples > zero_indices.shape[0]:
        raise ValueError(
            "sampling_factor results in more zero samples than available zeros."
        )

    selected_idx = np.random.choice(
        zero_indices.shape[0], size=nb_samples, replace=False
    )
    selected_zero_indices = zero_indices[selected_idx]

    nonzero_indices = np.argwhere(dense_matrix != 0)
    all_indices = np.vstack((nonzero_indices, selected_zero_indices))
    rows = all_indices[:, 0]
    cols = all_indices[:, 1]
    data = dense_matrix[rows, cols]
    return sp.csr_matrix((data, (rows, cols)), shape=dense_matrix.shape)


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
    sparse_matrix: sp.coo_matrix, folds: List[TrainValTestMasks]
) -> pd.DataFrame:
    """
    Computes summary statistics (mean, variance) of unique testing entries across splits.

    Args:
        sparse_matrix (sp.coo_matrix): The input sparse matrix.
        folds (List[TrainTestMasks]): List of TrainTestMasks objects.

    Returns:
        pd.DataFrame: DataFrame summarizing the counts and statistics of testing entries
            across all splits.
    """
    counts = []
    for _, test_mask, _ in folds:
        data = sp.find(sparse_matrix.multiply(test_mask))[1]
        counts.append(len(set(data)))
    average_count = np.mean(counts)
    variance_count = np.std(counts)
    return pd.DataFrame(
        [[*counts, average_count, variance_count]],
        columns=(
            [f"Fold {i+1}" for i in range(len(counts))]
            + ["Average", "Standard Deviation"]
        ),
        index=["Counts"],
    ).T
