from dataclasses import dataclass
from typing import Any, List

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import KFold, train_test_split


class Fold:
    """
    Represents a single train-test split for a dataset.

    Attributes:
        train_idx (np.ndarray): Indices for training samples.
        test_idx (np.ndarray): Indices for test samples.
        omim (sp.coo_matrix): Sparse matrix representation of the dataset.
    """

    def __init__(
        self, train_idx: np.ndarray, test_idx: np.ndarray, omim: sp.coo_matrix
    ) -> None:
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.omim = omim

    @property
    def train(self) -> sp.coo_matrix:
        """Returns the training set as a sparse matrix."""
        return self.omim[self.train_idx]

    @property
    def test(self) -> sp.coo_matrix:
        """Returns the test set as a sparse matrix."""
        return self.omim[self.test_idx]


@dataclass
class SplitResult:
    """
    Represents a single train-test split result.

    Attributes:
        train (sp.coo_matrix): Sparse matrix for the training set.
        test (sp.coo_matrix): Sparse matrix for the test set.
    """

    train: sp.coo_matrix
    test: sp.coo_matrix


def omim_as_coo(gene_disease: pd.DataFrame) -> sp.coo_matrix:
    """
    Converts a gene-disease DataFrame into a COO sparse matrix.

    Args:
        gene_disease (pd.DataFrame): DataFrame with gene and disease columns.

    Returns:
        sp.coo_matrix: Sparse matrix representation of gene-disease associations.
    """
    geneIDs, diseaseIDs = gene_disease.to_numpy().T
    data = np.ones(len(geneIDs))
    omim = sp.coo_matrix((data, (geneIDs, diseaseIDs)))
    return omim


def from_omim2_to_folds(omim2: sp.coo_matrix, n_folds: int) -> List[Fold]:
    """
    Splits a COO sparse matrix into K folds.

    Args:
        omim2 (sp.coo_matrix): Sparse matrix to be split.
        n_folds (int): Number of folds.

    Returns:
        List[Fold]: List of Fold objects containing train and test splits.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    folds = []
    row_indices = np.arange(omim2.shape[0])  # Rows for splitting
    for train_idx, test_idx in kf.split(row_indices):
        folds.append(Fold(train_idx, test_idx, omim2))
    return folds


def sample_zeros(omim: sp.coo_matrix, factor: int, replace=True) -> sp.coo_matrix:
    """
    Samples zeros from a COO sparse matrix and adds them to the dataset.

    Args:
        omim (sp.coo_matrix): Original sparse matrix.
        factor (int): Factor by which to sample zeros compared to existing ones.
        replace (bool, optional): Whether the sampling is with replacement.
            Defaults to True.

    Returns:
        sp.coo_matrix: Updated sparse matrix with sampled zeros.
    """
    num_ones = omim.count_nonzero()
    num_zeros = num_ones * factor

    rows_allowed_indices = np.arange(omim.shape[0])[~omim.row]
    col_allowed_indices = np.arange(omim.shape[1])[~omim.col]

    row_indices = np.random.choice(
        rows_allowed_indices, size=num_zeros, replace=replace
    )
    col_indices = np.random.choice(col_allowed_indices, size=num_zeros, replace=replace)

    data = np.zeros(num_zeros)
    zeros = sp.coo_matrix((data, (row_indices, col_indices)), shape=omim.shape)

    new_row = np.hstack([omim.row, zeros.row])
    new_col = np.hstack([omim.col, zeros.col])
    new_data = np.hstack([omim.data, zeros.data])

    new_omim = sp.coo_matrix((new_data, (new_row, new_col)), shape=omim.shape)
    return new_omim


def gene_disease_to_omim2_df(
    gene_disease: pd.DataFrame, threshold: int
) -> pd.DataFrame:
    """
    Filters diseases with a minimum number of associations.

    Args:
        gene_disease (pd.DataFrame): DataFrame with gene-disease associations.
        threshold (int): Minimum number of associations for a disease.

    Returns:
        pd.DataFrame: Filtered DataFrame with valid diseases.
    """
    disease_counts = gene_disease["disease ID"].value_counts()
    valid_diseases = disease_counts[disease_counts >= threshold].index
    omim2_df = gene_disease[
        gene_disease["disease ID"].isin(valid_diseases)
    ].reset_index(drop=True)
    return omim2_df


def disease_count(splits: List[SplitResult]) -> pd.DataFrame:
    """
    Computes disease counts for each split and their statistics.

    Args:
        splits (List[SplitResult]): List of SplitResult objects.

    Returns:
        pd.DataFrame: DataFrame with counts and statistics for diseases in splits.
    """
    count = []
    for split in splits:
        columns_with_ones = np.unique(split.test.col[split.test.data == 1])
        count.append(len(columns_with_ones))
    mean = np.mean(count)
    var = np.var(count)
    count.extend([mean, var])
    df = pd.DataFrame(
        [count],
        columns=(
            [f"count on random split {i+1}" for i in range(len(count) - 2)]
            + ["average number of diseases", "variance of the number of diseases"]
        ),
    )
    return df


def from_omim1_to_splits(omim1: sp.coo_matrix, n_splits: int) -> List[SplitResult]:
    """
    Splits a COO sparse matrix into random train-test splits.

    Args:
        omim1 (sp.coo_matrix): Sparse matrix to be split.
        n_splits (int): Number of random splits.

    Returns:
        List[SplitResult]: List of SplitResult objects for train-test splits.
    """
    splits = []
    for random_state in range(n_splits):
        train, test = train_test_split(
            omim1,
            test_size=None,
            train_size=0.9,
            random_state=random_state,
            shuffle=True,
            stratify=None,
        )
        splits.append(SplitResult(train.tocoo(), test.tocoo()))
    return splits
