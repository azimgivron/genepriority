import pytest
import pandas as pd
import numpy as np
import scipy.sparse as sp
from NEGradient_GenePriority import (
    Indices,
    convert_dataframe_to_sparse_matrix,
    sample_zeros,
    combine_indices,
    combine_splits,
    combine_matrices,
    filter_by_number_of_association,
    create_random_splits,
    create_folds,
    compute_statistics,
)


# Test data setup
@pytest.fixture
def example_sparse_matrix():
    rows = np.array([0, 1, 2])
    cols = np.array([1, 2, 0])
    data = np.array([1, 1, 1])
    return sp.coo_matrix((data, (rows, cols)), shape=(3, 3))


@pytest.fixture
def example_dataframe():
    data = {
        "gene_id": [0, 1, 2],
        "disease_id": [1, 2, 0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def example_indices():
    return np.array([[0, 1], [1, 2], [2, 0]])


# Tests
def test_convert_dataframe_to_sparse_matrix(example_dataframe):
    sparse_matrix = convert_dataframe_to_sparse_matrix(example_dataframe)
    assert sparse_matrix.shape == (3, 3)
    assert sparse_matrix.count_nonzero() == 3
    assert np.array_equal(sparse_matrix.data, np.ones(3))


def test_sample_zeros(example_sparse_matrix):
    sampled_zeros = sample_zeros(example_sparse_matrix, sampling_factor=2)
    assert sampled_zeros.shape == example_sparse_matrix.shape
    assert sampled_zeros.count_nonzero() > 0  # New zeros were sampled


def test_combine_indices():
    indices1 = Indices(
        training_indices=np.array([[0, 1]]),
        testing_indices=np.array([[1, 2]]),
    )
    indices2 = Indices(
        training_indices=np.array([[2, 0]]),
        testing_indices=np.array([[0, 1]]),
    )
    combined = combine_indices(indices1, indices2)
    assert combined.training_indices.shape[0] == 2
    assert combined.testing_indices.shape[0] == 2


def test_combine_splits():
    splits1 = [Indices(np.array([[0, 1]]), np.array([[1, 2]]))]
    splits2 = [Indices(np.array([[2, 0]]), np.array([[0, 1]]))]
    combined_splits = combine_splits(splits1, splits2)
    assert len(combined_splits) == 1
    assert combined_splits[0].training_indices.shape[0] == 2
    assert combined_splits[0].testing_indices.shape[0] == 2


def test_combine_matrices(example_sparse_matrix):
    matrix2 = sp.coo_matrix(
        ([1, 1], ([0, 1], [2, 0])), shape=example_sparse_matrix.shape
    )
    combined_matrix = combine_matrices(example_sparse_matrix, matrix2)
    assert combined_matrix.shape == example_sparse_matrix.shape
    assert combined_matrix.count_nonzero() == 5


def test_filter_by_number_of_association(example_dataframe):
    filtered_df = filter_by_number_of_association(
        example_dataframe, threshold=1, col_name="disease_id"
    )
    assert len(filtered_df) == 3


def test_create_random_splits(example_indices):
    splits = create_random_splits(example_indices, num_splits=2)
    assert len(splits) == 2
    assert isinstance(splits[0], Indices)
    assert splits[0].training_indices.shape[0] > 0
    assert splits[0].testing_indices.shape[0] > 0


def test_create_folds(example_sparse_matrix):
    folds = create_folds(example_sparse_matrix, num_folds=2)
    assert len(folds) == 2
    assert isinstance(folds[0], Indices)
    assert folds[0].training_indices.shape[0] > 0
    assert folds[0].testing_indices.shape[0] > 0


def test_compute_statistics():
    splits = [
        Indices(
            training_indices=np.array([[0, 1], [1, 2]]),
            testing_indices=np.array([[2, 0]]),
        ),
        Indices(
            training_indices=np.array([[1, 2], [2, 0]]),
            testing_indices=np.array([[0, 1]]),
        ),
    ]
    stats = compute_statistics(splits)
    assert stats.shape[0] == 1
    assert stats.columns[-1] == "Variance of Count"
