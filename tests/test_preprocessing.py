"""Preprocessing test module"""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from NEGradient_GenePriority import (
    Indices,
    TrainTestIndices,
    combine_matrices,
    combine_splits,
    compute_statistics,
    convert_dataframe_to_sparse_matrix,
    create_folds,
    create_random_splits,
    filter_by_number_of_association,
    from_indices,
    sample_zeros,
)


@pytest.fixture(name="sample_sparse_matrix")
def get_sparse_matrix() -> sp.coo_matrix:
    """Fixture: Create a sample sparse COO matrix."""
    rows = [0, 1, 2]
    cols = [0, 1, 2]
    data = [1, 1, 1]
    return sp.coo_matrix((data, (rows, cols)), shape=(5, 5))


@pytest.fixture(name="sample_dataframe")
def get_dataframe() -> pd.DataFrame:
    """Fixture: Create a sample pandas DataFrame."""
    return pd.DataFrame({"gene": [0, 1, 2], "disease": [1, 2, 3]})


@pytest.fixture(name="sample_indices")
def get_indices() -> np.ndarray:
    """Fixture: Create sample indices for testing."""
    return np.array([[0, 0], [1, 1], [2, 2]])


### Tests for `Indices` class


def test_indices_init(sample_indices: np.ndarray):
    """Test initialization of the Indices class."""
    indices = Indices(sample_indices)
    assert (indices.indices == sample_indices).all()
    assert isinstance(indices, Indices)


def test_indices_set(sample_indices: np.ndarray):
    """Test the indices_set property."""
    indices = Indices(sample_indices)
    expected_set = {(0, 0), (1, 1), (2, 2)}
    assert indices.indices_set == expected_set


def test_indices_get_data(
    sample_sparse_matrix: sp.coo_matrix, sample_indices: np.ndarray
):
    """Test the get_data method of Indices."""
    indices = Indices(sample_indices)
    submatrix = indices.get_data(sample_sparse_matrix)
    assert isinstance(submatrix, sp.csr_matrix)
    assert submatrix.nnz == len(sample_indices)


### Tests for `TrainTestIndices` class


@pytest.fixture(name="training_test_indices")
def tt_indices(sample_indices: np.ndarray) -> TrainTestIndices:
    """Fixture: Create a TrainTestIndices object."""
    training_indices = sample_indices[:2]
    testing_indices = sample_indices[1:]
    return TrainTestIndices(Indices(training_indices), Indices(testing_indices))


def test_training_test_indices_init(training_test_indices: TrainTestIndices):
    """Test initialization of the TrainTestIndices class."""
    assert isinstance(training_test_indices, TrainTestIndices)
    assert isinstance(training_test_indices.training_indices, Indices)
    assert isinstance(training_test_indices.testing_indices, Indices)


### Test Utility Functions


def test_from_indices(
    sample_sparse_matrix: sp.coo_matrix, sample_indices: np.ndarray
):
    """Test the from_indices function."""
    indices_set = set(map(tuple, sample_indices))
    submatrix = from_indices(sample_sparse_matrix, indices_set)
    assert isinstance(submatrix, sp.coo_matrix)


def test_convert_dataframe_to_sparse_matrix(sample_dataframe: pd.DataFrame):
    """Test the convert_dataframe_to_sparse_matrix function."""
    sparse_matrix = convert_dataframe_to_sparse_matrix(sample_dataframe)
    assert isinstance(sparse_matrix, sp.coo_matrix)


def test_sample_zeros(sample_sparse_matrix: sp.coo_matrix):
    """Test the sample_zeros function."""
    sampled = sample_zeros(sample_sparse_matrix, sampling_factor=2, seed=42)
    assert isinstance(sampled, sp.coo_matrix)
    assert sampled.nnz > 0


def test_combine_indices(sample_indices: np.ndarray):
    """Test merging of TrainTestIndices objects."""
    indices1 = TrainTestIndices(
        Indices(sample_indices[:2]), Indices(sample_indices[2:])
    )
    indices2 = TrainTestIndices(
        Indices(sample_indices[:1]), Indices(sample_indices[1:])
    )
    combined = indices1.merge(indices2)
    assert isinstance(combined, TrainTestIndices)


def test_combine_splits(sample_indices: np.ndarray):
    """Test combining splits of TrainTestIndices."""
    splits1 = [
        TrainTestIndices(Indices(sample_indices[:2]), Indices(sample_indices[2:]))
    ]
    splits2 = [
        TrainTestIndices(Indices(sample_indices[:1]), Indices(sample_indices[1:]))
    ]
    combined = combine_splits(splits1, splits2)
    assert len(combined) == len(splits1)
    assert isinstance(combined[0], TrainTestIndices)


def test_combine_matrices(sample_sparse_matrix: sp.coo_matrix):
    """Test the combine_matrices function."""
    combined = combine_matrices(sample_sparse_matrix, sample_sparse_matrix)
    assert isinstance(combined, sp.coo_matrix)


def test_filter_by_number_of_association(sample_dataframe: pd.DataFrame):
    """Test the filter_by_number_of_association function."""
    filtered = filter_by_number_of_association(
        sample_dataframe, threshold=1, col_name="gene"
    )
    assert len(filtered) > 0
    assert "gene" in filtered.columns


def test_create_random_splits(sample_sparse_matrix: sp.coo_matrix):
    """Test the create_random_splits function."""
    splits = create_random_splits(sample_sparse_matrix, num_splits=3)
    assert len(splits) == 3
    assert isinstance(splits[0], TrainTestIndices)


def test_create_folds(sample_sparse_matrix: sp.coo_matrix):
    """Test the create_folds function."""
    folds = create_folds(sample_sparse_matrix, num_folds=3)
    assert len(folds) == 3
    assert isinstance(folds[0], TrainTestIndices)


def test_compute_statistics(sample_sparse_matrix: sp.coo_matrix):
    """Test the compute_statistics function."""
    splits = create_random_splits(sample_sparse_matrix, num_splits=2)
    stats = compute_statistics(sample_sparse_matrix, splits)
    assert isinstance(stats, pd.DataFrame)
