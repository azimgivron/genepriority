import pytest
import numpy as np
import pandas as pd
import scipy.sparse as sp

from NEGradient_GenePriority import (
    Indices,
    TrainingTestIndices,
    convert_dataframe_to_sparse_matrix,
    sample_zeros,
    combine_matrices,
    filter_by_number_of_association,
    create_random_splits,
    create_folds,
    compute_statistics,
)

@pytest.fixture
def sample_sparse_matrix():
    rows = [0, 1, 2]
    cols = [0, 1, 2]
    data = [1, 1, 1]
    return sp.coo_matrix((data, (rows, cols)), shape=(5, 5))

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({"gene": [0, 1, 2], "disease": [1, 2, 3]})

@pytest.fixture
def sample_indices():
    return np.array([[0, 1], [1, 2], [2, 3]])

### Test Indices Class

@pytest.fixture
def sample_indices():
    """Fixture for sample indices."""
    return np.array([[0, 1], [1, 2], [2, 3]])


@pytest.fixture
def sample_sparse_matrix():
    """Fixture for a sample sparse matrix."""
    rows = [0, 1, 2]
    cols = [1, 2, 3]
    data = [10, 20, 30]
    return sp.coo_matrix((data, (rows, cols)), shape=(5, 5))


### Tests for `Indices` class

def test_indices_init(sample_indices):
    """Test initialization of the Indices class."""
    indices = Indices(sample_indices)
    assert (indices.indices == sample_indices).all()
    assert isinstance(indices, Indices)


def test_indices_set(sample_indices):
    """Test the indices_set property."""
    indices = Indices(sample_indices)
    expected_set = {(0, 1), (1, 2), (2, 3)}
    assert indices.indices_set == expected_set


def test_indices_get_data(sample_sparse_matrix, sample_indices):
    """Test get_data method of Indices."""
    indices = Indices(sample_indices)
    submatrix = indices.get_data(sample_sparse_matrix)
    assert isinstance(submatrix, sp.coo_matrix)
    assert submatrix.nnz == len(sample_indices)


### Tests for `TrainingTestIndices` class

@pytest.fixture
def training_test_indices(sample_indices):
    """Fixture for a TrainingTestIndices object."""
    training_indices = sample_indices[:2]
    testing_indices = sample_indices[1:]
    return TrainingTestIndices(Indices(training_indices), Indices(testing_indices))


def test_training_test_indices_init(training_test_indices):
    """Test initialization of the TrainingTestIndices class."""
    assert isinstance(training_test_indices, TrainingTestIndices)
    assert isinstance(training_test_indices.training_indices, Indices)
    assert isinstance(training_test_indices.testing_indices, Indices)


### Test Utility Functions

def test_from_indices(sample_sparse_matrix, sample_indices):
    indices_set = set(map(tuple, sample_indices))
    submatrix = from_indices(sample_sparse_matrix, indices_set)
    assert isinstance(submatrix, sp.coo_matrix)

def test_convert_dataframe_to_sparse_matrix(sample_dataframe):
    sparse_matrix = convert_dataframe_to_sparse_matrix(sample_dataframe)
    assert isinstance(sparse_matrix, sp.coo_matrix)

def test_sample_zeros(sample_sparse_matrix):
    sampled = sample_zeros(sample_sparse_matrix, sampling_factor=2, seed=42)
    assert isinstance(sampled, sp.coo_matrix)
    assert sampled.nnz > 0

def test_combine_indices(sample_indices):
    indices1 = Indices(sample_indices[:2], sample_indices[2:])
    indices2 = Indices(sample_indices[:1], sample_indices[1:])
    combined = combine_indices(indices1, indices2)
    assert isinstance(combined, Indices)

def test_combine_splits(sample_indices):
    splits1 = [Indices(sample_indices[:2], sample_indices[2:])]
    splits2 = [Indices(sample_indices[:1], sample_indices[1:])]
    combined = combine_splits(splits1, splits2)
    assert len(combined) == len(splits1)
    assert isinstance(combined[0], Indices)

def test_combine_matrices(sample_sparse_matrix):
    combined = combine_matrices(sample_sparse_matrix, sample_sparse_matrix)
    assert isinstance(combined, sp.coo_matrix)

def test_filter_by_number_of_association(sample_dataframe):
    filtered = filter_by_number_of_association(
        sample_dataframe, threshold=1, col_name="gene"
    )
    assert len(filtered) > 0
    assert "gene" in filtered.columns

def test_create_random_splits(sample_indices):
    splits = create_random_splits(sample_indices, num_splits=3)
    assert len(splits) == 3
    assert isinstance(splits[0], Indices)

def test_create_folds(sample_sparse_matrix):
    folds = create_folds(sample_sparse_matrix, num_folds=3)
    assert len(folds) == 3
    assert isinstance(folds[0], Indices)

def test_compute_statistics(sample_sparse_matrix):
    splits = create_random_splits(
        np.vstack((sample_sparse_matrix.row, sample_sparse_matrix.col)).T, num_splits=2
    )
    stats = compute_statistics(sample_sparse_matrix, splits)
    assert isinstance(stats, pd.DataFrame)
    assert "Average Count" in stats.columns
