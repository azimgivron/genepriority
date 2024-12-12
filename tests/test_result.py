import numpy as np
import pytest
import scipy.sparse as sp

from NEGradient_GenePriority import Results


@pytest.fixture(name="diseases")
def get_diseases() -> int:
    """Return the number of diseases."""
    return 5

@pytest.fixture(name="genes")
def get_genes() -> int:
    """Return the number of genes."""
    return 3

@pytest.fixture(name="ground_truth")
def get_ground_truth(diseases: int, genes: int) -> sp.coo_matrix:
    """Return the ground truth matrix as a sparse COO matrix."""
    rows = [0, 1, 1, 2, 3, 4]
    cols = [0, 0, 1, 1, 0, 2]
    return sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(diseases, genes))

@pytest.fixture(name="predictions")
def get_predictions(diseases: int, genes: int) -> np.ndarray:
    """Return predictions as a dense array from the ground truth matrix."""
    np.random.seed(42)
    return np.random.rand(diseases, genes)

@pytest.fixture(name="expected_filtered_result")
def get_expected_filtered_result() -> np.ndarray:
    """Return the expected filtered result for testing."""
    expected = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]])
    expected = np.stack((expected, expected), axis=2)
    expected = np.swapaxes(expected, 1, 2)
    return expected

@pytest.fixture(name="results")
def get_results(
    ground_truth: sp.coo_matrix, predictions: np.ndarray
) -> Results:
    """Return a Results object initialized with test data."""
    return Results(
        y_true=ground_truth.T.tocsr(),
        y_pred=predictions.T,
    )

def test_results2(results: Results, expected_filtered_result: np.ndarray):
    """Test Results iteration and filtering logic."""
    for i, (actual_result, expected_result) in enumerate(
        zip(results, expected_filtered_result)
    ):
        assert np.array_equal(actual_result, expected_result), (
            f"Mismatch at index {i}: Actual result does not match expected result.\n"
            f"Actual: \n{actual_result}\nExpected: \n{expected_result}"
        )
