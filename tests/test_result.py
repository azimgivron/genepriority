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
def get_predictions(ground_truth: sp.coo_matrix) -> np.ndarray:
    """Return predictions as a dense array from the ground truth matrix."""
    return ground_truth.toarray()


@pytest.fixture(name="mask_some_1s")
def get_mask_some_1s() -> np.ndarray:
    """Return a binary mask for filtering some 1s."""
    return np.array(
        [
            [True, False, False],
            [True, False, False],
            [False, False, False],  # This disease will be gone
            [True, False, False],
            [False, False, True],
        ]
    )


@pytest.fixture(name="expected_filtered_result")
def get_expected_filtered_result() -> np.ndarray:
    """Return the expected filtered result for testing."""
    expected = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 0], [0, 0, 1]])
    return np.stack((expected, expected), axis=2)


@pytest.fixture(name="results")
def get_results(
    ground_truth: sp.coo_matrix, predictions: np.ndarray, mask_some_1s: np.ndarray
) -> Results:
    """Return a Results object initialized with test data."""
    return Results(
        y_true=ground_truth.T.tocsr(),  # Pass gene-disease shape
        y_pred=predictions.T,
        mask_1s=mask_some_1s.T,
        on_test_data_only=True,
    )


def test_results(
    ground_truth: sp.coo_matrix,
    predictions: np.ndarray,
    results: Results,
    diseases: int,
    genes: int,
    expected_filtered_result: np.ndarray,
):
    """Test the Results filter method."""
    pred_truth_mat = np.stack((predictions, ground_truth.toarray()), axis=2)
    assert pred_truth_mat.shape == (
        diseases,
        genes,
        2,
    ), f"Shape mismatch in pred_truth_mat. Expected {(diseases, genes, 2)}, but got {pred_truth_mat.shape}."

    pred_truth_mat_res = results.filter(pred_truth_mat)
    assert (pred_truth_mat_res == expected_filtered_result).all(), (
        f"Filtered results do not match expected results.\n"
        f"Filtered Result:\n{pred_truth_mat_res.T}\nExpected Result:\n{expected_filtered_result.T}"
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
