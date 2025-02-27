from genepriority.evaluation import Results
import pytest
import numpy as np
import scipy.sparse as sp

# Set seed for reproducibility
np.random.seed(42)

def select(N: int, N1s: int) -> sp.csr_matrix:
    """Return an NxN sparse matrix with ones at N1s random positions."""
    indexes = np.array([(i, j) for i in range(N) for j in range(N)])
    mask = np.random.choice(np.arange(len(indexes)), N1s, replace=False)
    selected_indexes = indexes[mask]
    data = np.ones(N1s)
    return sp.coo_matrix((data, [*selected_indexes.T]), shape=(N, N)).tocsr()

@pytest.fixture(name="N", scope="session")
def matrix_size() -> int:
    """Matrix size fixture."""
    return 10

@pytest.fixture(name="N1s", scope="session")
def number_of_1s(N: int) -> int:
    """Number of ones to place in the matrix."""
    return np.random.randint(1, N * N)

@pytest.fixture(name="y_pred", scope="session")
def pred(N: int) -> np.ndarray:
    """Random prediction matrix."""
    return np.random.rand(N, N)

@pytest.fixture(name="y_true", scope="session")
def truth(N: int, N1s: int) -> sp.csr_matrix:
    """Sparse true matrix with ones at random positions."""
    return select(N, N1s)

@pytest.fixture(name="test_size", scope="session")
def test(N1s: int) -> int:
    """Test set size as a fraction of ones."""
    test_frac = 0.2
    return int(N1s * test_frac)

@pytest.fixture(name="test_mask", scope="session")
def mask(N: int, test_size: int) -> sp.csr_matrix:
    """Sparse mask matrix for the test set."""
    return select(N, test_size)

def test_result(y_true: sp.csr_matrix, y_pred: np.ndarray, test_mask: sp.csr_matrix,
                N: int, test_size: int):
    """Test the Results evaluation with and without applying a mask."""
    res = Results(y_true, y_pred, test_mask, apply_mask=False)
    assert len(res.y_true.flatten()) == N * N
    assert len(res.y_pred.flatten()) == N * N

    res.apply_mask = True
    assert len(res.y_true.flatten()) == test_size
    assert len(res.y_pred.flatten()) == test_size
