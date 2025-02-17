from typing import List

import numpy as np
import pytest
import scipy.sparse as sp
from genepriority import Evaluation, Results


@pytest.fixture(name="diseases")
def get_diseases() -> int:
    """Return the number of diseases."""
    return 5


@pytest.fixture(name="genes")
def get_genes() -> int:
    """Return the number of genes."""
    return 3


@pytest.fixture(name="results")
def create_results(genes: int, diseases: int):
    """Fixture to provide Results data for testing."""
    np.random.seed(42)
    y_true = sp.coo_matrix(
        (np.ones(5), (np.hstack((np.zeros(3), np.ones(2))), np.array([0, 1, 2, 3, 4]))),
        shape=(genes, diseases),
    ).tocsr()
    y_pred1 = np.random.rand(genes, diseases)
    y_pred2 = np.random.rand(genes, diseases)
    res1 = Results(y_true, y_pred1)
    res2 = Results(y_true, y_pred2)
    return [res1, res2]


def test_evaluation_init(results: List[Results]):
    """Test the __init__ method of Evaluation class."""
    evaluation = Evaluation(results)
    assert all(res1 == res2 for res1, res2 in zip(evaluation.results, results))
    assert all(isinstance(result, Results) for result in evaluation.results)


def test_evaluation_init_invalid():
    """Test __init__ method with invalid input."""
    with pytest.raises(TypeError):
        Evaluation(["invalid", 123])


def test_compute_bedroc_scores(results):
    """Test the compute_bedroc_scores method."""
    alpha_values = [0.1, 0.5, 1.0]
    Evaluation.alphas = alpha_values
    Evaluation.alpha_map = {alpha: f"alpha_{alpha}" for alpha in alpha_values}
    evaluation = Evaluation(results)
    scores = evaluation.compute_bedroc_scores()
    assert scores.shape == (len(evaluation.results), len(alpha_values))


def test_compute_avg_auc_loss(results):
    """Test the compute_avg_auc_loss method."""
    evaluation = Evaluation(results)
    loss = evaluation.compute_avg_auc_loss()
    assert isinstance(loss, np.ndarray)
    assert loss.shape == (len(evaluation.results),)


def test_compute_roc_curve(results):
    """Test the compute_roc_curve method."""
    evaluation = Evaluation(results)
    fpr_tpr_per_disease = evaluation.compute_roc_curve()
    assert isinstance(fpr_tpr_per_disease, np.ndarray)
    assert fpr_tpr_per_disease.shape[0] == 2
