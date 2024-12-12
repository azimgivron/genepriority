from typing import List
from unittest.mock import Mock

import numpy as np
import pytest
from sklearn import metrics

from NEGradient_GenePriority import Evaluation, Results, bedroc_score


@pytest.fixture(name="results")
def create_results():
    """Fixture to provide Results data for testing."""
    return [
        Results(np.array([1, 0, 1, 0, 1]), np.array([0.9, 0.1, 0.8, 0.4, 0.7]), np.array([1, 0, 1, 0, 1]).astype(bool)),
        Results(np.array([0, 1, 1, 0, 1]), np.array([0.3, 0.85, 0.65, 0.2, 0.9]), np.array([0, 1, 1, 0, 1]).astype(bool)),
    ]


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
    assert scores.shape == (len(results), len(alpha_values))


def test_compute_avg_auc_loss(results):
    """Test the compute_avg_auc_loss method."""
    evaluation = Evaluation(results)
    mean_loss, std_loss = evaluation.compute_avg_auc_loss()
    assert isinstance(mean_loss, float)
    assert isinstance(std_loss, float)


def test_compute_roc_curve(results):
    """Test the compute_roc_curve method."""
    evaluation = Evaluation(results)
    fpr, tpr = evaluation.compute_roc_curve()
    assert isinstance(fpr, np.ndarray)
    assert isinstance(tpr, np.ndarray)
    assert len(fpr) == len(tpr)
