"""
Metrics module
===============

Implements evaluation metrics for model predictions, including ROC curves and BEDROC scores,
to assess the performance of gene prioritization models.
"""
from NEGradient_GenePriority.evaluation.evaluation_result import EvaluationResult
from NEGradient_GenePriority.evaluation.metrics import bedroc_score
from sklearn import metrics
import numpy as np


def bedroc_score(y_true, y_pred, decreasing=True, alpha=20.0):
    """
    Calculate the BEDROC (Boltzmann Enhanced Discrimination of the
    Receiver Operator Characteristic) score.

    The BEDROC metric evaluates a predictive model's ability to
    prioritize positive class instances, particularly focusing on
    early recognition.

    References:
        This code was adapted from:
        https://scikit-chem.readthedocs.io/en/latest/_modules/skchem/metrics.html

        Original paper: Truchon and Bayley (2007),
        `10.1021/ci600426e <http://dx.doi.org/10.1021/ci600426e>`_.

    Args:
        y_true (array_like): Binary class labels (1 for positive class, 0 for negative class).
        y_pred (array_like): Predicted values or scores.
        decreasing (bool): Whether higher `y_pred` values correspond to positive class.
        alpha (float): Early recognition parameter.

    Returns:
        float: BEDROC score in [0, 1], indicating the degree of early recognition
            of positive class instances.
    """
    assert len(y_true) == len(
        y_pred
    ), "The number of predictions must equal the number of labels."

    total_instances = len(y_true)
    positive_instances = sum(y_true == 1)

    if decreasing:
        order = np.argsort(-y_pred)
    else:
        order = np.argsort(y_pred)

    positive_ranks = (y_true[order] == 1).nonzero()[0] + 1
    sum_exp = np.sum(np.exp(-alpha * positive_ranks / total_instances))

    positive_ratio = positive_instances / total_instances
    random_sum = (
        positive_ratio * (1 - np.exp(-alpha)) / (np.exp(alpha / total_instances) - 1)
    )
    scaling_factor = (
        positive_ratio
        * np.sinh(alpha / 2)
        / (np.cosh(alpha / 2) - np.cosh(alpha / 2 - alpha * positive_ratio))
    )
    constant = 1 / (1 - np.exp(alpha * (1 - positive_ratio)))

    return sum_exp * scaling_factor / random_sum + constant


def evaluate_scores(
    y_true: List[int], y_pred: List[float], alphas: List[float]
) -> EvaluationResult:
    """
    Evaluates a model's predictions using ROC and BEDROC metrics.

    Args:
        y_true (List[int]): Ground truth binary labels (1 for positive, 0 for negative).
        y_pred (List[float]): Model's predicted scores or probabilities.
        alphas (List[float]): Alpha values for calculating BEDROC scores,
            where larger values emphasize early correct predictions.

    Returns:
        EvaluationResult: Contains:
            - FPR, TPR, and thresholds from the ROC curve.
            - AUC loss (1 - AUC score).
            - BEDROC scores for the given alpha values.
    """
    bedroc = {
        f"{alpha:.3f}": bedroc_score(y_true, y_pred, decreasing=True, alpha=alpha)
        for alpha in alphas
    }
    fpr, tpr, thresholds = metrics.roc_curve(
        y_true, y_pred, pos_label=1, drop_intermediate=True
    )
    auc = metrics.roc_auc_score(y_true, y_pred)
    return EvaluationResult(
        fpr=fpr, tpr=tpr, thresholds=thresholds, auc_loss=1 - auc, bedroc=bedroc
    )