"""
Metrics module
===============

Implements evaluation metrics for model predictions, including ROC curves and BEDROC scores,
to assess the performance of gene prioritization models.
"""
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
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch detected: 'y_true' and 'y_pred' must have the same length. "
            f"Received len(y_true)={len(y_true)} and len(y_pred)={len(y_pred)}. "
            "Ensure that the number of predicted values matches the number of true labels."
        )

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError(
            f"Invalid class labels detected in 'y_true': All values must be binary (0 or 1). "
            f"Found unique values: {np.unique(y_true)}. "
            "Please ensure your input labels are properly formatted as binary values."
        )

    if alpha <= 0:
        raise ValueError(
            f"Invalid alpha parameter: 'alpha' must be a positive number. "
            f"Received alpha={alpha}. "
            "Choose a positive value to control the emphasis on early recognition of positives."
        )

    total_instances = len(y_true)
    positive_instances = sum(y_true == 1)

    if positive_instances == 0:
        raise ValueError(
            "No positive class instances found in 'y_true'. "
            "The BEDROC score cannot be calculated without at least one positive label (1)."
        )

    if positive_instances == total_instances:
        raise ValueError(
            "No negative class instances found in 'y_true'. "
            "The BEDROC score requires both positive (1) and negative (0) class instances."
        )

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
