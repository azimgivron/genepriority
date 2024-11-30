from dataclasses import dataclass
from typing import List

from metrics import bedroc_score
from sklearn import metrics


@dataclass
class EvaluationResult:
    """
    A class to store the results of the evaluation metrics.

    Attributes:
        fpr (list): False positive rates at various thresholds.
        tpr (list): True positive rates at various thresholds.
        thresholds (list): Threshold values used to compute FPR and TPR.
        auc_loss (float): Loss derived from the area under the ROC curve (1 - AUC).
        bedroc (list): BEDROC scores computed for specified alpha values.
    """

    fpr: List[float]
    tpr: List[float]
    thresholds: List[float]
    auc_loss: float
    bedroc: List[float]


def evaluate(
    y_true: List[int], y_score: List[float], alphas: List[float]
) -> EvaluationResult:
    """
    Evaluate a classifier's performance using various metrics.

    Args:
        y_true (List[int]): True binary labels for the samples (0 or 1).
        y_score (List[float]): Predicted scores or probabilities for the samples.
        alphas (List[float]): List of alpha values for computing BEDROC scores.

    Returns:
        EvaluationResult: A dataclass instance containing evaluation metrics:
                          - FPR, TPR, thresholds from ROC curve.
                          - AUC loss (1 - AUC score).
                          - BEDROC scores for given alphas.
    """
    bedroc = [
        bedroc_score(y_true, y_score, decreasing=True, alpha=alpha) for alpha in alphas
    ]
    fpr, tpr, thresholds = metrics.roc_curve(
        y_true, y_score, pos_label=None, sample_weight=None, drop_intermediate=True
    )
    auc = metrics.roc_auc_score(
        y_true,
        y_score,
        average="macro",
        sample_weight=None,
        max_fpr=None,
        multi_class="raise",
        labels=None,
    )
    return EvaluationResult(
        fpr=fpr, tpr=tpr, thresholds=thresholds, auc_loss=1 - auc, bedroc=bedroc
    )
