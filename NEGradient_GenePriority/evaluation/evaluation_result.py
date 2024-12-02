"""EvaluationResult module"""
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class EvaluationResult:
    """
    Represents the evaluation metrics for model predictions.

    Attributes:
        fpr (List[float]): False positive rates at various thresholds from the ROC curve.
        tpr (List[float]): True positive rates at various thresholds from the ROC curve.
        thresholds (List[float]): Threshold values corresponding to FPR and TPR.
        auc_loss (float): Loss metric defined as 1 - AUC score.
        bedroc (Dict[float, float]): BEDROC scores computed for the provided alpha values.
            Keys are alpha values.
    """

    fpr: List[float]
    tpr: List[float]
    thresholds: List[float]
    auc_loss: float
    bedroc: Dict[float, float]
