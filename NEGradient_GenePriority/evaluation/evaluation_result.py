"""
EvaluationResult module
=======================

Defines the `EvaluationResult` class for storing and managing evaluation metrics 
such as ROC curve data, AUC loss, and BEDROC scores.
"""

from typing import Dict, List, Tuple
from sklearn import metrics
from NEGradient_GenePriority.evaluation.metrics import bedroc_score
from NEGradient_GenePriority.evaluation.evaluation import Results


class EvaluationResult:
    """
    Represents the evaluation metrics for model predictions.

    Attributes:
        result (Results): Contains ground truth (`y_true`) and predictions (`y_pred`).
        alphas (List[float]): Alpha values for computing BEDROC scores.
        alpha_map (Dict[float, str]): Mapping of alpha values to descriptive labels.
    """

    def __init__(self, result: Results, alphas: List[float], alpha_map: Dict[float, str]):
        """
        Initializes the EvaluationResult with model results and parameters.

        Args:
            result (Results): Contains ground truth (`y_true`) and predictions (`y_pred`).
            alphas (List[float]): Alpha values for computing BEDROC scores.
            alpha_map (Dict[float, str]): Mapping of alpha values to descriptive labels.

        Raises:
            ValueError: If any alpha value in `alphas` is missing from `alpha_map`.
        """
        self.result = result

        # Validate that all alphas are present in alpha_map
        for alpha in alphas:
            if alpha not in alpha_map:
                raise ValueError(
                    "Missing mapping in `alpha_map`. "
                    "All alpha values must have a mapping in `alpha_map`"
                )
        self.alphas = alphas
        self.alpha_map = alpha_map

    def compute_bedroc_scores(self) -> Tuple[List[float], List[str], List[float]]:
        """
        Computes BEDROC scores for the given alpha values.

        Returns:
            Tuple[List[float], List[str], List[float]]:
                - Alpha values.
                - Corresponding labels from `alpha_map`.
                - Computed BEDROC scores.
        """
        scores = [
            bedroc_score(*self.result, decreasing=True, alpha=alpha)
            for alpha in self.alphas
        ]
        mappings = [self.alpha_map[alpha] for alpha in self.alphas]
        return self.alphas, mappings, scores

    def compute_auc_loss(self) -> float:
        """
        Computes the AUC loss (1 - AUC).

        Returns:
            float: The computed AUC loss.
        """
        auc_loss = 1 - metrics.roc_auc_score(*self.result)
        return auc_loss

    def compute_roc_curve(self) -> Tuple[List[float], List[float], List[float]]:
        """
        Computes the ROC curve metrics (FPR, TPR, and thresholds).

        Returns:
            Tuple[List[float], List[float], List[float]]:
                - False Positive Rates (FPR).
                - True Positive Rates (TPR).
                - Threshold values.
        """
        fpr, tpr, thresholds = metrics.roc_curve(
            *self.result, pos_label=1, drop_intermediate=True
        )
        return fpr, tpr, thresholds
