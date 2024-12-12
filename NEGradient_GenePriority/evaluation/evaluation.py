"""
Evaluation module
=======================

Defines the `Evaluation` class for storing and managing evaluation metrics 
such as ROC curve data, AUC loss, and BEDROC scores.
"""

from typing import Dict, List, Tuple

import numpy as np
from NEGradient_GenePriority.evaluation.metrics import bedroc_score
from NEGradient_GenePriority.evaluation.results import Results
from sklearn import metrics


class Evaluation:
    """
    Represents the evaluation metrics for model predictions.

    Attributes:
        alphas (List[float]): Alpha values for computing BEDROC scores.
        alpha_map (Dict[float, str]): Mapping of alpha values to descriptive labels.
        avg_results (Results): Calculate the average of `y_pred` across all results.
        results (List[Results]): List of results, each corresponding to the results
            of a fold/split.
    """

    alphas: List[float]
    alpha_map: Dict[float, str]
    avg_results: Results
    result: List[Results]

    def __init__(self, results: List[Results]):
        """
        Initializes the Evaluation with model results and parameters.

        Args:
            results (List[Results]): List of results, each corresponding to the
                results of a fold/split.

        Raises:
            TypeError: If any element in `results` is not an instance of `Results`.
        """
        for i, result in enumerate(results):
            if not isinstance(result, Results):
                raise TypeError(
                    f"Invalid type at index {i}: Expected `Results`, but got {type(result)}. "
                    "Ensure all elements in the `results` list are instances of the "
                    "`Results` class."
                )
        self.results = results
        self.avg_results = Results(
            y_true=self.results[0].y_true,
            y_pred=sum(result.y_pred for result in self.results) / len(self.results),
        )

    def compute_bedroc_scores(self) -> np.ndarray:
        """
        Computes BEDROC (Boltzmann-Enhanced Discrimination of Receiver Operating Characteristic)
        scores for the given alpha values.

        Returns:
            np.ndarray: A 2D array of BEDROC scores with shape `(disease, alphas)`,
            where each row corresponds to a disease and each column corresponds
            to a specific alpha value.
        """
        bedroc = []
        for disease_result in self.avg_results:
            y_true, y_pred = disease_result
            bedroc_per_disease = [
                bedroc_score(y_true, y_pred, decreasing=True, alpha=alpha)
                for alpha in self.alphas
            ]
            bedroc.append(bedroc_per_disease)
        bedroc = np.array(bedroc)  # shape=(disease, alpha)
        return bedroc

    def compute_avg_auc_loss(self) -> float:
        """
        Computes the average AUC loss, which is defined as `1 - AUC`
        for each disease, indicating the model's inability to achieve perfect separation.

        Returns:
            np.ndarray: A 1D array where each element represents the AUC loss
            for a specific disease.
        """
        auc_loss = []
        for disease_result in self.avg_results:
            y_true, y_pred = disease_result
            auc_loss.append(1 - metrics.roc_auc_score(y_true, y_pred))
        auc_loss = np.array(auc_loss)  # shape (disease)
        return auc_loss

    def compute_roc_curve(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the Receiver Operating Characteristic (ROC) curve metrics,
        including False Positive Rates (FPR) and True Positive Rates (TPR),
        for each disease.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - np.ndarray: A 2D array where each row contains the FPR for a specific disease.
                - np.ndarray: A 2D array where each row contains the TPR for a specific disease.
        """
        fpr = []
        tpr = []
        for disease_result in self.avg_results:
            y_true, y_pred = disease_result
            fpr_per_disease, tpr_per_disease, _ = metrics.roc_curve(
                y_true, y_pred, pos_label=1, drop_intermediate=True
            )
            fpr.append(fpr_per_disease)
            tpr.append(tpr_per_disease)
        fpr = np.array(fpr)
        tpr = np.array(tpr)
        return fpr, tpr
