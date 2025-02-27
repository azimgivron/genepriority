"""
Evaluation module
=======================

Defines the `Evaluation` class for storing and managing evaluation metrics 
such as ROC curve data, AUC loss, and BEDROC scores.
"""
from typing import Dict, List

import numpy as np
from sklearn import metrics

from genepriority.evaluation.metrics import bedroc_score
from genepriority.evaluation.results import Results


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
    results: List[Results]

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

    def apply_mask(self, value: bool):
        """
        Set the mask application flag for all result objects.

        Args:
            value (bool): Flag indicating whether to apply the mask
                (True) or not (False).
        """
        for res in self.results:
            res.apply_mask = value

    def compute_bedroc_scores(self) -> np.ndarray:
        """
        Computes BEDROC (Boltzmann-Enhanced Discrimination of Receiver Operating Characteristic)
        scores for the given alpha values.

        Returns:
            np.ndarray: A 2D array of BEDROC scores with shape `(fold, alphas)`,
            where each row corresponds to a fold and each column corresponds
            to a specific alpha value.
        """
        bedroc = []
        for fold_res in self.results:
            y_true = fold_res.y_true.flatten()
            y_pred = fold_res.y_pred.flatten()
            bedroc_per_fold = [
                bedroc_score(y_true=y_true, y_pred=y_pred, decreasing=True, alpha=alpha)
                for alpha in self.alphas
            ]
            bedroc.append(bedroc_per_fold)
        bedroc = np.array(bedroc)  # shape=(fold, alphas)
        return bedroc

    def compute_avg_auc_loss(self) -> float:
        """
        Computes the average AUC loss, which is defined as `1 - AUC`
        for each fold, indicating the model's inability to achieve perfect separation.

        Returns:
            np.ndarray: A 1D array where each element represents the AUC loss
            for a fold.
        """
        auc_loss = []
        for fold_res in self.results:
            y_true = fold_res.y_true.flatten()
            y_pred = fold_res.y_pred.flatten()
            auc_loss.append(1 - metrics.roc_auc_score(y_true, y_pred))
        auc_loss = np.array(auc_loss)
        return auc_loss

    def compute_roc_curve(self) -> np.ndarray:
        """
        Computes the Receiver Operating Characteristic (ROC) curve metrics,
        including False Positive Rates (FPR) and True Positive Rates (TPR)
        for all folds.

        Returns:
            np.ndarray: A 2D array with the following structure:
                - Shape: (2, number of thresholds)
                - The first dimension contains FPR values.
                - The second dimension contains TPR values.
        """
        roc_curves = []
        for fold_res in self.results:
            y_true = fold_res.y_true.flatten()
            y_pred = fold_res.y_pred.flatten()
            fpr, tpr, thresholds = metrics.roc_curve(
                y_true, y_pred, pos_label=1, drop_intermediate=True
            )
            roc_curves.append((fpr, tpr, thresholds))

        # Get common thresholds from all curves (sorted in ascending order)
        common_thresholds = np.sort(
            np.unique(np.hstack([curve[-1] for curve in roc_curves]))
        )

        # Interpolate each ROC curve so that they share the same thresholds
        data = []
        for fpr, tpr, thresholds in roc_curves:
            # Reverse arrays because thresholds from roc_curve are in descending order
            sorted_thresholds = thresholds[::-1]
            sorted_fpr = fpr[::-1]
            sorted_tpr = tpr[::-1]
            interp_fpr = np.interp(common_thresholds, sorted_thresholds, sorted_fpr)
            interp_tpr = np.interp(common_thresholds, sorted_thresholds, sorted_tpr)
            data.append((interp_fpr, interp_tpr))
        data = np.array(data)
        return data.mean(axis=0)
