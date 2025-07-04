"""
Evaluation module
=======================

Defines the `Evaluation` class for storing and managing evaluation metrics 
such as ROC curve data, AUC loss, and BEDROC scores.
"""
from typing import Dict, List

import numpy as np

from genepriority.evaluation.metrics import auc_scores, bedroc_scores
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

    def compute_bedroc_scores(self) -> np.ndarray:
        """
        Computes BEDROC (Boltzmann-Enhanced Discrimination of Receiver Operating Characteristic)
        scores for the given alpha values.

        Returns:
            np.ndarray: A 2D array of BEDROC scores with shape `(alphas, diseases)`.
        """
        bedroc = []
        masks = []
        for fold_res in self.results:
            y_true = fold_res.y_true
            y_pred = fold_res.y_pred
            bedroc.append([])
            masks.append([])
            for alpha in self.alphas:
                bedroc_per_fold, mask_per_fold = bedroc_scores(
                    y_true=y_true,
                    y_pred=y_pred,
                    gene_number=fold_res.gene_number,
                    alpha=alpha,
                )
                bedroc[-1].append(bedroc_per_fold)
                masks[-1].append(mask_per_fold)

        mask = np.stack(masks).astype(bool)
        bedroc = np.stack(bedroc).astype(np.float64)  # shape=(fold, alphas, diseases)

        valid = mask.any(axis=(0, 1))
        mask = mask[:, :, valid]
        bedroc = bedroc[:, :, valid]

        bedroc_masked = np.ma.array(bedroc, mask=~mask)
        bedroc = bedroc_masked.mean(axis=0).data
        return bedroc

    def compute_avg_auc(self) -> float:
        """
        Computes the average AUC for each fold, indicating the model's
        ability to achieve perfect separation.

        Returns:
            np.ndarray: A 1D array where each element represents the AUC
                for a disease.
        """
        auc = []
        masks = []
        for fold_res in self.results:
            y_true = fold_res.y_true
            y_pred = fold_res.y_pred
            auc_per_fold, mask_per_fold = auc_scores(
                y_true=y_true, y_pred=y_pred, gene_number=fold_res.gene_number
            )
            masks.append(mask_per_fold)
            auc.append(auc_per_fold)

        mask = np.stack(masks).astype(bool)
        auc = np.stack(auc).astype(np.float64)  # shape=(fold, diseases)

        valid = mask.any(axis=0)
        mask = mask[:, valid]
        auc = auc[:, valid]

        auc_masked = np.ma.array(auc, mask=~mask)
        auc = auc_masked.mean(axis=0).data
        return auc

    def compute_avg_roc_curve(self) -> float:
        """
        Computes the average ROC curve across folds.

        Returns:
            np.ndarray: A 1D array where each element represents the AUC loss
            for a disease.
        """
        roc = []
        masks = []
        for fold_res in self.results:
            y_true = fold_res.y_true
            y_pred = fold_res.y_pred
            auc_per_fold, mask_per_fold = roc_curves(
                y_true=y_true, y_pred=y_pred, gene_number=fold_res.gene_number
            )
            masks.append(mask_per_fold)
            roc.append(auc_per_fold)

        mask = np.stack(masks).astype(bool)
        roc = np.stack(roc).astype(np.float64)  # shape=(fold, diseases)

        valid = mask.any(axis=0)
        mask = mask[:, valid]
        roc = roc[:, valid]

        roc_masked = np.ma.array(roc, mask=~mask)
        roc = roc_masked.mean(axis=0).data
        return roc
