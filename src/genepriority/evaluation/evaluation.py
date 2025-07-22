"""
Evaluation module
=======================

Defines the `Evaluation` class for storing and managing evaluation metrics
such as ROC curve data, AUC loss, and BEDROC scores.
"""

from typing import Callable, Dict, List

import numpy as np

from genepriority.evaluation.metrics import (auc_per_disease,
                                             avg_precision_per_disease,
                                             bedroc_per_disease,
                                             pr_per_disease, roc_per_disease)
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
                bedroc_per_fold, mask_per_fold = bedroc_per_disease(
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
        return self.compute_avg_metric(auc_per_disease)

    def compute_avg_precision(self) -> float:
        """
        Computes the average Precision for each fold.

        Returns:
            np.ndarray: A 1D array where each element represents the Precision
                for a disease.
        """
        return self.compute_avg_metric(avg_precision_per_disease)

    def compute_avg_metric(self, func: Callable) -> float:
        """
        Computes the average metric for each fold.

        Returns:
            np.ndarray: A 1D array where each element represents the metric
                for a disease.
        """
        metric = []
        masks = []
        for fold_res in self.results:
            y_true = fold_res.y_true
            y_pred = fold_res.y_pred
            metric_per_fold, mask_per_fold = func(
                y_true=y_true, y_pred=y_pred, gene_number=fold_res.gene_number
            )
            masks.append(mask_per_fold)
            metric.append(metric_per_fold)

        mask = np.stack(masks).astype(bool)
        metric = np.stack(metric).astype(np.float64)  # shape=(fold, diseases)

        valid = mask.any(axis=0)
        mask = mask[:, valid]
        metric = metric[:, valid]

        metric_masked = np.ma.array(metric, mask=~mask)
        metric = metric_masked.mean(axis=0).data
        return metric

    def compute_avg_roc_curve(self) -> np.ndarray:
        """
        Computes the average ROC curve across folds and diseases.

        Returns:
            np.ndarray: 2D array of shape (2, n_thresholds) where:
                - [0, :] is the mean FPR.
                - [1, :] is the mean TPR.
        """
        return self.compute_avg_metric_curve(roc_per_disease)

    def compute_avg_pr_curve(self) -> np.ndarray:
        """
        Computes the average PR curve across folds and diseases.

        Returns:
            np.ndarray: 2D array of shape (2, n_thresholds) where:
                - [0, :] is the mean Precision.
                - [1, :] is the mean Recall.
        """
        return self.compute_avg_metric_curve(pr_per_disease)

    def compute_avg_metric_curve(self, func: Callable) -> np.ndarray:
        """
        Computes the average metric curve across folds and diseases.

        Returns:
            np.ndarray: 2D array of shape (2, n_thresholds)
        """
        metric_list = []
        threshold_list = []
        cross_fold_thresholds = set()
        for fold_res in self.results:
            y_true = fold_res.y_true
            y_pred = fold_res.y_pred
            metric_per_fold, thresholds = func(
                y_true=y_true, y_pred=y_pred, gene_number=fold_res.gene_number
            )
            cross_fold_thresholds |= set(thresholds)
            # metric_per_fold: np.ndarray of shape (2, n_thresholds)
            metric_list.append(metric_per_fold)
            threshold_list.append(thresholds)

        cross_fold_thresholds = list(cross_fold_thresholds)
        cross_fold_thresholds.sort()
        avg = aggregate(metric_list, threshold_list, cross_fold_thresholds)
        return avg


def aggregate(
    scores: List[np.ndarray],
    thresholds: List[np.ndarray],
    target_thresholds: List[float],
) -> np.ndarray:
    """
    Linearly interpolate metric values across a shared threshold grid
    and average across folds.

    Args:
        scores (List[np.ndarray]): Arrays of shape (2, n_thresholds)
            for each fold.
        thresholds (List[np.ndarray]): The initial thresholds for each fold.
        target_thresholds (List[float]):
            Sorted list of unique thresholds to interpolate across.

    Returns:
        np.ndarray:
            2D array of shape [2, n_thresholds] containing
                interpolated metrics.
    """
    final = []
    for score, current_thresholds in zip(scores, thresholds):
        new_score = np.empty(shape=(2, len(target_thresholds)))
        new_score[0] = np.interp(target_thresholds, current_thresholds, score[0])
        new_score[1] = np.interp(target_thresholds, current_thresholds, score[1])
        final.append(new_score)
    return np.array(final).mean(axis=0)
