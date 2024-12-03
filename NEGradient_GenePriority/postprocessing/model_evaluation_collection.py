"""
ModelEvaluationCollection module
===============================

This module provides the `ModelEvaluationCollection` class, which encapsulates the
evaluation results of multiple models. It offers utility methods to compute and
aggregate evaluation metrics such as AUC loss and BEDROC scores across different models.
This class facilitates comparison and analysis of model performance in a structured
and efficient manner.

"""
from typing import Dict, Iterator, List

import numpy as np

from NEGradient_GenePriority.evaluation.evaluation import Evaluation


class ModelEvaluationCollection:
    """
    A collection of evaluation results for multiple models, with methods to compute
    and access aggregated metrics such as AUC loss and BEDROC scores.

    Attributes:
        model_results (Dict[str, Evaluation]): A dictionary where keys are model names
            and values are Evaluation objects containing metrics for the respective model.
    """

    def __init__(self, model_results: Dict[str, Evaluation]):
        """
        Initializes the ModelEvaluationCollection.

        Args:
            model_results (Dict[str, Evaluation]): A dictionary mapping model names
                to their respective Evaluation objects.
        """
        self.model_results: Dict[str, Evaluation] = model_results

    @property
    def model_names(self) -> List[str]:
        """
        Retrieves the names of the models in the collection.

        Returns:
            List[str]: A list of model names.
        """
        return list(self.model_results.keys())

    @property
    def evaluations(self) -> List[Evaluation]:
        """
        Retrieves the list of Evaluation objects in the collection.

        Returns:
            List[Evaluation]: A list of Evaluation objects corresponding to the models.
        """
        return list(self.model_results.values())

    def items(self) -> Iterator[tuple]:
        """
        Provides an iterator over the model names and their corresponding Evaluation objects.

        Returns:
            Iterator[tuple]: An iterator of (model_name, Evaluation) pairs.
        """
        return self.model_results.items()

    def __iter__(self) -> Iterator[Evaluation]:
        """
        Allows the `ModelEvaluationCollection` object to be iterable over its Evaluation objects.

        Returns:
            Iterator[Evaluation]: An iterator over the Evaluation objects.
        """
        return iter(self.evaluations)

    def compute_auc_losses(self) -> List[tuple]:
        """
        Computes the average AUC loss (1 - AUC) for each model.

        Returns:
            List[tuple]: A list of tuples, where each tuple contains the mean and
            standard deviation of AUC loss for a model.
        """
        auc_losses = []
        for eval_res in self.evaluations:
            auc_losses.append(eval_res.compute_avg_auc_loss())
        return auc_losses

    def compute_bedroc_scores(self) -> np.ndarray:
        """
        Computes BEDROC scores for each model and rearranges the results for analysis.

        Returns:
            np.ndarray: A NumPy array of shape (alphas, folds, latent), where:
                - alphas: BEDROC scores for different alpha values.
                - folds: BEDROC scores for different cross-validation folds.
                - latent: BEDROC scores for different latent dimensions.
        """
        bedroc_scores = np.array(
            [eval_res.compute_bedroc_scores() for eval_res in self.evaluations]
        )  # Shape: (latent, folds, alphas)
        return np.transpose(
            bedroc_scores, axes=(2, 1, 0)
        )  # Shape: (alphas, folds, latent)
