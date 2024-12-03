# pylint: disable=R0913,R0914
"""
Evaluation module
=================

This module orchestrates model training and evaluation across train-test splits
or cross-validation folds. It integrates preprocessing and metrics functionalities
to compute and log performance metrics. Key components include the `EvaluationResult`
data class to store metrics and functions like `extract_results` and `train_and_test`
to streamline the evaluation process.
"""
import logging
from dataclasses import dataclass
from typing import List

import numpy as np
import scipy.sparse as sp
import smurff

from NEGradient_GenePriority.preprocessing import Indices, TrainTestIndices


@dataclass
class Results:
    """
    Prediction results data structure.

    Attributes:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values from the trained model.
    """

    y_true: np.ndarray
    y_pred: np.ndarray

    def __iter__(self):
        """
        Makes the `Results` object iterable to allow unpacking with the `*` operator.

        Returns:
            Iterator: An iterator over the `y_true` and `y_pred` arrays.
        """
        return iter([self.y_true, self.y_pred])


def extract_results(
    session: smurff.MacauSession,
    sparse_matrix: sp.coo_matrix,
    testing_indices: Indices,
) -> Results:
    """Extract predictions from the trained model for the specified
    testing indices.

    Args:
        session (smurff.MacauSession): The smurff session.
        sparse_matrix (sp.coo_matrix): The full matrix.
        testing_indices (Indices): The indices in the matrix
            that must be used for testing.

    Returns:
        Results: Contains `y_true` (ground truth) and `y_pred` (predictions).
    """
    y_true = testing_indices.get_data(sparse_matrix).data
    predict_session = session.makePredictSession()
    y_pred_full = np.mean(predict_session.predict_all(), axis=0)
    y_pred = testing_indices.mask(y_pred_full)
    return Results(y_true, y_pred)


def train_and_test(
    sparse_matrix: sp.coo_matrix,
    folds_list: List[TrainTestIndices],
    num_samples: int,
    burnin_period: int,
    direct: bool,
    univariate: bool,
    num_latent: int,
    seed: int,
    save_freq: int,
    output_path: str,
    save_name: str,
    verbose: int,
) -> List[Results]:
    r"""
    Trains and evaluates the model across multiple folds for cross-validation.

    This function performs cross-validation using a provided list of train-test
    folds (`folds_list`). For each fold, the algorithm trains a model, makes predictions,
    and extracts performance metrics.

    Args:
        sparse_matrix (sp.coo_matrix): The sparse matrix representation of the dataset
            to be factored. Rows and columns correspond to features and samples, respectively.
        folds_list (List[TrainTestIndices]): A list of `TrainTestIndices` objects defining
            the train-test folds for cross-validation.
        num_samples (int): The number of posterior samples to draw during training.
        burnin_period (int): The number of burn-in iterations before collecting posterior samples.
        direct (bool): Whether to use a Cholesky instead of conjugate gradient (CG) solver.
            Cholesky is recommanded up to $dim(F_e) \approx 20,000$.
        univariate (bool): Whether to use univariate or multivariate sampling.
            Multivariate sampling require computing the whole precision matrix
            $D \cdot F_e \times D \cdot F_e$ where $D$ is the latent vector size and $F_e$
            is the dimensionality of the entity features. If True, it uses a Gibbs sampler.
        num_latent (int): The number of latent factors to be used by the model.
        seed (int): The random seed to ensure reproducibility in stochastic operations.
        save_freq (int): The frequency at which the model state is saved (e.g., every N samples).
        output_path (str): The path to the directory where the snapshots will be saved.
        save_name (str): The base filename to use when saving model snapshots.
        verbose (int): The verbosity level of the algorithm (0: Silent, 1: Minimal, 2: Detailed).

    Returns:
        List[Results]: List of `Results` objects containing ground truth and predictions.

    """
    results = []
    for i, fold in enumerate(folds_list):
        logger = logging.getLogger(__name__)
        logger.debug("Initiating training on fold %s", i + 1)

        session = smurff.MacauSession(
            Ytrain=fold.training_indices.get_data(sparse_matrix),
            is_scarce=False,
            direct=direct,
            univariate=univariate,
            num_latent=num_latent,
            burnin=burnin_period,
            nsamples=num_samples,
            seed=seed,
            save_freq=save_freq,
            save_name=str(output_path / f"{i}:{save_name}"),
            verbose=verbose,
        )
        session.run()  # run training
        logger.debug("Training on fold %s ended successfully.", i + 1)

        y_true_pred = extract_results(session, sparse_matrix, fold.testing_indices)

        logger.debug("Evaluation on fold %s ended successfully.", i + 1)
        results.append(y_true_pred)
    return results
