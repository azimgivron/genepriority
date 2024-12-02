# pylint: disable=R0913,R0914
"""
This module orchestrates model training and evaluation across train-test splits
or cross-validation folds. It integrates preprocessing and metrics functionalities
to compute and log performance metrics. Key components include the EvaluationResult
data class to store metrics and functions like evaluate_scores, train_and_test_splits,
and train_and_test_folds to streamline the evaluation process.
"""
import logging
from typing import List

import numpy as np
import scipy.sparse as sp
import smurff
from sklearn import metrics

from NEGradient_GenePriority.evaluation.evaluation_result import EvaluationResult
from NEGradient_GenePriority.evaluation.metrics import bedroc_score
from NEGradient_GenePriority.preprocessing import Indices, TrainTestIndices


def evaluate_scores(
    y_true: List[int], y_pred: List[float], alphas: List[float]
) -> EvaluationResult:
    """
    Evaluates a model's predictions using ROC and BEDROC metrics.

    Args:
        y_true (List[int]): Ground truth binary labels (1 for positive, 0 for negative).
        y_pred (List[float]): Model's predicted scores or probabilities.
        alphas (List[float]): Alpha values for calculating BEDROC scores,
            where larger values emphasize early correct predictions.

    Returns:
        EvaluationResult: Contains:
            - FPR, TPR, and thresholds from the ROC curve.
            - AUC loss (1 - AUC score).
            - BEDROC scores for the given alpha values.
    """
    bedroc = {
        f"{alpha:.3f}": bedroc_score(
            y_true,
            y_pred,
            decreasing=True,
            alpha=alpha
        ) for alpha in alphas
    }
    fpr, tpr, thresholds = metrics.roc_curve(
        y_true, y_pred, pos_label=1, drop_intermediate=True
    )
    auc = metrics.roc_auc_score(y_true, y_pred)
    return EvaluationResult(
        fpr=fpr, tpr=tpr, thresholds=thresholds, auc_loss=1 - auc, bedroc=bedroc
    )


def evaluate(
    session: smurff.MacauSession,
    sparse_matrix: sp.coo_matrix,
    testing_indices: Indices,
    alphas: List[float],
) -> EvaluationResult:
    """Evaluate the trained model.

    Args:
        session (smurff.MacauSession): The smurff session.
        sparse_matrix (sp.coo_matrix): The full matrix.
        testing_indices (Indices): The indices in the matrix
            that must be used for testing.
        alphas (List[float]): The alphas for the bedroc metric evaluation.

    Returns:
        EvaluationResult: The results of the evaluation.
    """
    y_true = testing_indices.get_data(sparse_matrix).data
    predict_session = session.makePredictSession()
    y_pred_full = np.mean(predict_session.predict_all(), axis=0)
    y_pred = testing_indices.mask(y_pred_full)
    evaluation_result = evaluate_scores(y_true, y_pred, alphas)
    return evaluation_result


def train_and_test(
    sparse_matrix: sp.coo_matrix,
    folds_list: List[TrainTestIndices],
    num_samples: int,
    burnin_period: int,
    direct: bool,
    univariate: bool,
    num_latent: int,
    alphas: List[float],
    seed: int,
    save_freq: int,
    output_path: str,
    save_name: str,
    verbose: int,
) -> List[EvaluationResult]:
    r"""
    Trains and evaluates the model across multiple folds for cross-validation.

    This function performs cross-validation using a provided list of train-test
    folds (`folds_list`). For each fold, the algorithm is trained, predictions
    are made, and the performance is evaluated using ROC and BEDROC metrics.

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
        alphas (List[float]): A list of alpha values for computing BEDROC scores, where larger
                              alpha emphasizes early recognition of positive cases.
        seed (int): The random seed to ensure reproducibility in stochastic operations.
        save_freq (int): The frequency at which the model state is saved (e.g., every N samples).
        output_path (str): The path to the directory where the snapshots will be saved.
        save_name (str): The base filename to use when saving model snapshots.
        verbose (int): The verbosity level of the algorithm (0: Silent, 1: Minimal, 2: Detailed).

    Returns:
        List[EvaluationResult]: A list of `EvaluationResult` objects, each containing the
                                evaluation metrics for the corresponding fold. Metrics include:
                                - ROC curve metrics (FPR, TPR, thresholds).
                                - AUC loss (1 - AUC score).
                                - BEDROC scores for the specified alpha values.

    Raises:
        ValueError: If the sizes of `y_true` and `y_pred` are mismatched after prediction.
        Exception: Any error encountered during training or evaluation is logged and re-raised.
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

        evaluation_result = evaluate(
            session, sparse_matrix, fold.testing_indices, alphas
        )

        logger.debug("Evaluation on fold %s ended successfully.", i + 1)
        results.append(evaluation_result)
    return results
