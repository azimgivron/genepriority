from dataclasses import dataclass
from typing import List

import scipy.sparse as sp
import smurff
from metrics import bedroc_score
from sklearn import metrics

from NEGradient_GenePriority.preprocessing import Indices


@dataclass
class EvaluationResult:
    """
    Stores the results of evaluation metrics for model predictions.

    Attributes:
        fpr (List[float]): False positive rates at various thresholds from the ROC curve.
        tpr (List[float]): True positive rates at various thresholds from the ROC curve.
        thresholds (List[float]): Threshold values used to compute FPR and TPR.
        auc_loss (float): 1 - AUC score (represents loss based on AUC).
        bedroc (List[float]): BEDROC scores for the specified alpha values.
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
    Evaluates a classifier's performance using various metrics, including ROC and BEDROC.

    Args:
        y_true (List[int]): Ground-truth binary labels (1 for positive, 0 for negative).
        y_score (List[float]): Predicted scores or probabilities.
        alphas (List[float]): Alpha values to compute BEDROC scores (higher emphasizes early recognition).

    Returns:
        EvaluationResult: A dataclass instance containing evaluation metrics:
                          - FPR, TPR, thresholds from the ROC curve.
                          - AUC loss (1 - AUC score).
                          - BEDROC scores for the provided alpha values.
    """
    bedroc = [
        bedroc_score(y_true, y_score, decreasing=True, alpha=alpha) for alpha in alphas
    ]
    fpr, tpr, thresholds = metrics.roc_curve(
        y_true, y_score, pos_label=1, drop_intermediate=True
    )
    auc = metrics.roc_auc_score(y_true, y_score)
    return EvaluationResult(
        fpr=fpr, tpr=tpr, thresholds=thresholds, auc_loss=1 - auc, bedroc=bedroc
    )


def train_and_test_splits(
    sparse_matrix: sp.coo_matrix,
    splits_list: List[Indices],
    num_samples: int,
    burnin_period: int,
    num_latent: int,
    alphas: List[float]
) -> List[EvaluationResult]:
    """
    Trains and evaluates BPMF on dataset splits using specified evaluation metrics.

    Args:
        sparse_matrix (sp.coo_matrix): Sparse matrix representation of the dataset.
        splits_list (List[Indices]): List of train-test splits (Indices objects).
        num_samples (int): Number of samples for the BPMF model.
        burnin_period (int): Burn-in period for the BPMF model.
        num_latent (int): Number of latent dimensions for the BPMF model.
        alphas (List[float]): Alpha values for computing BEDROC scores.

    Returns:
        List[EvaluationResult]: List of evaluation results for each split, including:
                                - ROC metrics (FPR, TPR, thresholds, AUC loss).
                                - BEDROC scores for each alpha value.
    """
    results = []
    for split in splits_list:
        session = smurff.BPMFSession(
            Ytrain=split.get_training_data(sparse_matrix),
            Ytest=split.get_testing_data(sparse_matrix),
            is_scarce=False,
            direct=True,
            univariate=True,
            num_latent=num_latent,
            burnin=burnin_period,
            nsamples=num_samples,
        )
        session.train()
        session.test()
        y_true, y_score = session.get_test_predictions()
        evaluation_result = evaluate(y_true, y_score, alphas)
        results.append(evaluation_result)
    return results


def train_and_test_folds(
    sparse_matrix: sp.coo_matrix,
    folds_list: List[Indices],
    num_samples: int,
    burnin_period: int,
    num_latent: int,
    alphas: List[float]
) -> List[EvaluationResult]:
    """
    Trains and evaluates BPMF on dataset folds using specified evaluation metrics.

    Args:
        sparse_matrix (sp.coo_matrix): Sparse matrix representation of the dataset.
        folds_list (List[Indices]): List of train-test folds (Indices objects).
        num_samples (int): Number of samples for the BPMF model.
        burnin_period (int): Burn-in period for the BPMF model.
        num_latent (int): Number of latent dimensions for the BPMF model.
        alphas (List[float]): Alpha values for computing BEDROC scores.

    Returns:
        List[EvaluationResult]: List of evaluation results for each fold, including:
                                - ROC metrics (FPR, TPR, thresholds, AUC loss).
                                - BEDROC scores for each alpha value.
    """
    results = []
    for fold in folds_list:
        session = smurff.BPMFSession(
            Ytrain=fold.get_training_data(sparse_matrix),
            Ytest=fold.get_testing_data(sparse_matrix),
            is_scarce=False,
            direct=True,
            univariate=True,
            num_latent=num_latent,
            burnin=burnin_period,
            nsamples=num_samples,
        )
        session.train()
        session.test()
        y_true, y_score = session.get_test_predictions()
        evaluation_result = evaluate(y_true, y_score, alphas)
        results.append(evaluation_result)
    return results
