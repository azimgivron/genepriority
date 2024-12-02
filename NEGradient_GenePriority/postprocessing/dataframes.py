"""
Dataframes module
=================

Post-processing producing dataframes.
"""
from typing import Dict, List

import numpy as np
import pandas as pd
from NEGradient_GenePriority.evaluation.evaluation_result import EvaluationResult


def generate_auc_loss_table(result: Dict[int, List[EvaluationResult]]):
    """
    Generates a table summarizing 1-AUC loss averages and standard deviations and saves to file.

    Args:
        result (Dict[int, List[EvaluationResult]]): A dictionary where keys are latent dimensions
            and values are a list of evaluation results, one per fold/split.

    Returns:
        (pd.DataFrame): A dataframe with the 'Averaged 1-AUC Error'.
    """
    auc_loss = [[eval_res.auc_loss for eval_res in result[latent]] for latent in result]
    mean_std_auc = np.hstack(
        (
            np.mean(auc_loss, axis=1).reshape(-1, 1),
            np.std(auc_loss, axis=1).reshape(-1, 1),
        )
    )
    dataframe = pd.DataFrame(
        mean_std_auc,
        columns=["Averaged 1-AUC Error", "Std 1-AUC Error"],
        index=[f"Latent Dim={dim}" for dim in result],
    )
    return dataframe
