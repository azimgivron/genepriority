"""
Dataframes module
=================

Post-processing producing dataframes for summarizing evaluation metrics and results.
"""
from typing import List, Tuple

import pandas as pd


def generate_auc_loss_table(
    auc_loss: List[Tuple[float, float]],
    model_names: List[str],
    avg_auc_loss_name: str = "Averaged 1-AUC error",
    std_auc_loss_name: str = "Std 1-AUC error",
) -> pd.DataFrame:
    """
    Generates a table summarizing AUC loss averages and standard
    deviations for each model.

    Args:
        auc_loss (List[Tuple[float, float]]): List of mean and standard
            deviation of the AUC loss, for each model.
        model_names (List[str]): Names of the models corresponding to the AUC losses.
        avg_auc_loss_name (str, optional): Column name for averaged 1-AUC error.
            Defaults to "Averaged 1-AUC error".
        std_auc_loss_name (str, optional): Column name for standard deviation of 1-AUC error.
            Defaults to "Std 1-AUC error".

    Returns:
        pd.DataFrame: A dataframe summarizing AUC loss metrics.
    """
    dataframe = pd.DataFrame(
        auc_loss, columns=[avg_auc_loss_name, std_auc_loss_name], index=model_names
    )
    dataframe[avg_auc_loss_name] = dataframe[avg_auc_loss_name].map(
        lambda x: f"{x:.2e}"
    )
    dataframe[std_auc_loss_name] = dataframe[std_auc_loss_name].map(
        lambda x: f"{x:.2e}"
    )
    return dataframe
