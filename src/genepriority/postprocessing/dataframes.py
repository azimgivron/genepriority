"""
Dataframes module
=================

Post-processing producing dataframes for summarizing evaluation metrics and results.
"""

from typing import Dict, List

import numpy as np
import pandas as pd


def generate_table(
    metric: np.ndarray,
    model_names: List[str],
    avg_name: str = "Averaged",
    std_name: str = "Std",
) -> pd.DataFrame:
    """
    Generates a table summarizing metric averages and standard
    deviations for each model.

    Args:
        metric (np.ndarray): A 2D array containing the metric for
            each model and fold. Shape: (fold, models).
        model_names (List[str]): Names of the models corresponding to the metric.
        avg_name (str, optional): Column name for averaged metric.
            Defaults to "Averaged".
        std_name (str, optional): Column name for standard deviation of metric.
            Defaults to "Std".

    Returns:
        pd.DataFrame: A dataframe summarizing AUC metrics.
    """
    metric = np.hstack((np.mean(metric, axis=0), np.std(metric, axis=0)))
    metric = metric.reshape((2, -1)).T
    dataframe = pd.DataFrame(
        metric, columns=[avg_name, std_name], index=model_names
    ).map(lambda x: f"{x:.2e}")
    return dataframe


def generate_bedroc_table(
    bedroc: np.ndarray,
    model_names: List[str],
    alpha_map: Dict[float, str],
    avg_bedroc_score_name: str = "Averaged BEDROC score",
    std_bedroc_score_name: str = "Std BEDROC score",
) -> pd.DataFrame:
    """
    Generates a table summarizing the averaged BEDROC scores and their
    standard deviations for each model across specified alpha values.

    Args:
        bedroc (np.ndarray): A 3D array containing BEDROC scores for each model,
            each fold and different alpha values.
            Shape: (alphas, fold, models).
        model_names (List[str]): Names of the models corresponding to the BEDROC scores.
        alpha_map (Dict[float, str]): A mapping of alpha values (e.g., 0.2, 0.5) to
            descriptive strings used for table column naming.
        avg_bedroc_score_name (str, optional): Column name prefix for averaged BEDROC scores.
            Defaults to "Averaged BEDROC score".
        std_bedroc_score_name (str, optional): Column name prefix for standard deviation
            of BEDROC scores. Defaults to "Std BEDROC score".

    Returns:
        pd.DataFrame: A DataFrame summarizing the averaged BEDROC scores and their
        standard deviations for each model across different alpha values. Columns
        are dynamically generated based on the alpha values.
    """
    if len(alpha_map) != bedroc.shape[0]:
        raise ValueError

    mean = bedroc.mean(axis=1).T  # shape = (model, alphas)
    std = bedroc.std(axis=1).T  # shape = (model, alphas)
    bedroc = np.hstack((mean, std))  # shape = (model, 2*alphas)

    column_names = np.array(
        [f"{avg_bedroc_score_name} (top {alpha})" for alpha in alpha_map.values()]
        + [f"{std_bedroc_score_name} (top {alpha})" for alpha in alpha_map.values()]
    )

    dataframe = pd.DataFrame(bedroc, columns=column_names, index=model_names).map(
        lambda x: f"{x:.2e}"
    )

    return dataframe
