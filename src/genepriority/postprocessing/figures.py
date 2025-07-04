# pylint: disable=R0913
"""
Figures module
==============

This module provides post-processing functions to generate and save visualizations 
of evaluation metrics such as ROC curves and BEDROC boxplots. These visualizations 
help in comparing model performance across different configurations, splits, and metrics.

"""
from typing import List, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from genepriority.evaluation.evaluation import Evaluation


def plot_bedroc_boxplots(
    bedroc: np.ndarray,
    model_names: List[str],
    output_file: str,
    figsize: Tuple[int, int],
    sharey: bool,
):
    """
    Plots boxplots of BEDROC scores for multiple alpha values and latent dimensions
    without plotting outliers, with a shared y-axis and a single legend on the side.

    Args:
        bedroc (np.ndarray): BEDROC scores array of shape (alphas, diseases, models).
            Each entry represents the BEDROC score for a specific alpha value, disease,
            and model.
        model_names (List[str]): Names of the models being compared.
        output_file (str): File path where the BEDROC boxplot figure will be saved.
        figsize (Tuple[int, int]): Figure size in inches (width, height).
        sharey (bool): Whether to share the y axis.

    """
    # Okabe-Ito color palette
    colors = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#F0E442"]

    if bedroc.shape[-1] > len(colors):
        raise ValueError("Not enough colors.")

    # Number of subplots = number of alpha values
    n_alphas = len(Evaluation.alphas)

    fig, axs = plt.subplots(1, n_alphas, figsize=figsize, sharey=sharey)

    # If there's only one alpha, axs might not be a list; make it iterable
    if n_alphas == 1:
        axs = [axs]
    # Plot each alpha in its own subplot
    for i, alpha in enumerate(Evaluation.alphas):
        # Create the boxplot for this alpha
        box = sns.boxplot(
            data=bedroc[i],
            ax=axs[i],
            palette=colors[: bedroc.shape[-1]],
            showfliers=False,  # Do not plot outliers
        )
        # Set x-axis ticks to model names
        axs[i].set_xticks(range(bedroc.shape[2]))  # bedroc.shape[2] = number of models
        axs[i].set_xticklabels(["" for _ in model_names])
        axs[i].yaxis.set_tick_params(labelsize=14)

        # Title showing alpha and top %
        axs[i].set_title(
            f"$\\alpha={float(alpha):.1f}$\nTop {Evaluation.alpha_map[alpha]}",
            fontsize=16,
            weight="bold",
        )
        axs[i].grid(axis="y", alpha=0.3)

        # Remove any automatic legend from this subplot (we'll add one big legend later)
        if box.legend_ is not None:
            box.legend_.remove()

    # Adjust the spacing so we have room on the right for the legend
    fig.subplots_adjust(right=1 - (bedroc.shape[-1] * 0.05), wspace=0.3)

    # Create a custom legend on the side
    # Each model gets one color, so we make patches for each color-model pair
    handles = [mpatches.Patch(color=c, label=m) for c, m in zip(colors, model_names)]
    fig.legend(
        handles,
        model_names,
        loc="center right",
        bbox_to_anchor=(0.98, 0.5),  # adjust as needed
        fontsize=18,
    )

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def plot_auc_boxplots(
    auc: np.ndarray,
    model_names: List[str],
    output_file: str,
    figsize: Tuple[int, int],
):
    """
    Plots boxplots of AUC scores for latent dimensions
    without plotting outliers and a single legend on the side.

    Args:
        auc (np.ndarray): A 2D array containing the AUC for
            each model and disease. Shape: (diseases, models).
        model_names (List[str]): Names of the models being compared.
        output_file (str): File path where the BEDROC boxplot figure will be saved.
        figsize (Tuple[int, int]): Figure size in inches (width, height).

    """
    # Okabe-Ito color palette
    colors = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#F0E442"]

    if auc.shape[-1] > len(colors):
        raise ValueError("Not enough colors.")

    fig, axis = plt.subplots(1, 1, figsize=figsize)
    _ = sns.boxplot(
        data=auc,
        ax=axis,
        palette=colors[: auc.shape[-1]],
        showfliers=False,  # Do not plot outliers
    )
    # Set x-axis ticks to model names
    axis.set_xticks(range(auc.shape[1]))
    axis.set_xticklabels(["" for _ in model_names])
    axis.yaxis.set_tick_params(labelsize=14)
    axis.grid(axis="y", alpha=0.3)
    axis.set_title(
        "AUC",
        fontsize=16,
        weight="bold",
    )
    # Adjust the spacing so we have room on the right for the legend
    fig.subplots_adjust(right=1 - (auc.shape[-1] * 0.1))

    # Create a custom legend on the side
    # Each model gets one color, so we make patches for each color-model pair
    handles = [mpatches.Patch(color=c, label=m) for c, m in zip(colors, model_names)]
    fig.legend(
        handles,
        model_names,
        loc="center right",
        bbox_to_anchor=(0.98, 0.5),  # adjust as needed
        fontsize=18,
    )

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
