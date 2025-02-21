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
from genepriority.postprocessing.model_evaluation_collection import (
    ModelEvaluationCollection,
)


def plot_roc_curves(
    evaluation_collection: ModelEvaluationCollection,
    output_file: str,
    figsize: Tuple[int, int],
):
    """
    Plots average ROC curves for multiple models and saves the plot to a file.

    Args:
        evaluation_collection (ModelEvaluationCollection): A collection of `Evaluation`
            objects, each containing the evaluation of a model.
        output_file (str): File path where the ROC curve plot will be saved.
        figsize (Tuple[int, int]): Figure size in inches (width, height).

    """
    colors = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#F0E442"]
    linestyles = [
        "solid",
        "dotted",
        "dashed",
        "dashdot",
        (0, (1, 1)),
        (0, (3, 10, 1, 10, 1, 10)),
    ]

    if len(evaluation_collection) > len(colors):
        raise ValueError("Not enough colors.")

    fig = plt.figure(figsize=figsize)
    for i, (name, evaluation) in enumerate(evaluation_collection.items()):
        fpr_tpr_avg = evaluation.compute_roc_curve()
        fpr, tpr = fpr_tpr_avg
        plt.plot(
            fpr,
            tpr,
            linewidth=4,
            c=colors[i],
            label=name,
            linestyle=linestyles[i],
        )
    plt.plot(
        [0, 1], [0, 1], linestyle=(0, (1, 10)), color="black", label="Random Guess"
    )
    plt.yticks(fontsize=14)
    plt.xlabel("Average FPR", fontsize=16)
    plt.ylabel("Average TPR", fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.3, wspace=0.4, top=0.9)
    plt.savefig(output_file, dpi=300)
    plt.close()


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
        bedroc (np.ndarray): BEDROC scores array of shape (alphas, folds, models).
            Each entry represents the BEDROC score for a specific alpha value, fold,
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
    fig.subplots_adjust(right=0.8, wspace=0.3)

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
