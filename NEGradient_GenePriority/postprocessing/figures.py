# pylint: disable=R0913
"""
Figures module
==============

This module provides post-processing functions to generate and save visualizations 
of evaluation metrics such as ROC curves and BEDROC boxplots. These visualizations 
help in comparing model performance across different configurations, splits, and metrics.

"""
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from NEGradient_GenePriority.evaluation.evaluation_result import EvaluationResult


def plot_roc_curves(
    eval_res_per_model: List[EvaluationResult],
    model_names: List[str],
    output_file: str,
    cmap: str = "Set2",
    figsize=(8, 6),
):
    """
    Plots average ROC curves for multiple models and saves the plot to a file.

    Args:
        eval_res_per_model (List[EvaluationResult]): A list of `EvaluationResult` objects,
            each containing the evaluation metrics for a specific model.
        model_names (List[str]): Names of the models being compared.
        output_file (str): File path where the ROC curve plot will be saved.
        cmap (str, optional): Matplotlib colormap to use for line colors. Defaults to "Set2".
        figsize (tuple, optional): Figure size in inches (width, height). Defaults to (8, 6).

    """
    colors = plt.get_cmap(cmap).colors
    fig = plt.figure(figsize=figsize)
    for i, eval_res in enumerate(eval_res_per_model):
        plt.plot(
            *eval_res.compute_roc_curve(),
            linewidth=2,
            c=colors[i],
            label=model_names[i],
        )
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random Guess")
    plt.xlabel("Average FPR")
    plt.ylabel("Average TPR")
    plt.legend()
    plt.grid(alpha=0.3)
    fig.suptitle("Average ROC Curve")
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.3, wspace=0.4, top=0.85)
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_bedroc_boxplots(
    bedroc: np.ndarray,
    model_names: List[str],
    output_file: str,
    subplots_config: Tuple[int, int],
    cmap: str = "Set2",
    figsize: Tuple[int, int] = (10, 8),
):
    """
    Plots boxplots of BEDROC scores for multiple alpha values and latent dimensions.

    Args:
        bedroc (np.ndarray): BEDROC scores array of shape (alphas, folds, latent).
            Each entry represents the BEDROC score for a specific alpha value, fold,
            and latent dimension.
        model_names (List[str]): Names of the models being compared.
        output_file (str): File path where the BEDROC boxplot figure will be saved.
        subplots_config (Tuple[int, int]): Configuration for the subplots grid (rows, columns).
        cmap (str, optional): Matplotlib colormap for the boxplot colors. Defaults to "Set2".
        figsize (Tuple[int, int], optional): Figure size in inches (width, height).
            Defaults to (10, 8).

    """
    fig, axs = plt.subplots(*subplots_config, figsize=figsize)
    axs = axs.flatten()
    for i, alpha in enumerate(EvaluationResult.alphas):
        sns.boxplot(data=bedroc[i], ax=axs[i], palette=cmap)
        axs[i].set_xticks(range(bedroc.shape[2]))  # Set fixed positions for ticks
        axs[i].set_xticklabels(model_names, rotation=0, ha="center", fontsize=10)
        axs[i].set_title(
            f"{float(alpha):.1f}\nTop {EvaluationResult.alpha_map[alpha]}", fontsize=12
        )
        axs[i].grid(axis="y", alpha=0.3)

    # Disable unused subplots
    if len(EvaluationResult.alphas) % 2 == 1:
        axs[-1].axis("off")
    fig.subplots_adjust(hspace=0.3, wspace=0.4, top=0.85)
    fig.suptitle("Averaged BEDROC", fontsize=16)
    plt.savefig(output_file, dpi=300)
    plt.close()
