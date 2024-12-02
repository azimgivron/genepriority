# pylint: disable=R0913
"""Post-processing producing figures"""
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from NEGradient_GenePriority.evaluation.evaluation_result import EvaluationResult

def plot_roc_curves(
    result: Dict[int, List[EvaluationResult]], output_file: str, cmap: str = "Set2", figsize=(8, 6)
):
    """
    Plots ROC curves for each latent dimension and split and saves to file.

    Args:
        result (Dict[int, List[EvaluationResult]]): A dictionary where keys are latent dimensions
            and values are a list of evaluation results, one per fold/split.
        output_file (str): Path to save the ROC curve plot.
        cmap (str, optional): Matplotlib colormap to use for the plot. Defaults to "Set2".
        figsize (tuple, optional): Size of the figure. Defaults to (8, 6).
    """
    plt.figure(figsize=figsize)
    colors = plt.get_cmap(cmap).colors

    for latent in result:
        for i, eval_res in enumerate(result[latent]):
            plt.plot(
                eval_res.fpr,
                eval_res.tpr,
                linewidth=2,
                c=colors[i],
                label=f"Split {i+1}",
            )

    plt.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random Guess")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_bedroc_boxplots(
    result: Dict[int, List[EvaluationResult]],
    alphas: List[float],
    alpha_map: Dict[float, str],
    output_file: str,
    cmap: str="Set2",
    figsize: Tuple[int, int]=(10, 8),
):
    """
    Plots boxplots of BEDROC scores for various alpha values and latent
    dimensions and saves to file.

    Args:
        result (Dict[int, List[EvaluationResult]]): A dictionary where keys are latent dimensions
            and values are a list of evaluation results, one per fold/split.
        alphas (list): List of alpha values for BEDROC computation.
        alpha_map (dict): Mapping of alpha values to descriptive strings
            (e.g., percentage labels).
        output_file (str): Path to save the BEDROC boxplots.
        cmap (str, optional): Matplotlib colormap for the boxplots. Defaults to "Set2".
        figsize (Tuple[int, int], optional): Size of the figure. Defaults to (10, 8).
    """
    bedroc = np.array(
        [[eval_res.bedroc for eval_res in result[latent]] for latent in result]
    )  # latent, fold, alpha
    bedroc = np.swapaxes(bedroc, 1, 2)  # latent, alpha, fold
    bedroc = np.swapaxes(bedroc, 0, 1)  # alpha, latent, fold

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    num_latent = bedroc.shape[1]

    for i, alpha in enumerate(alphas):
        axis = axes[i]
        data = bedroc[
            i
        ].T  # Transpose to align with boxplot expectations (methods on x-axis)
        sns.boxplot(data=[data[:, j] for j in range(num_latent)], ax=axis, palette=cmap)
        axis.set_xticks(range(num_latent))  # Set fixed positions for ticks
        axis.set_xticklabels(
            [f"{dim} latent dim." for dim in result.keys()],
            rotation=0,
            ha="center",
            fontsize=10,
        )
        axis.set_title(f"{alpha}\nTop {alpha_map[alpha]}", fontsize=12)
        axis.grid(axis="y", alpha=0.3)

    axis[-1].axis("off")

    # Increase padding between rows and columns
    fig.subplots_adjust(hspace=0.3, wspace=0.4, top=0.85)
    fig.suptitle("Averaged BEDROC", fontsize=16)
    plt.savefig(output_file, dpi=300)
    plt.close()
