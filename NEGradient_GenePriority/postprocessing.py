import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List


def plot_roc_curves(res: Dict[int, List], output_file: str, cmap: str = "Set2", figsize=(8, 6)):
    """
    Plots ROC curves for each latent dimension and split and saves to file.

    Args:
        res (dict): A dictionary where keys are latent dimensions and values are lists of evaluation results.
        output_file (str): Path to save the ROC curve plot.
        cmap (str, optional): Matplotlib colormap to use for the plot. Defaults to "Set2".
        figsize (tuple, optional): Size of the figure. Defaults to (8, 6).
    """
    plt.figure(figsize=figsize)
    colors = plt.get_cmap(cmap).colors

    for latent in res:
        for i, eval_res in enumerate(res[latent]):
            plt.plot(eval_res.fpr, eval_res.tpr, linewidth=2, c=colors[i], label=f"Split {i+1}")

    plt.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random Guess")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(output_file, dpi=300)
    plt.close()


def generate_auc_loss_table(res: Dict[int, List], output_file: str):
    """
    Generates a table summarizing 1-AUC loss averages and standard deviations and saves to file.

    Args:
        res (dict): A dictionary where keys are latent dimensions and values are lists of evaluation results.
        output_file (str): Path to save the AUC loss table as a CSV file.
    """
    auc_loss = [[eval_res.auc_loss for eval_res in res[latent]] for latent in res]
    mean_std_auc = np.hstack((
        np.mean(auc_loss, axis=1).reshape(-1, 1),
        np.std(auc_loss, axis=1).reshape(-1, 1)
    ))
    df = pd.DataFrame(mean_std_auc, columns=["Averaged 1-AUC Error", "Std 1-AUC Error"],
                      index=[f"Latent Dim={dim}" for dim in res])
    df.to_csv(output_file)


def plot_bedroc_boxplots(res: Dict[int, List], alphas: List[float], alpha_map: Dict[float, str],
                         output_file: str, cmap="Set2", figsize=(10, 8)):
    """
    Plots boxplots of BEDROC scores for various alpha values and latent dimensions and saves to file.

    Args:
        res (dict): A dictionary where keys are latent dimensions and values are lists of evaluation results.
        alphas (list): List of alpha values for BEDROC computation.
        alpha_map (dict): Mapping of alpha values to descriptive strings (e.g., percentage labels).
        output_file (str): Path to save the BEDROC boxplots.
        cmap (str, optional): Matplotlib colormap for the boxplots. Defaults to "Set2".
        figsize (tuple, optional): Size of the figure. Defaults to (10, 8).
    """
    bedroc = np.array([[eval_res.bedroc for eval_res in res[latent]] for latent in res])  # latent, fold, alpha
    bedroc = np.swapaxes(bedroc, 1, 2)  # latent, alpha, fold
    bedroc = np.swapaxes(bedroc, 0, 1)  # alpha, latent, fold

    fig, axs = plt.subplots(2, 3, figsize=figsize)
    axs = axs.flatten()
    num_latent = bedroc.shape[1]

    for i, alpha in enumerate(alphas):
        ax = axs[i]
        data = bedroc[i].T  # Transpose to align with boxplot expectations (methods on x-axis)
        sns.boxplot(data=[data[:, j] for j in range(num_latent)], ax=ax, palette=cmap)
        ax.set_xticks(range(num_latent))  # Set fixed positions for ticks
        ax.set_xticklabels([f"{dim} latent dim." for dim in res.keys()], rotation=0, ha="center", fontsize=10)
        ax.set_title(f"{alpha}\nTop {alpha_map[alpha]}", fontsize=12)
        ax.grid(axis="y", alpha=0.3)

    axs[-1].axis("off")

    # Increase padding between rows and columns
    fig.subplots_adjust(hspace=0.3, wspace=0.4, top=0.85)
    fig.suptitle("Averaged BEDROC", fontsize=16)
    plt.savefig(output_file, dpi=300)
    plt.close()
