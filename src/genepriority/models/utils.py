# pylint: disable=R0913, R0914
"""
Utilities Module
================

Contains utility functions for the internal functioning of the models module.
This module provides tools for creating visualizations of high-dimensional embeddings,
such as generating a t-SNE plot and converting it to a TensorFlow tensor for
logging with tf.summary.image. Additional utility functions may be added in the future.
"""
import io
from pathlib import Path
from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE


def init_from_svd(
    observed_matrix: np.ndarray, rank: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize low-rank factors from a truncated SVD of the observed matrix.

    Args:
        observed_matrix (np.ndarray): Matrix of shape (n_rows, n_cols).
        rank (int): Target rank for the approximation.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - left_factor: shape (n_rows, rank)
            - right_factor: shape (rank, n_cols)
    """
    U, S, Vt = np.linalg.svd(observed_matrix, full_matrices=False)
    U_r = U[:, :rank] # shape (n_rows, rank)
    S_r = S[:rank] # shape (rank,)
    Vt_r = Vt[:rank, :] # shape (rank, n_cols)

    sqrt_S = np.sqrt(S_r)
    left_factor = U_r * sqrt_S # shape (n_rows, rank)
    right_factor = sqrt_S[:, None] * Vt_r # shape (rank, n_cols)

    return left_factor, right_factor


def tsne_plot_to_tensor(
    high_dim_embeddings: np.ndarray,
    color: str,
    perplexity: float = 25,
    random_state: int = 0,
    figsize: tuple = (8, 8),
    max_iter: int = 1000,
    dpi: int = 50,
    max_samples: int = 500,
) -> tf.Tensor:
    """
    Generate a t-SNE visualization of high-dimensional embeddings and return the plot
        as a tensor.

    The output tensor is formatted as [k, h, w, c] where:
      - k: number of images (batch size, here 1)
      - h: image height
      - w: image width
      - c: number of channels (4 for RGBA)

    Args:
        high_dim_embeddings (np.ndarray): Array of shape (n_samples, n_features) containing
            the input embeddings.
        color (str or array-like): Color or an array of colors for the scatter plot points.
        perplexity (float, optional): Perplexity parameter for t-SNE. Default is 25.
        random_state (int, optional): Seed for reproducibility. Default is 0.
        figsize (tuple, optional): Size of the output figure. Default is (8, 8).
        max_iter (int, optional): Maximum iterations for t-SNE. Default is 1000.
        dpi (int, optional): Dots per inch for the figure resolution. Default is 50.
        max_samples (int, optional): The maximum number of samples to use. Default is 500.

    Returns:
        tf.Tensor: A tensor representing the t-SNE plot image with shape
            [1, height, width, channels].
    """
    n_samples = high_dim_embeddings.shape[0]
    if n_samples > max_samples:
        indices = np.random.choice(n_samples, size=max_samples, replace=False)
        high_dim_embeddings = high_dim_embeddings[indices]

    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(high_dim_embeddings) / 2),
        random_state=random_state,
        max_iter=max_iter,
    )
    low_dim_embedding = tsne.fit_transform(high_dim_embeddings)

    fig, axis = plt.subplots(figsize=figsize, dpi=dpi)
    axis.scatter(low_dim_embedding[:, 0], low_dim_embedding[:, 1], c=color)
    axis.set_xlabel("Component 1")
    axis.set_ylabel("Component 2")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)

    image_tensor = tf.io.decode_png(buf.getvalue(), channels=4)
    image_tensor = tf.expand_dims(image_tensor, axis=0)  # shape becomes [1, h, w, c]
    plt.close(fig)
    return image_tensor


def check_save_name(save_name: Any) -> Path:
    """
    Validate and convert save_name to an absolute Path.

    Args:
        save_name (Any): A file path (str or Path) or None.

    Returns:
        Path: An absolute Path corresponding to save_name, or None if save_name is None.

    Raises:
        TypeError: If save_name is not a str, Path, or None.
        FileNotFoundError: If the parent directory of save_name does not exist.
    """
    if save_name is None:
        pass
    elif isinstance(save_name, str):
        save_name = Path(save_name).absolute()
    elif isinstance(save_name, Path):
        save_name = save_name.absolute()
    else:
        raise TypeError(
            "`save_name` must be either of type str or Path. "
            f"Provided `save_name` of type {type(save_name)} is not supported."
        )

    if save_name is not None and not save_name.parent.exists():
        raise FileNotFoundError(
            f"The directory {save_name.parent} does not exist. "
            "Please provide a valid path for `save_name`."
        )
    return save_name
