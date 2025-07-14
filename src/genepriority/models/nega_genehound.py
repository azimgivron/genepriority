# pylint: disable=C0103,R0914
"""
NEGA with Side Information following the Bayesian formulation Module.
=====================================================================

This module implements Non-Euclidean Matrix Completion Algorithm following the Bayesian
formulation from GeneHound.
"""
from typing import Tuple

import numpy as np

from genepriority.models.nega_base import NegaBase
from genepriority.models.utils import init_from_svd


class NegaGeneHound(NegaBase):
    """
    Matrix completion with side information following the Bayesian
    formulation from GeneHound (Macau-style priors).

    This model solves the following optimization problem:

        Minimize:
            0.5 * || B ⊙ (h1 @ h2 - R) ||_F^2
            + 0.5 * λ * || h1 - 1 @ μ_h1 - G @ β_G ||_F^2
            + 0.5 * λ * || h2 - 1 @ μ_h2 - β_D.T @ D.T ||_F^2
            + 0.5 * λ' * || β_G ||_F^2
            + 0.5 * λ' * || β_D ||_F^2

    Attributes:
        gene_side_info (np.ndarray): Side information for genes (G ∈ R^{n x g}).
        disease_side_info (np.ndarray): Side information for diseases (D ∈ R^{m x d}).
        h1 (np.ndarray): Latent factor matrix for genes (n x k).
        h2 (np.ndarray): Latent factor matrix for diseases (k x m).
        beta_g (np.ndarray): Link matrix for gene side information (g x k).
        beta_d (np.ndarray): Link matrix for disease side information (d x k).
    """

    def __init__(
        self,
        *args,
        side_info: Tuple[np.ndarray, np.ndarray],
        side_information_reg: float,
        svd_init: bool = False,
        **kwargs,
    ):
        """
        Initialize the NegaGeneHound model with side information and regularization settings.

        Args:
            *args: Positional arguments forwarded to BaseNEGA.
            side_info (Tuple[np.ndarray, np.ndarray]):
                A tuple (G, D) of dense matrices containing gene side information
                G (n x g) and disease side information D (m x d).
            side_information_reg (float): Regularization weight for
                for the side information.
            svd_init (bool, optional): Whether to initialize the latent
                matrices with SVD decomposition. Default to False.
            **kwargs: Additional keyword arguments forwarded to BaseNEGA.

        Raises:
            ValueError: If side_info is None or if matrix and side-info dimensions mismatch.
        """
        super().__init__(*args, **kwargs)

        self.side_information_reg = side_information_reg

        if side_info is None:
            raise ValueError("Side information must be provided.")
        gene_side_info, disease_side_info = side_info

        if self.matrix.shape[0] != gene_side_info.shape[0]:
            raise ValueError("Matrix rows and gene side info rows mismatch.")
        if self.matrix.shape[1] != disease_side_info.shape[0]:
            raise ValueError("Matrix columns and disease side info rows mismatch.")

        self.gene_side_info = gene_side_info
        self.disease_side_info = disease_side_info

        if svd_init:
            # Apply the train mask: unobserved entries are set to zero
            observed_matrix = np.zeros_like(self.matrix)
            observed_matrix[self.train_mask] = self.matrix[self.train_mask]

            self.h1, self.h2 = init_from_svd(observed_matrix, self.rank)
        else:
            nb_genes, nb_diseases = self.matrix.shape
            self.h1 = np.random.randn(nb_genes, self.rank)
            self.h2 = np.random.randn(self.rank, nb_diseases)

        nb_gene_features = gene_side_info.shape[1]
        nb_disease_features = disease_side_info.shape[1]
        self.beta_g = np.random.randn(nb_gene_features, self.rank)
        self.beta_d = np.random.randn(nb_disease_features, self.rank)

    def init_tau(self) -> float:
        """
        Initialize the tau parameter used in the kernel function.

        Returns:
            float: Initial tau value, set to ||X||_F / 3.
        """
        return np.linalg.norm(self.matrix, ord="fro") / 3

    def init_Wk(self) -> np.ndarray:
        """
        Initialize weight block matrix.

        Returns:
            np.ndarray: The weight block matrix.
        """
        return np.vstack([self.h1, self.h2.T, self.beta_g, self.beta_d])

    def set_weights(self, weight_matrix: np.ndarray):
        """
        Set the weights individually from the stacked block matrix.

        Args:
            weight_matrix (np.ndarray): The stacked block matrix.
        """
        nb_genes, nb_diseases = self.matrix.shape
        gene_feat_dim = self.beta_g.shape[0]
        self.h1 = weight_matrix[0:nb_genes, :]
        self.h2 = weight_matrix[nb_genes : nb_genes + nb_diseases, :].T
        self.beta_g = weight_matrix[
            nb_genes + nb_diseases : nb_genes + nb_diseases + gene_feat_dim, :
        ]
        self.beta_d = weight_matrix[(nb_genes + nb_diseases + gene_feat_dim) :, :]

    def kernel(self, W: np.ndarray, tau: float) -> float:
        """
        Compute the kernel function h(W) = 0.25 * ||W||_F^4 + 0.5 * tau * ||W||_F^2.

        Args:
            W (np.ndarray): Latent parameter matrix.
            tau (float): Regularization parameter.

        Returns:
            float: Kernel value.
        """
        norm = np.linalg.norm(W, ord="fro")
        return 0.25 * norm**4 + 0.5 * tau * norm**2

    def predict_all(self) -> np.ndarray:
        """
        Compute the full matrix reconstruction R_hat = h1 @ h2.

        Returns:
            np.ndarray: Reconstructed matrix of shape (n, m).
        """
        return self.h1 @ self.h2

    def compute_grad_f_W_k(self) -> np.ndarray:
        """
        Compute the stacked gradient of the objective function w.r.t. all variables:

        grad_f_W_k = (∇_h1, ∇_h2.T, ∇_beta_g, ∇_beta_d).T

        with:
            - ∇_h1 = R @ h2.T + λ * (h1 - G @ β_G - 1 @ μ_{h1_res})
            - ∇_h2 = h1.T @ R + λ * (h2 - β_D.T @ D.T - μ_{h2_res} @ 1.T)
            - ∇_beta_g = λ′ * G.T @ (G @ β_G + 1 @ μ_h1 - h1) + λ′ * β_G
            - ∇_beta_d = λ′ * D.T @ (D @ β_D + 1 @ μ_h2.T - h2.T) + λ′ * β_D

        where
            R = mask ⊙ (h1 @ h2 - M)
            μ_{h1_res} = row mean of (h1 - G @ β_G), shape (1, k)
            μ_{h2_res} = column mean of (h2 - β_D.T @ D.T), shape (k, 1)
            1 = vector of ones
        """
        residuals = self.calculate_training_residual()  # shape (nb_genes, nb_diseases)
        nb_genes, nb_diseases = self.matrix.shape
        gene_prediction = self.gene_side_info @ self.beta_g  # (nb_genes, k)
        h1_residual = self.h1 - gene_prediction  # (nb_genes, k)
        mu_h1_residual = h1_residual.mean(axis=0, keepdims=True)  # (1, k)
        centered_h1 = (
            h1_residual - np.ones((nb_genes, 1)) @ mu_h1_residual
        )  # (nb_genes, k)

        grad_h1 = residuals @ self.h2.T + self.regularization_parameter * centered_h1

        disease_prediction = (
            self.beta_d.T @ self.disease_side_info.T
        )  # (k, nb_diseases)
        h2_residual = self.h2 - disease_prediction  # (k, nb_diseases)
        mu_h2_residual = h2_residual.mean(axis=1, keepdims=True)  # (k, 1)
        centered_h2 = h2_residual - mu_h2_residual @ np.ones(
            (1, nb_diseases)
        )  # (k, nb_diseases)

        grad_h2 = self.h1.T @ residuals + self.regularization_parameter * centered_h2

        mu_h1 = self.h1.mean(axis=0, keepdims=True)  # (1, k)
        grad_beta_g = (
            self.regularization_parameter
            * self.gene_side_info.T
            @ (gene_prediction + np.ones((nb_genes, 1)) @ mu_h1 - self.h1)
            + self.side_information_reg * self.beta_g
        )

        mu_h2 = self.h2.mean(axis=1, keepdims=True)  # (k, 1)
        grad_beta_d = (
            self.regularization_parameter
            * self.disease_side_info.T
            @ (
                self.disease_side_info @ self.beta_d
                + np.ones((nb_diseases, 1)) @ mu_h2.T
                - self.h2.T
            )
            + self.side_information_reg * self.beta_d
        )
        grad_Wk_next = np.vstack(
            [
                grad_h1,
                grad_h2.T,
                grad_beta_g,
                grad_beta_d,
            ]
        )
        return grad_Wk_next
