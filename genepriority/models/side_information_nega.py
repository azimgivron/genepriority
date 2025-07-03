from typing import Tuple

import numpy as np
import scipy.sparse as sp

from genepriority.models.base_nega import BaseNEGA


class NegaSi(BaseNEGA):
    """
    Specialized matrix completion session that incorporates side information.

    Attributes:
        gene_side_info (np.ndarray): Side information for genes (shape: n x g).
        disease_side_info (np.ndarray): Side information for diseases (shape: m x d).
        gene_similarity_inv (np.ndarray): Pseudoinverse of the gene-by-gene similarity
            matrix (shape: n x n).
        disease_similarity_inv (np.ndarray): Pseudoinverse of the disease-by-disease
            similarity matrix (shape: m x m).
        h1 (np.ndarray): Left latent factor matrix (shape: g x rank).
        h2 (np.ndarray): Right latent factor matrix (shape: rank x d).
    """

    def __init__(
        self, *args, side_info: Tuple[sp.csr_matrix, sp.csr_matrix] = None, **kwargs
    ):
        """
        Initializes the session with side information.

        Args:
            side_info (Tuple[sp.csr_matrix, sp.csr_matrix]): Tuple containing gene and disease
                side information matrices.
        """
        super().__init__(*args, **kwargs)
        if side_info is None:
            raise ValueError("Side information must be provided for this session.")
        gene_side_info, disease_side_info = side_info
        if self.matrix.shape[0] != gene_side_info.shape[0]:
            raise ValueError(
                "Dimension 0 of matrix does not match dimension 0 of gene side information."
            )
        if self.matrix.shape[1] != disease_side_info.shape[0]:
            raise ValueError(
                "Dimension 1 of matrix does not match dimension 0 of disease side information."
            )

        self.gene_side_info = gene_side_info
        self.disease_side_info = disease_side_info

        # Initialize factor matrices based on side information dimensions.
        self.h1 = np.random.randn(self.gene_side_info.shape[1], self.rank)
        self.h2 = np.random.randn(self.rank, self.disease_side_info.shape[1])

        self.logger.debug(
            "Initialized h1 with shape %s and h2 with shape %s",
            self.h1.shape,
            self.h2.shape,
        )

    def init_tau(self) -> float:
        """
        Initialize tau value.

        Returns:
            float: tau value.
        """
        return np.linalg.norm(self.matrix, ord="fro") / 3

    def kernel(self, W: np.ndarray, tau: float) -> float:
        """
        Computes the value of the kernel function h for a given matrix W and
        regularization parameter tau.

        The h function is defined as:
            h(W) = 0.75 * (||X @ h1||_F^2 + ||h2 @ Y.T||_F^2)^2
            + 1.5 * tau * (||X @ h1||_F^2 + ||h2 @ Y.T||_F^2)

        Args:
            W (np.ndarray): The input matrix.
            tau (float): Regularization parameter.

        Returns:
            float: The computed value of the h function.
        """
        norm = np.linalg.norm(W, ord="fro")
        h_value = 0.25 * norm**4 + 0.5 * tau * norm**2
        return h_value

    def predict_all(self) -> np.ndarray:
        """
        Computes the reconstructed matrix from the factor matrices h1 and h2.

        Mathematically, the completed matrix is computed as:
            M_pred = (X @ h1) @ (h2 @ Y.T)

        where:
        - X is the gene feature matrix (shape: n x g),
        - h1 is the left factor matrix (shape: g x rank),
        - h2 is the right factor matrix (shape: rank x d),
        - Y is the disease feature matrix (shape: m x d).

        Returns:
            np.ndarray: The reconstructed (completed) matrix.
        """
        return (self.gene_side_info @ self.h1) @ (self.h2 @ self.disease_side_info.T)

    def compute_grad_f_W_k(self) -> np.ndarray:
        """Compute the gradients for for each latent as:

        grad_f_W_k = (∇_h1, ∇_h2.T).T

        with:
        - ∇_h1 = X.T @ (R @ (Y @ h2.T)) + mu * h1,
        - ∇_h2 = ((X @ h1).T @ R) @ Y + mu * h2

        with R = (mask ⊙ ((X @ h1) @ (h2 @ Y.T) - M))

        Returns:
            np.ndarray: The gradient of the latents ((g+d) x rank)
        """
        residual = self.calculate_training_residual()
        grad_h1 = (
            self.gene_side_info.T @ (residual @ (self.disease_side_info @ self.h2.T))
            + self.regularization_parameter * self.h1
        )
        grad_h2 = (
            ((self.gene_side_info @ self.h1).T @ residual)
        ) @ self.disease_side_info + self.regularization_parameter * self.h2
        return np.vstack([grad_h1, grad_h2.T])

    def substep(
        self,
        W_k: np.ndarray,
        tau: float,
        step_size: float,
        grad_f_W_k: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Performs a single non-Euclidean Bregman-proximal update step for the side-information kernel.

        Given the current stacked latent W_k = [h1_k; h2_kᵀ], with
            h1_k ∈ R^{gxr},   h2_k ∈ R^{rxd},
        and side-information matrices
            X ∈ R^{nxg},   Y ∈ R^{mxd},

        the update is:

            1. Energy scalar:
            φ_k = ‖X h1_k‖_F² + ‖h2_k Yᵀ‖_F²

            2. Shift-and-scale numerator:
            step = (φ_k + τ) W_k - (step_size / 3) K⁻¹ ∇f(W_k),
            where K = diag(XᵀX, Y Yᵀ) and ∇f(W_k) = grad_f_W_k

            3. Quadratic form for normalization:
            Δ = ‖X step_h1‖_F² + ‖step_h2 Yᵀ‖_F²

            4. Normalizer via Cardano’s formula:
            s_k = cardano(tau, Δ),
            solving s³ - τ s² - Δ = 0

            5. Final update:
            W_{k+1} = step / s_k,
            then unstack h1 ← first g rows, h2 ← last d rows (transposed)

        Args:
            W_k (np.ndarray of shape (g+d, r)): Current stacked latent [h1_k; h2_kᵀ].
            tau (float): Regularization parameter τ.
            step_size (float): Step-size alpha in the Bregman update.
            grad_f_W_k (np.ndarray of shape (g+d, r)): Euclidean gradient ∇f(W_k).

        Returns:
            Tuple[np.ndarray, float]:
                - np.ndarray of shape (g+d, r): Updated stacked latent W_{k+1}.
                - float: Loss value at W_{k+1}.
        """
        step = (np.linalg.norm(W_k, ord="fro") ** 2 + tau) * W_k - (
            step_size * grad_f_W_k
        )
        delta = np.linalg.norm(step, ord="fro") ** 2
        # Solve the cubic s³ – τ s² – Δ = 0 by Cardano
        t_k = self.cardano(tau, delta)

        W_k_next = (1 / t_k) * step
        m = self.h1.shape[0]
        self.h1 = W_k_next[:m, :]
        self.h2 = W_k_next[m:, :].T

        loss = self.calculate_loss()
        return W_k_next, loss
