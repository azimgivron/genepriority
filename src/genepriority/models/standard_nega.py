from typing import Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD

from genepriority.models.base_nega import BaseNEGA


class Nega(BaseNEGA):
    """
    Matrix completion session without side information.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the session without side information.
        """
        super().__init__(*args, **kwargs)
        # Initialize factor matrices based solely on the matrix dimensions.
        num_rows, num_cols = self.matrix.shape

        # Apply the train mask: unobserved entries are set to zero
        observed_matrix = np.zeros_like(self.matrix)
        observed_matrix[self.train_mask] = self.matrix[self.train_mask]

        # Perform Truncated SVD on the observed matrix
        svd = TruncatedSVD(n_components=self.rank, n_iter=7, random_state=0)
        row_embeddings = svd.fit_transform(observed_matrix)  # shape: (n_rows, rank)
        singular_values = svd.singular_values_  # shape: (rank,)
        column_embeddings = svd.components_  # shape: (rank, n_cols)

        # Distribute singular values evenly across the two factor matrices
        sqrt_singular_values = np.diag(np.sqrt(singular_values))
        self.h1 = row_embeddings @ sqrt_singular_values  # shape: (n_rows, rank)
        self.h2 = sqrt_singular_values @ column_embeddings  # shape: (rank, n_cols)

        self.logger.debug(
            "Initialized h1 with shape %s and h2 with shape %s using masked TruncatedSVD",
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
            h(W) = 0.25 * ||W||_F^4 + 0.5 * tau * ||W||_F^2

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
            M_pred = h1 @ h2

        where:
        - h1 is the left factor matrix (shape: n x rank),
        - h2 is the right factor matrix (shape: rank x m).

        Returns:
            np.ndarray: The reconstructed (completed) matrix.
        """
        return self.h1 @ self.h2

    def compute_grad_f_W_k(self) -> np.ndarray:
        """Compute the gradients for for each latent as:

        grad_f_W_k = (∇_h1, ∇_h2.T).T

        where:
        - ∇_h1 = R @ h2.T + mu * h1,
        - ∇_h2 = h1.T @ R + mu * h2

        with R = (mask ⊙ (h1 @ h2 - M))

        Returns:
            np.ndarray: The gradient of the latents ((n+m) x rank)
        """
        residual = self.calculate_training_residual()
        grad_h1 = residual @ self.h2.T + self.regularization_parameter * self.h1
        grad_h2 = self.h1.T @ residual + self.regularization_parameter * self.h2
        return np.vstack([grad_h1, grad_h2.T])

    def substep(
        self,
        W_k: np.ndarray,
        tau: float,
        step_size: float,
        grad_f_W_k: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Performs a single substep in the optimization process to update the factor matrices.

        This substep calculates the next iterate W_{k+1} using the gradient of the objective
        function and an adaptive step size.

        Steps in the Substep Process:

        1. Compute the Gradient Step:
           grad = (||W_k||_F^2 + tau) * W_k - step_size * grad_f_W_k

           - ||W_k||_F: Frobenius norm of the current matrix W_k.
           - tau: Regularization parameter.
           - step_size: Learning rate for the gradient step.
           - grad_f_W_k: Gradient of the objective function at W_k.

        2. Solve the Cubic Equation for the Step Size t:
           Δ = tau_2^2 + (tau1/3)^3
           if Δ >= 0:

           t = (tau / 3) + cube_root(T_1) + cube_root(T_2)

           - T_1 = -tau_2 + sqrt(Δ)
           - T_2 = -tau_2 - sqrt(Δ)
           - tau_2 = (-2 * tau^3 - 27 * ||grad||_F^2) / 27

           The cubic root function ensures stability, even for negative T_1 and T_2.

        3. Update the Next Iterate W_{k+1}:
           W_{k+1} = (1 / t) * grad

        4. Split W_{k+1} into Factor Matrices h1 and h2:
           - h1 = W_{k+1}[:m, :]
           - h2 = W_{k+1}[m:, :].T

        Args:
            W_k (np.ndarray): Current stacked factor matrices.
            tau (float): Regularization parameter.
            step_size (float): Learning rate for the gradient step.
            grad_f_W_k (np.ndarray): Gradient of the objective function at W_k.

        Returns:
            Tuple[np.ndarray, float]:
                - Updated stacked matrix W_{k+1}.
                - New loss value f(W_{k+1}).
        """
        step = (np.linalg.norm(W_k, ord="fro") ** 2 + tau) * W_k - (
            step_size * grad_f_W_k
        )
        delta = np.linalg.norm(step, ord="fro") ** 2
        # Solve the cubic s³ – τ s² – Δ = 0 by Cardano
        t_k = self.cardano(tau, delta)

        W_k_next = (1 / t_k) * step
        split = self.h1.shape[0]
        self.h1 = W_k_next[:split, :]
        self.h2 = W_k_next[split:, :].T

        loss = self.calculate_loss()
        return W_k_next, loss
