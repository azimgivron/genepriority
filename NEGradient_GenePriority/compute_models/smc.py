# pylint: disable=C0103, R0913, R0914, R0915, R0902
"""
Matrix Completion Module
========================

Implements a matrix completion algorithm with adaptive step size. The API is designed
to be straightforward and intuitive, with explicitly defined parameters for configuration.

Features:
- Training with adaptive optimization.
- Evaluation through RMSE and loss metrics.
"""
import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import mean_squared_error


@dataclass
class MatrixCompletionResult:
    """
    Stores the results of the matrix completion process.

    Attributes:
        completed_matrix (sp.csr_matrix):
            The completed matrix after the optimization process, representing
            the low-rank approximation of the input sparse matrix.
        loss_history (List[float]):
            A list of loss values recorded at each iteration of the training process.
        rmse_history (List[float]):
            A list of RMSE (Root Mean Squared Error) values recorded during each
            iteration on the test set to monitor performance.
        runtime (float):
            The total runtime of the optimization process, measured in seconds.
        iterations (int):
            The total number of iterations performed during the optimization process.
    """

    completed_matrix: sp.csr_matrix
    loss_history: List[float]
    rmse_history: List[float]
    runtime: float
    iterations: int


class MatrixCompletionSession:
    """
    Manages the configuration, training, and evaluation of a matrix completion model.

    Attributes:
        matrix (sp.csr_matrix): Input matrix to be approximated. Shape: (m, n).
        train_mask (sp.csr_matrix): Mask indicating observed entries in `matrix` for training.
            Shape: (m, n).
        test_mask (sp.csr_matrix): Mask indicating observed entries in `matrix` for test.
            Shape: (m, n).
        rank (int): The rank of the low-rank approximation.
        optim_reg (float): Regularization parameter for the optimization.
        iterations (int): Maximum number of optimization iterations.
        lam (float): Regularization parameter for the gradient adjustment.
        step_size (float): Initial step size for gradient-based updates.
        rho_increase (float): Factor for increasing the step size dynamically.
        rho_decrease (float): Factor for decreasing the step size dynamically.
        threshold (int): Maximum iterations allowed for the inner optimization loop.
        h1 (sp.csr_matrix): Left factor matrix in the low-rank approximation
            (shape: rows of `matrix` x `rank`).
        h2 (sp.csr_matrix): Right factor matrix in the low-rank approximation
            (shape: `rank` x columns of `matrix`).
        logger (logging.Logger): Logger instance for debugging and progress tracking.
        seed (int, optional): Seed for reproducible random initialization.
        save_name (str, optional): The file path where the model will be saved.
            If set to None, the model will not be saved after training.
    """

    def __init__(
        self,
        matrix: sp.csr_matrix,
        train_mask: sp.csr_matrix,
        test_mask: sp.csr_matrix,
        rank: int,
        optim_reg: float,
        iterations: int,
        lam: float,
        step_size: float,
        rho_increase: float,
        rho_decrease: float,
        threshold: int,
        logger: logging.Logger = None,
        seed: int = 123,
        save_name: str = None,
    ):
        """
        Initializes the MatrixCompletionSession instance.

        Args:
            matrix (sp.csr_matrix): Input matrix to be approximated.
                Shape: (m, n).
            train_mask (sp.csr_matrix): Mask indicating observed entries in `matrix` for training.
                Shape: (m, n).
            test_mask (sp.csr_matrix): Mask indicating observed entries in
                `matrix` for test. Shape: (m, n).
            rank (int): Desired rank for the low-rank approximation.
            optim_reg (float): Regularization parameter for the optimization.
            iterations (int): Maximum number of optimization iterations.
            lam (float): Regularization parameter for gradient adjustment.
            step_size (float): Initial step size for optimization updates.
            rho_increase (float): Multiplicative factor to increase step size dynamically.
            rho_decrease (float): Multiplicative factor to decrease step size dynamically.
            threshold (int): Maximum number of iterations for the inner loop.
            logger (logging.Logger, optional): Logger instance for tracking progress.
                Default is `None`.
            seed (int, optional): Seed for reproducible random initialization.
                Default is 123.
            save_name (str, optional): The file path where the model will be saved.
                If set to None, the model will not be saved after training.  Defaults to None.
        """
        self.matrix = matrix
        self.train_mask = train_mask
        self.test_mask = test_mask
        self.rank = rank
        self.optim_reg = optim_reg
        self.iterations = iterations
        self.lam = lam
        self.step_size = step_size
        self.rho_increase = rho_increase
        self.rho_decrease = rho_decrease
        self.threshold = threshold

        if save_name is None:
            # If save_name is None, set self.save_name to None (no saving)
            self.save_name = None
        elif isinstance(save_name, str):
            # If save_name is a string, convert it to an absolute Path
            self.save_name = Path(save_name).absolute()
        elif isinstance(save_name, Path):
            # If save_name is already a Path object, ensure it's absolute
            self.save_name = save_name.absolute()
        else:
            # Raise a TypeError for unsupported types
            raise TypeError(
                "`save_name` must be either of type str or Path. "
                f"Provided `save_name` of type {type(save_name)} is not supported."
            )

        # Additional check to ensure the path's parent directory exists
        if self.save_name is not None and not self.save_name.parent.exists():
            raise FileNotFoundError(
                f"The directory {self.save_name.parent} does not exist. "
                "Please provide a valid path for `save_name`."
            )

        # Set random seed for reproducibility
        np.random.seed(seed)

        # Initialize logger
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

        # Initialize factor matrices h1 and h2 with random values
        num_rows, num_cols = matrix.shape
        self.h1 = sp.csr_matrix(np.random.randn(num_rows, rank))
        self.h2 = sp.csr_matrix(np.random.randn(rank, num_cols))
        self.logger.debug(
            "Initialized h1 and h2 with shapes %s and %s", self.h1.shape, self.h2.shape
        )

    def predict_all(self) -> sp.csr_matrix:
        """
        Computes the reconstructed matrix from the factor matrices h1 and h2.

        Mathematically, the completed matrix is computed as:
            M_pred = h1 * h2

        where:
        - h_1 is the left factor matrix (shape: m x rank),
        - h_2 is the right factor matrix (shape: rank x n).

        Returns:
            sp.csr_matrix: The reconstructed (completed) matrix.
        """
        return self.h1 @ self.h2

    def calculate_loss(self) -> float:
        """
        Computes the loss function value for the training data.

        The loss is defined as the Frobenius norm of the residual matrix
        for observed entries only:
            Loss = 0.5 * ||M - M_pred||_F^2

        (only over observed entries indicated by `mask`)

        where:
        - M is the input matrix (shape: m x n),
        - M_pred is the predicted (reconstructed) matrix

        Returns:
            float: The computed loss value.
        """
        residual = (self.matrix - self.predict_all()).multiply(self.train_mask)
        return 0.5 * sp.linalg.norm(residual, ord="fro") ** 2

    def calculate_rmse(self) -> float:
        """
        Computes the Root Mean Square Error (RMSE) for the test data.

        The RMSE measures the average deviation between the predicted matrix
        and the test matrix, considering only the observed entries.

        Formula for RMSE:
            RMSE = sqrt( (1 / |Omega|) * sum( (M_test[i, j] - M_pred[i, j])^2 ) )

        Where:
            - Omega: The set of observed indices in `test_mask`.
            - M_test: The test matrix containing actual values.
            - M_pred: The predicted matrix (low-rank approximation).

        Process:
        1. Extract the observed entries from both the test matrix (`M_test`)
           and the predicted matrix (`M_pred`) using the `test_mask`.
        2. Compute the squared differences for the observed entries.
        3. Calculate the mean of these squared differences.
        4. Take the square root of the mean to obtain the RMSE.

        Returns:
            float: The computed RMSE value, representing the prediction error
                   on the observed test entries.
        """
        # Extract observed test values and corresponding predictions
        row_indices, col_indices = self.test_mask.nonzero()
        test_values_actual = np.asarray(self.matrix[row_indices, col_indices])
        test_predictions = np.asarray(self.predict_all()[row_indices, col_indices])

        # Compute RMSE
        rmse = np.sqrt(mean_squared_error(test_values_actual, test_predictions))
        return rmse

    def update_factors(self, Wk, tau, alpha_k, grad_f_Wk, t1, m):
        """
        Performs a single substep in the optimization process to update the factor matrices.

        This substep calculates the next iterate W_{k+1} using the gradient of the objective
        function and an adaptive step size.

        Steps in the Substep Process:

        1. Compute the Gradient Step:
           grad = (||Wk||_F^2 + tau) * Wk - alpha_k * grad_f_Wk

           - ||Wk||_F: Frobenius norm of the current matrix Wk.
           - tau: Regularization parameter.
           - alpha_k: Learning rate for the gradient step.
           - grad_f_Wk: Gradient of the objective function at Wk.

        2. Solve the Cubic Equation for the Step Size t:
           t = (tau / 3) + cube_root(T_1) + cube_root(T_2)

           - T_1 = -tau_2 + sqrt(tau_2^2 + t1^3)
           - T_2 = -tau_2 - sqrt(tau_2^2 + t1^3)
           - tau_2 = (-2 * tau^3 - 27 * ||grad||_F^2) / 54

           The cubic root function ensures stability, even for negative T_1 and T_2.

        3. Update the Next Iterate W_{k+1}:
           W_{k+1} = (1 / t) * grad

        4. Split W_{k+1} into Factor Matrices h_1 and h_2:
           - h_1 = W_{k+1}[:m, :]
           - h_2 = W_{k+1}[m:, :].T

        Args:
            Wk (sp.csr_matrix): Current stacked factor matrices.
            tau (float): Regularization parameter.
            alpha_k (float): Learning rate for the gradient step.
            grad_f_Wk (sp.csr_matrix): Gradient of the objective function at Wk.
            t1 (float): Cubic root parameter for step size adjustment.
            m (int): Number of rows in the original matrix.

        Returns:
            tuple:
                - Updated stacked matrix W_{k+1} (sp.csr_matrix).
                - New loss value f(W_{k+1}) (float).
        """
        # Compute the gradient step
        grad = (sp.linalg.norm(Wk, ord="fro") ** 2 + tau) * Wk - (alpha_k * grad_f_Wk)
        tau2 = (-2 * (tau**3) - 27 * (sp.linalg.norm(grad, ord="fro") ** 2)) / 27 / 2

        # Compute step sizes
        T1 = -tau2 + np.sqrt(tau2**2 + t1**3)
        T2 = -tau2 - np.sqrt(tau2**2 + t1**3)
        t = (tau / 3) + pow(T1, 1 / 3) + pow(T2, 1 / 3)

        # Update Wk
        Wk1 = (1 / t) * grad
        self.h1 = sp.csr_matrix(Wk1[:m, :])
        self.h2 = sp.csr_matrix(Wk1[m:, :].T)

        # Compute the new loss
        fk1 = self.calculate_loss()
        return Wk1, fk1

    def run(self) -> MatrixCompletionResult:
        """
        Performs matrix completion using adaptive step size optimization.

        The optimization process minimizes the objective function:
            f(W) = 0.5 * ||M - M_pred||_F^2 + mu * ||W||_F^2

        where:
            - M: The input matrix (shape: m x n).
            - M_pred: The low-rank approximation, computed as h_1 @ h_2.
            - ||W||_F: The Frobenius norm of the stacked factor matrix W.
            - mu: The regularization parameter.

        **Key Steps in the Optimization Process:**

        1. **Initialization**:
           Initialize the factor matrices:
               - h_1 (shape: m x rank),
               - h_2 (shape: rank x n).

        2. **Iterative Updates**:
           Update the stacked factor matrix W_k = [h_1; h_2^T] using gradients and adaptive
           step sizes:
               - ∇_u = (mask ⊙ (h_1 @ h_2 - M)) @ h_2^T + mu * h_1,
               - ∇_v = (mask.T ⊙ (h_2^T @ h_1^T - M.T)) @ h_1 + mu * h_2^T,
           where ⊙ represents element-wise multiplication.

        3. **Dynamic Step Size Adjustment**:
           Adjust the step size if the new loss does not satisfy the improvement condition:
               f(W_{k+1}) > f(W_k) + ∇f(W_k)^T (W_{k+1} - W_k) + η * D_h(W_{k+1}, W_k, τ),
           where:
               - η is the step size adjustment parameter,
               - D_h is the difference in h-function values.

        4. **Termination**:
           Stop when the maximum number of iterations is reached or the loss converges.

        **Returns**:
            MatrixCompletionResult: A dataclass containing:
                - completed_matrix: The reconstructed matrix (low-rank approximation).
                - loss_history: List of loss values at each iteration.
                - rmse_history: List of RMSE values at each iteration.
                - runtime: Total runtime of the optimization process.
                - iterations: Total number of iterations performed.
        """
        # Start measuring runtime
        start_time = time.time()
        m, n = self.matrix.shape

        # Initialize loss and RMSE history
        f = self.calculate_loss()
        loss = [f / (m * n)]
        rmse = [self.calculate_rmse()]

        # Stack h1 and h2 for optimization
        Wk = sp.vstack([self.h1, self.h2.T])
        alpha_k = 1 / self.step_size
        tau = sp.linalg.norm(self.matrix, ord="fro") / 3
        tau1 = -(tau**2) / 3
        t1 = tau1 / 3

        self.logger.debug("Starting optimization with tau=%f, alpha_k=%f", tau, alpha_k)

        # Main optimization loop
        iterations_count = 0
        for i in range(self.iterations):
            iterations_count = i
            self.logger.debug("[Main Loop] Iteration %d started.", i)
            # Compute gradients for h1 and h2
            residual = (
                self.train_mask.multiply(self.h1 @ self.h2) - self.matrix
            ).toarray()
            grad_u = residual @ self.h2.T + self.optim_reg * self.h1
            grad_v = residual.T @ self.h1 + self.optim_reg * self.h2.T
            grad_f_Wk = sp.vstack([grad_u, grad_v])

            Wk1, fk1 = self.update_factors(Wk, tau, alpha_k, grad_f_Wk, t1, m)
            self.logger.debug("[Iteration %d] Loss calculated: %.6e", i, fk1)
            j = 0
            flag = 0

            # Inner loop to adjust step size
            gradient_term = (grad_f_Wk.T @ (Wk1 - Wk)).sum()
            h_difference_term = self.step_size * compute_h_difference(Wk1, Wk, tau)
            self.logger.debug(
                "[Inner Loop Entry] Iteration %d: Loss=%.6e, GradientTerm=%.6e, H-Difference=%.6e",
                i,
                fk1,
                gradient_term,
                h_difference_term,
            )
            while fk1 > f + gradient_term + h_difference_term:
                flag = 1
                j += 1
                self.logger.debug(
                    "[Step Adjustment] Iteration %d, Inner Loop %d: Step size being adjusted.",
                    i,
                    j,
                )
                # Detect overflow and exit if necessary
                if self.rho_increase**j > np.finfo(float).max:
                    self.logger.warning(
                        "[Overflow Detected] Iteration %d, Inner Loop %d: "
                        "Exiting to prevent overflow.",
                        i,
                        j,
                    )
                    break
                if j == self.threshold:
                    self.logger.warning(
                        "[Inner Loop Limit] Iteration %d, Inner Loop %d: Maximum "
                        "allowed iterations reached.",
                        i,
                        j,
                    )
                    break
                # Adjust step size
                self.logger.debug(
                    "[Step Size Adjustment] Increasing step size at Iteration %d, Inner Loop %d.",
                    i,
                    j,
                )
                self.step_size *= self.rho_increase**j
                alpha_k = (1 + self.lam) / self.step_size

                Wk1, fk1 = self.update_factors(Wk, tau, alpha_k, grad_f_Wk, t1 / 3, m)
                self.logger.debug(
                    "[Inner Loop Update] Iteration %d, Inner Loop %d: Updated Loss=%.6e",
                    i,
                    j,
                    fk1,
                )

                gradient_term = (grad_f_Wk.T @ (Wk1 - Wk)).sum()
                h_difference_term = self.step_size * compute_h_difference(Wk1, Wk, tau)
            self.logger.debug("[Inner Loop Exit] Iteration %d: Loss=%.6e", i, fk1)
            if flag == 1:
                # Adjust step size
                self.logger.debug(
                    "[Step Size Adjustment] Decreasing step size after Inner Loop at Iteration %d.",
                    i,
                )
                self.step_size *= self.rho_decrease

            # Update variables for the next iteration
            Wk = Wk1
            alpha_k = (1 + self.lam) / self.step_size
            loss.append(fk1 / (m * n))
            rmse.append(self.calculate_rmse())

            # Log iteration metrics
            self.logger.debug(
                "[Metrics Log] Iteration %d: RMSE=%.6f, Normalized Loss=%.6f",
                i,
                rmse[-1],
                loss[-1],
            )

            # Break if loss becomes NaN
            if np.isnan(fk1):
                self.logger.warning(
                    "[NaN Loss] Iteration %d: Loss is NaN, exiting loop.", i
                )
                break
            f = fk1

        # Compute runtime
        runtime = time.time() - start_time
        self.logger.debug(
            "[Completion] Optimization finished in %.2f seconds.", runtime
        )
        if self.save_name is not None:
            with open(self.save_name, "wb") as handler:
                pickle.dump(self, handler)
        training_data = MatrixCompletionResult(
            completed_matrix=self.predict_all(),
            loss_history=loss,
            iterations=iterations_count,
            rmse_history=rmse,
            runtime=runtime,
        )
        return training_data


def compute_h_function(W: sp.csr_matrix, tau: float) -> float:
    """
    Computes the value of the h function for a given matrix W and regularization parameter tau.

    The h function is defined as:
        h(W) = 0.25 * ||W||_F^4 + 0.5 * tau * ||W||_F^2

    Args:
        W (sp.csr_matrix): The input sparse matrix.
        tau (float): Regularization parameter.

    Returns:
        float: The computed value of the h function.
    """
    # Calculate the Frobenius norm of the matrix W
    frobenius_norm = sp.linalg.norm(W, ord="fro")

    # Compute the h function value
    h_value = 0.25 * frobenius_norm**4 + 0.5 * tau * frobenius_norm**2
    return h_value


def compute_h_difference(W1: sp.csr_matrix, W2: sp.csr_matrix, tau: float) -> float:
    """
    Computes the difference in h function values between two matrices (W1 and W2),
    adjusted by the gradient of h at W2.

    The difference is computed as:
        D_h = h(W1) - h(W2) - <grad_h(W2), W1 - W2>

    Args:
        W1 (sp.csr_matrix): The first input sparse matrix.
        W2 (sp.csr_matrix): The second input sparse matrix.
        tau (float): Regularization parameter.

    Returns:
        float: The computed difference in h function values.
    """
    # Compute h function values for W1 and W2
    h_value_W1 = compute_h_function(W1, tau)
    h_value_W2 = compute_h_function(W2, tau)

    # Compute the gradient of the h function at W2
    frobenius_norm_W2 = sp.linalg.norm(W2, ord="fro")
    grad_h_W2 = (frobenius_norm_W2**2 + tau) * W2

    # Compute the difference in h values, incorporating the gradient adjustment
    gradient_adjustment = (grad_h_W2.T @ (W1 - W2)).sum()
    h_difference = h_value_W1 - h_value_W2 - gradient_adjustment

    return h_difference
