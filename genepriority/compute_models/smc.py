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
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from sklearn import metrics

from genepriority.evaluation.metrics import bedroc_score

from genepriority.compute_models.matrix_completion_result import MatrixCompletionResult
from genepriority.utils import serialize


class MatrixCompletionSession:
    """
    Manages the configuration, training, and evaluation of a matrix completion model.

    Attributes:
        matrix (np.ndarray): Input matrix to be approximated. Shape: (m, n).
        train_mask (np.ndarray): Mask indicating observed entries in `matrix` for training.
            Shape: (m, n).
        test_mask (np.ndarray): Mask indicating observed entries in `matrix` for test.
            Shape: (m, n).
        rank (int): The rank of the low-rank approximation.
        regularization_parameter (float): Regularization parameter for the optimization.
        iterations (int): Maximum number of optimization iterations.
        symmetry_parameter (float): Symmetry parameter for the gradient adjustment.
        smoothness_parameter (float): Initial smoothness parameter.
        rho_increase (float): Factor for increasing the step size dynamically.
        rho_decrease (float): Factor for decreasing the step size dynamically.
        threshold (int): Maximum iterations allowed for the inner optimization loop.
        h1 (sp.csr_matrix): Left factor matrix in the low-rank approximation
            (shape: rows of `matrix` x `rank`).
        h2 (sp.csr_matrix): Right factor matrix in the low-rank approximation
            (shape: `rank` x columns of `matrix`).
        logger (logging.Logger): Logger instance for debugging and progress tracking.
        writer (tf.summary.SummaryWriter): Tensorflow summary writer.
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
        regularization_parameter: float,
        iterations: int,
        symmetry_parameter: float,
        smoothness_parameter: float,
        rho_increase: float,
        rho_decrease: float,
        threshold: int,
        writer: tf.summary.SummaryWriter = None,
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
            regularization_parameter (float): Regularization parameter for the optimization.
            iterations (int): Maximum number of optimization iterations.
            symmetry_parameter (float): The symmetry parameter.
            smoothness_parameter (float): Initial smoothness parameter.
            rho_increase (float): Multiplicative factor to increase step size dynamically.
            rho_decrease (float): Multiplicative factor to decrease step size dynamically.
            threshold (int): Maximum number of iterations for the inner loop.
            writer (tf.summary.SummaryWriter, optional): Tensorflow summary writer.
                Default is `None`.
            seed (int, optional): Seed for reproducible random initialization.
                Default is 123.
            save_name (str, optional): The file path where the model will be saved.
                If set to None, the model will not be saved after training.  Defaults to None.
        """
        self.matrix = matrix.toarray()
        self.train_mask = train_mask.toarray().astype(bool)
        self.test_mask = test_mask.toarray().astype(bool)
        self.rank = rank
        self.regularization_parameter = regularization_parameter
        self.iterations = iterations
        self.symmetry_parameter = symmetry_parameter
        self.smoothness_parameter = smoothness_parameter
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

        self.logger = logging.getLogger(self.__class__.__name__)
        self.writer = writer

        # Initialize factor matrices h1 and h2 with random values
        num_rows, num_cols = matrix.shape
        self.h1 = np.random.randn(num_rows, rank)
        self.h2 = np.random.randn(rank, num_cols)
        self.logger.debug(
            "Initialized h1 and h2 with shapes %s and %s", self.h1.shape, self.h2.shape
        )

    def predict_all(self) -> np.ndarray:
        """
        Computes the reconstructed matrix from the factor matrices h1 and h2.

        Mathematically, the completed matrix is computed as:
            M_pred = h1 * h2

        where:
        - h_1 is the left factor matrix (shape: m x rank),
        - h_2 is the right factor matrix (shape: rank x n).

        Returns:
            np.ndarray: The reconstructed (completed) matrix.
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
        residual = (self.matrix - self.predict_all())
        residual[~self.train_mask] = 0
        return 0.5 * np.linalg.norm(residual, ord='fro') ** 2

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
        mask = self.test_mask & (self.matrix == 1)
        test_values_actual = self.matrix[mask]
        test_predictions = self.predict_all()[mask]

        # Compute RMSE
        rmse = np.sqrt(metrics.mean_squared_error(test_values_actual, test_predictions))
        return rmse

    def calculate_auc_bedroc(self, alpha: float = 160.9) -> tuple[float, float, float]:
        """
        Computes the Area Under the Curve (AUC-ROC), Average Precision (AP),
        and the Boltzmann-Enhanced Discrimination of ROC (BEDROC) score for the test data.

        - AUC-ROC measures the model's ability to rank positive instances
        higher than negative ones, capturing its overall classification performance.

        - Average Precision (AP) represents the weighted mean of precisions
        at different thresholds, considering both precision and recall.

        - BEDROC extends AUC-ROC by emphasizing early retrieval, making it
        particularly useful for ranking problems where top predictions matter most.

        Formula for AUC-ROC:
            AUC = ∫ TPR(FPR) d(FPR)

        Formula for AP:
            AP = sum( (R_n - R_{n-1}) * P_n )

        Formula for BEDROC:
            BEDROC = (1 / N) * sum(exp(-alpha * rank(y_i))) * normalization_factor

        Where:
            - TPR: True Positive Rate.
            - FPR: False Positive Rate.
            - N: Number of positive samples.
            - rank(y_i): Rank of the positive sample in the sorted predictions.
            - alpha: Controls the early recognition emphasis.
            - R_n: Recall at threshold n.
            - P_n: Precision at threshold n.

        Process:
        1. Extract the observed entries from both the test matrix (`M_test`) and
        the predicted matrix (`M_pred`) using `test_mask`.
        2. Compute AUC-ROC using `roc_auc_score`.
        3. Compute Average Precision (AP) using `average_precision_score`.
        4. Compute BEDROC using an exponential weighting scheme.
        5. Return all three scores.

        Args:
            alpha (float): The alpha value for the bedroc metric.
                Default to 160.9.

        Returns:
            tuple[float, float, float]: A tuple containing:
                - AUC-ROC score: Measures overall ranking performance.
                - Average Precision (AP) score: Reflects precision-recall trade-off.
                - BEDROC score: Emphasizes early recognition quality.
        """
        # Extract observed test values and corresponding predictions
        test_values_actual = self.matrix[self.test_mask]
        test_predictions = self.predict_all()[self.test_mask]

        # Compute AUC-ROC
        auc = metrics.roc_auc_score(test_values_actual, test_predictions)

        # Compute Average Precision (AP)
        avg_precision = metrics.average_precision_score(
            test_values_actual, test_predictions
        )

        # Compute BEDROC
        bedroc = bedroc_score(
            y_true=test_values_actual,
            y_pred=test_predictions,
            decreasing=True,
            alpha=alpha,
        )
        return auc, avg_precision, bedroc

    def substep(
        self,
        W_k: np.ndarray,
        tau: float,
        step_size: float,
        grad_f_W_k: np.ndarray,
        tau1: float,
        m: int,
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

        4. Split W_{k+1} into Factor Matrices h_1 and h_2:
           - h_1 = W_{k+1}[:m, :]
           - h_2 = W_{k+1}[m:, :].T

        Args:
            W_k (np.ndarray): Current stacked factor matrices.
            tau (float): Regularization parameter.
            step_size (float): Learning rate for the gradient step.
            grad_f_W_k (np.ndarray): Gradient of the objective function at W_k.
            tau1 (float): Cubic root parameter for step size adjustment.
            m (int): Number of rows in the original matrix.

        Returns:
            Tuple[np.ndarray, float]:
                - Updated stacked matrix W_{k+1}.
                - New loss value f(W_{k+1}).
        """
        step = (np.linalg.norm(W_k, ord="fro") ** 2 + tau) * W_k - (
            step_size * grad_f_W_k
        )
        tau2 = (-2 * (tau**3) - 27 * (np.linalg.norm(step, ord="fro") ** 2)) / 27
        discriminant = (tau2 / 2) ** 2 + (tau1 / 3) ** 3
        discriminant_sqrt = np.sqrt(discriminant, dtype=np.complex128)
        t_k = (tau / 3) + (
            np.power(-tau2 + discriminant_sqrt, 1 / 3, dtype=np.complex128)
            + np.power(-tau2 - discriminant_sqrt, 1 / 3, dtype=np.complex128)
        ).real
        W_k_next = (1 / t_k) * step
        self.h1 = W_k_next[:m, :]
        self.h2 = W_k_next[m:, :].T
        res_norm_next_it = self.calculate_loss()
        return W_k_next, res_norm_next_it

    def run(self, log_freq: int = 10) -> MatrixCompletionResult:
        """
        Performs matrix completion using adaptive step size optimization.

        The optimization process minimizes the objective function:
            f(W) = 0.5 * ||M - M_pred||_F^2 + mu * ||W||_F^2

        where:
            - M: The input matrix (shape: m x n).
            - M_pred: The low-rank approximation, computed as h_1 @ h_2.
            - ||W||_F: The Frobenius norm of the stacked factor matrix W.
            - mu: The regularization parameter.

        Key Steps in the Optimization Process:

        1. Initialization:
           Initialize the factor matrices:
               - h_1 (shape: m x rank),
               - h_2 (shape: rank x n).

        2. Iterative Updates:
           Update the stacked factor matrix W_k = [h_1; h_2^T] using gradients and adaptive
           step sizes:
               - ∇_u = (mask ⊙ (h_1 @ h_2 - M)) @ h_2^T + mu * h_1,
               - ∇_v = (mask.T ⊙ (h_2^T @ h_1^T - M.T)) @ h_1 + mu * h_2^T,
           where ⊙ represents hadamard product.

        3. Dynamic Step Size Adjustment:
           Adjust the step size if the new loss does not satisfy the improvement condition:
               f(W_{k+1}) <= f(W_k) + ∇f(W_k)^T (W_{k+1} - W_k) + η * D_h(W_{k+1}, W_k, τ),
           where:
               - η is the step size/smoothness adjustment parameter,
               - D_h is the difference in h-function values.

        4. Termination:
           Stop when the maximum number of iterations is reached or the loss converges.

        Args:
            log_freq (int, optional): Period at which to log data in Tensorboard.
                Default to 10 (iterations).

        Returns:
            MatrixCompletionResult: A dataclass containing:
                - completed_matrix: The reconstructed matrix (low-rank approximation).
                - loss_history: List of loss values at each iteration.
                - rmse_history: List of RMSE values at each iteration.
                - runtime: Total runtime of the optimization process.
                - iterations: Total number of iterations performed.
        """
        # Start measuring runtime
        start_time = time.time()
        rows, columns = self.matrix.shape
        nb_elements = rows * columns

        # Initialize loss and RMSE history
        res_norm = self.calculate_loss()
        training_loss = res_norm / nb_elements
        testing_loss = self.calculate_rmse()
        loss = [training_loss]
        rmse = [testing_loss]

        # Stack h1 and h2 for optimization
        W_k = np.vstack([self.h1, self.h2.T])
        step_size = 1 / self.smoothness_parameter
        tau = np.linalg.norm(self.matrix, ord="fro") / 3
        tau1 = -(tau**2) / 3

        self.logger.debug(
            "Starting optimization with tau=%f, step_size=%f", tau, step_size
        )

        # Main optimization loop
        iterations_count = 0
        for ith_iteration in range(self.iterations):
            iterations_count = ith_iteration
            inner_loop_it = 0
            flag = 0

            self.logger.debug(
                (
                    "[Main Loop] Iteration %d, Inner Loop %d:"
                    " RMSE=%.6e (testing), Mean Loss=%.6e (training)"
                ),
                ith_iteration,
                inner_loop_it,
                rmse[-1],
                loss[-1],
            )
            # Compute gradients for h1 and h2
            residual = (self.predict_all() - self.matrix)
            residual[~self.train_mask] = 0

            grad_u = residual @ self.h2.T + self.regularization_parameter * self.h1
            grad_v = residual.T @ self.h1 + self.regularization_parameter * self.h2.T
            grad_f_W_k = np.vstack([grad_u, grad_v])

            substep_res = self.substep(W_k, tau, step_size, grad_f_W_k, tau1, rows)
            if substep_res is None:
                break
            W_k_next, res_norm_next_it = substep_res
            # Inner loop to adjust step size
            linear_approx = (grad_f_W_k.T @ (W_k_next - W_k)).sum()
            bregman = bregman_distance(W_k_next, W_k, tau)
            non_euclidean_descent_lemma_cond = (
                res_norm_next_it
                <= res_norm + linear_approx + self.smoothness_parameter * bregman
            )
            if self.writer is not None and ith_iteration % log_freq == 0:
                try:
                    with self.writer.as_default():
                        tf.summary.scalar(
                            name="training_loss", data=training_loss, step=ith_iteration
                        )
                        tf.summary.scalar(
                            name="testing_loss", data=testing_loss, step=ith_iteration
                        )
                        auc, avg_precision, bedroc = self.calculate_auc_bedroc()
                        tf.summary.scalar(name="auc", data=auc, step=ith_iteration)
                        tf.summary.scalar(
                            name="average precision",
                            data=avg_precision,
                            step=ith_iteration,
                        )
                        tf.summary.scalar(
                            name="bedroc top1%", data=bedroc, step=ith_iteration
                        )
                        tf.summary.flush()
                except ValueError as e:
                    self.logger.warning("Tensorboard logging error: %s", e)
            while not non_euclidean_descent_lemma_cond:
                flag = 1
                inner_loop_it += 1
                # Detect overflow and exit if necessary
                if self.rho_increase**inner_loop_it > np.finfo(float).max:
                    self.logger.warning(
                        "[Overflow Detected] Iteration %d, Inner Loop %d: "
                        "Exiting to prevent overflow.",
                        ith_iteration,
                        inner_loop_it,
                    )
                    break
                if inner_loop_it == self.threshold:
                    break
                # Adjust step size
                self.smoothness_parameter *= self.rho_increase**inner_loop_it
                step_size = (1 + self.symmetry_parameter) / self.smoothness_parameter

                substep_res = self.substep(W_k, tau, step_size, grad_f_W_k, tau1, rows)
                if substep_res is None:
                    break
                W_k_next, res_norm_next_it = substep_res
                linear_approx = (grad_f_W_k.T @ (W_k_next - W_k)).sum()
                bregman = bregman_distance(W_k_next, W_k, tau)
                # Detect overflow and exit if necessary
                if self.rho_increase**inner_loop_it > np.finfo(float).max:
                    self.logger.warning(
                        "[Overflow Detected] Iteration %d, Inner Loop %d: "
                        "Exiting to prevent overflow.",
                        ith_iteration,
                        inner_loop_it,
                    )
                    break
                non_euclidean_descent_lemma_cond = (
                    res_norm_next_it
                    <= res_norm + linear_approx + self.smoothness_parameter * bregman
                )
            # Break if loss becomes NaN
            if np.isnan(res_norm_next_it):
                self.logger.warning(
                    "[NaN Loss] Iteration %d: Loss is NaN, exiting loop.", ith_iteration
                )
                break
            if flag == 1:
                # Adjust step size
                self.smoothness_parameter *= self.rho_decrease

            # Update variables for the next iteration
            W_k = W_k_next
            step_size = (1 + self.symmetry_parameter) / self.smoothness_parameter

            training_loss = res_norm_next_it / nb_elements
            testing_loss = self.calculate_rmse()
            loss.append(training_loss)
            rmse.append(testing_loss)
            res_norm = res_norm_next_it

        # Compute runtime
        runtime = time.time() - start_time
        self.logger.debug(
            "[Completion] Optimization finished in %.2f seconds.", runtime
        )
        if self.save_name is not None:
            writer = self.writer
            self.writer = None
            serialize(self, self.save_name)
            self.writer = writer
        training_data = MatrixCompletionResult(
            loss_history=loss,
            iterations=iterations_count,
            rmse_history=rmse,
            runtime=runtime,
        )
        return training_data


def kernel(W: np.ndarray, tau: float) -> float:
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


def bregman_distance(W1: np.ndarray, W2: np.ndarray, tau: float) -> float:
    """
    Computes the Bregman distance:
        D_h = h(W1) - h(W2) - <grad_h(W2), W1 - W2>

    Args:
        W1 (np.ndarray): The first input sparse matrix.
        W2 (np.ndarray): The second input sparse matrix.
        tau (float): Regularization parameter.

    Returns:
        float: The computed Bregman distance.
    """
    h_W1 = kernel(W1, tau)
    h_W2 = kernel(W2, tau)
    grad_h_W2 = (np.linalg.norm(W2, ord="fro") ** 2 + tau) * W2
    linear_approx = (grad_h_W2.T @ (W1 - W2)).sum()
    dist = h_W1 - h_W2 - linear_approx
    return dist
