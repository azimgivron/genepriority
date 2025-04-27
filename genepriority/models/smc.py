# pylint: disable=C0103,R0913,R0914,R0915,R0902,R0903
"""
Structured Matrix Completion Module
====================================

This module implements matrix completion algorithms using adaptive step size optimization.
The API is designed to be straightforward and intuitive, offering configurable parameters
for training, evaluation, and prediction of low-rank approximations.

Key Components:
- BaseMatrixCompletion: An abstract base class that encapsulates the common configuration,
  training routines, and evaluation metrics for matrix completion.
- SideInfoMatrixCompletion: A specialized session that integrates auxiliary side information
  (e.g., gene and disease features) into the matrix completion process.
- StandardMatrixCompletion: A standard implementation for matrix completion without side
    information.

Features:
- Adaptive optimization with dynamic step size adjustment.
- Label noise simulation via controlled flipping of positive entries.
- Comprehensive evaluation using RMSE and loss metrics.
- Optional TensorBoard logging for monitoring training progress.
"""
import abc
import logging
import time
from typing import Tuple, Union

import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from sklearn import metrics

from genepriority.models.early_stopping import EarlyStopping
from genepriority.models.flip_labels import FlipLabels
from genepriority.models.matrix_completion_result import MatrixCompletionResult
from genepriority.models.utils import check_save_name, tsne_plot_to_tensor
from genepriority.utils import calculate_auroc_auprc, serialize


class BaseMatrixCompletion(metaclass=abc.ABCMeta):
    """
    Manages the configuration, training, and evaluation of a matrix completion model.

    The class is designed for scenarios where the objective is to approximate a
    partially observed matrix by a low-rank factorization. It handles the conversion
    of sparse matrices to dense format for internal use, configuration of optimization
    parameters, and logging of progress. Optional mechanisms for label flipping and
    early stopping can also be integrated.

    Attributes:
        matrix (np.ndarray): Input matrix to be approximated. Shape: (n, m),
            where n is the number of genes and m is the number of diseases.
        train_mask (np.ndarray): Boolean mask indicating observed entries in `matrix`
            for training. Shape: (n, m).
        test_mask (np.ndarray): Boolean mask indicating observed entries in `matrix`
            for testing. Shape: (n, m).
        rank (int): The target rank for the low-rank approximation.
        regularization_parameter (float): Regularization parameter used in the optimization
            objective.
        iterations (int): Maximum number of optimization iterations.
        symmetry_parameter (float): Parameter used to adjust gradient symmetry during
            the optimization process.
        smoothness_parameter (float): Initial smoothness parameter for the optimization steps.
        rho_increase (float): Factor used to dynamically increase the optimization step size.
        rho_decrease (float): Factor used to dynamically decrease the optimization step size.
        threshold (int): Maximum iterations allowed for the inner optimization loop.
        h1 (np.ndarray): Left factor matrix in the low-rank approximation.
        h2 (np.ndarray): Right factor matrix in the low-rank approximation.
        logger (logging.Logger): Logger instance for debugging and monitoring training progress.
        writer (tf.summary.SummaryWriter): TensorFlow summary writer for logging training
            summaries.
        seed (int): Seed for reproducible random initialization.
        save_name (str or pathlib.Path or None): File path where the model will be saved.
            If None, the model will not be saved after training.
        flip_labels (FlipLabels or None): Object that simulates label noise by randomly flipping
            a fraction of positive (1) entries to negatives (0) in the training mask.
        early_stopping (EarlyStopping or None): Mechanism for monitoring training loss and
            triggering early termination of training if the performance does not improve.
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
        flip_labels: FlipLabels = None,
        early_stopping: EarlyStopping = None,
    ):
        """
        Initializes the BaseMatrixCompletion instance with the provided configuration
        parameters for matrix approximation.

        Args:
            matrix (sp.csr_matrix): Input matrix to be approximated. Shape: (n, m),
                where n is the number of genes and m is the number of diseases.
            train_mask (sp.csr_matrix): Mask indicating observed entries in `matrix` for training.
                Shape: (n, m).
            test_mask (sp.csr_matrix): Mask indicating observed entries in `matrix` for testing.
                Shape: (n, m).
            rank (int): Desired rank for the low-rank approximation.
            regularization_parameter (float): Regularization parameter for the optimization
                objective.
            iterations (int): Maximum number of optimization iterations.
            symmetry_parameter (float): Parameter for adjusting gradient symmetry during
                optimization.
            smoothness_parameter (float): Initial smoothness parameter for optimization steps.
            rho_increase (float): Multiplicative factor to dynamically increase the optimization
                step size.
            rho_decrease (float): Multiplicative factor to dynamically decrease the optimization
                step size.
            threshold (int): Maximum number of iterations for the inner optimization loop.
            writer (tf.summary.SummaryWriter, optional): TensorFlow summary writer for logging
                training summaries. Defaults to None.
            seed (int, optional): Seed for reproducible random initialization. Defaults to 123.
            save_name (str or pathlib.Path, optional): File path where the model will be saved.
                If set to None, the model will not be saved after training. Defaults to None.
            flip_labels (FlipLabels, optional): Object that simulates label noise by randomly
                flipping a fraction of positive (1) entries to negatives (0) in the training mask.
            early_stopping (EarlyStopping, optional): Early stopping object that implements
                a mechanism for monitoring the validation loss and triggering early termination
                if performance does not improve.
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
        self.h1 = None
        self.h2 = None
        self.flip_labels = flip_labels
        self.early_stopping = early_stopping

        self.save_name = check_save_name(save_name)

        # Set random seed for reproducibility
        np.random.seed(seed)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.writer = writer

    @abc.abstractmethod
    def predict_all(self) -> np.ndarray:
        """
        Computes the reconstructed matrix from the factor matrices h1 and h2.

        Returns:
            np.ndarray: The reconstructed (completed) matrix.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compute_grad_f_W_k(self) -> np.ndarray:
        """Compute the gradients for for each latent as:

        grad_f_W_k = (∇_h1, ∇_h2).T

        Returns:
            np.ndarray: The gradient of the latents (n+m x rank)
        """
        raise NotImplementedError

    def calculate_training_residual(self) -> np.ndarray:
        """
        Compute the training residual from the input matrix M (m x n), the model's prediction
        M_pred and the binary training mask. Optionally, if positive_flip_fraction ’d’ is set,
        a fraction ’d’ of the positive entries (ones) in M (where P is 1) is flipped to 0,
        yielding a modified label matrix L. Otherwise, L = M.

        The training residual R is computed as:
            R = (M_pred - L) ⊙  mask
        where ⊙ represents hadamard product.

        Returns:
            np.ndarray: The residual matrix R (m x n).
        """
        if self.flip_labels is not None:
            labels = self.flip_labels(self.matrix)
        else:
            labels = self.matrix
        residual = self.predict_all() - labels
        residual[~self.train_mask] = 0
        return residual

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
        residuals = self.calculate_training_residual()
        return 0.5 * np.linalg.norm(residuals, ord="fro") ** 2

    def calculate_rmse(self, mask: np.ndarray) -> float:
        """
        Computes the Root Mean Square Error (RMSE).

        The RMSE measures the average deviation between the predicted matrix
        and the ground truth matrix, considering only the observed entries.

        Formula for RMSE:
            RMSE = sqrt( (1 / |Omega|) * sum( (M[i, j] - M_pred[i, j])^2 ) )

        Where:
            - Omega: The set of observed indices in `mask`.
            - M: The matrix containing actual values.
            - M_pred: The predicted matrix (low-rank approximation).

        Process:
        1. Extract the observed entries from both the matrix (`M`)
           and the predicted matrix (`M_pred`) using the `mask`.
        2. Compute the squared differences for the observed entries.
        3. Calculate the mean of these squared differences.
        4. Take the square root of the mean to obtain the RMSE.

        Returns:
            float: The computed RMSE value, representing the prediction error
                   on the observed entries.
        """
        actual_values = self.matrix[mask]
        predictions = self.predict_all()[mask]

        # Compute RMSE
        rmse = np.sqrt(metrics.mean_squared_error(actual_values, predictions))
        return rmse

    def substep(
        self,
        W_k: np.ndarray,
        tau: float,
        step_size: float,
        grad_f_W_k: np.ndarray,
        tau1: float,
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
            tau1 (float): Cubic root parameter for step size adjustment.

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
        m = self.h1.shape[0]
        self.h1 = W_k_next[:m, :]
        self.h2 = W_k_next[m:, :].T
        res_norm_next_it = self.calculate_loss()
        return W_k_next, res_norm_next_it

    def tb_log(
        self, ith_iteration: int, testing_loss: np.ndarray, grad_f_W_k: np.ndarray
    ):
        """
        Logs training and evaluation metrics to Tensorboard for the current iteration.

        This method gathers various performance metrics during the optimization process and logs
        them to Tensorboard. It computes the training RMSE, logs the testing loss, and also logs
        histograms and scalar summaries for different evaluation metrics including AUC and average
        precision. In addition, it logs histograms of the predicted values on both
        the training and  testing sets, as well as the flattened gradient values.

        Args:
            ith_iteration (int): The current iteration index at which the logging is performed.
            testing_loss (np.ndarray): The computed loss value for the testing dataset at the
                current iteration.
            grad_f_W_k (np.ndarray): The gradient of the loss with respect to the model weights
                at the current iteration.
        """
        try:
            with self.writer.as_default():
                training_rmse = self.calculate_rmse(self.train_mask)
                tf.summary.scalar(
                    name="training_loss", data=training_rmse, step=ith_iteration
                )
                tf.summary.scalar(
                    name="testing_loss", data=testing_loss, step=ith_iteration
                )

                # Extract observed test values and corresponding predictions
                test_values_actual = self.matrix[self.test_mask]
                pred = self.predict_all()
                test_predictions = pred[self.test_mask]
                if (self.matrix[self.test_mask] == 0).any():
                    auc, avg_precision = calculate_auroc_auprc(
                        test_values_actual, test_predictions
                    )
                    tf.summary.scalar(name="auc", data=auc, step=ith_iteration)
                    tf.summary.scalar(
                        name="average precision",
                        data=avg_precision,
                        step=ith_iteration,
                    )
                tf.summary.histogram(
                    "Values on test points",
                    test_predictions,
                    step=ith_iteration,
                )
                tf.summary.histogram(
                    "Values on training points",
                    pred[self.train_mask],
                    step=ith_iteration,
                )
                tf.summary.histogram(
                    "Gradient f(W^k)", grad_f_W_k.flatten(), step=ith_iteration
                )
                fig_h1 = tsne_plot_to_tensor(
                    self.h1, color="#E69F00"
                )  # shape: (N x rank) or (g x rank)
                tf.summary.image("t-SNE: Gene embedding", fig_h1, step=ith_iteration)
                fig_h2 = tsne_plot_to_tensor(
                    self.h2.T, color="#009E73"
                )  # shape: (M x rank) or (d x rank)
                tf.summary.image("t-SNE: Disease embedding", fig_h2, step=ith_iteration)
                tf.summary.flush()
        except ValueError as e:
            self.logger.warning("Tensorboard logging error: %s", e)
            raise

    def run(self, log_freq: int = 10) -> MatrixCompletionResult:
        """
        Performs matrix completion using adaptive step size optimization.

        The optimization process minimizes the objective function:
            f(W) = 0.5 * ||M - M_pred||_F^2 + mu * ||W||_F^2

        where:
            - M: The input matrix (shape: m x n).
            - M_pred: The low-rank approximation.
            - ||W||_F: The Frobenius norm of the stacked factor matrix W.
            - mu: The regularization parameter.

        Key Steps in the Optimization Process:

        1. Initialization:
           Initialize the factor matrices:
               - h1,
               - h2

        2. Iterative Updates:
           Update the stacked factor matrix W_k = [h1; h2.T] using gradients and adaptive
           step sizes:
               - ∇_h1,
               - ∇_h2

        3. Dynamic Step Size Adjustment:
           Adjust the step size if the new loss does not satisfy the improvement condition:
               f(W_{k+1}) <= f(W_k) + ∇f(W_k).T (W_{k+1} - W_k) + η * D_h(W_{k+1}, W_k, τ),
           where:
               - η is the step size/smoothness adjustment parameter,
               - D_h is the Bregmann distance.

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
        testing_loss = self.calculate_rmse(self.test_mask)
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
            grad_f_W_k = self.compute_grad_f_W_k()

            substep_res = self.substep(W_k, tau, step_size, grad_f_W_k, tau1)
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
            if self.writer is not None and (
                (ith_iteration + 1) % log_freq == 0 or ith_iteration == 0
            ):
                self.tb_log(ith_iteration, testing_loss, grad_f_W_k)
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

                substep_res = self.substep(W_k, tau, step_size, grad_f_W_k, tau1)
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
            testing_loss = self.calculate_rmse(self.test_mask)
            loss.append(training_loss)
            rmse.append(testing_loss)
            res_norm = res_norm_next_it
            if self.early_stopping is not None and self.early_stopping(
                testing_loss, self.h1, self.h2
            ):
                self.h1, self.h2 = self.early_stopping.best_weights
                self.logger.debug("[Early Stopping] Training interrupted.")
                if ith_iteration % log_freq != 0:
                    self.tb_log(ith_iteration, testing_loss, grad_f_W_k)
                break
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


class SideInfoMatrixCompletion(BaseMatrixCompletion):
    """
    Specialized matrix completion session that incorporates side information.

    Attributes:
        gene_side_info (sp.csr_matrix): Side information for genes (shape: n x g).
        disease_side_info (sp.csr_matrix): Side information for diseases (shape: m x d).
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
                "Dimension 1 of matrix does not match dimension 0 of gene side information."
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

    def predict_all(self) -> np.ndarray:
        """
        Computes the reconstructed matrix from the factor matrices h1 and h2.

        Mathematically, the completed matrix is computed as:
            M_pred = (X @ h1) @ (h2 @ Y.T)

        where:
        - X is the gene feature matrix (shape: n x g),
        - h1 is the left factor matrix (shape: g x rank),
        - h2 is the right factor matrix (shape: rank x d),
        - Y is the disease feature matrix (shape: d x m).

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


class StandardMatrixCompletion(BaseMatrixCompletion):
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
        self.h1 = np.random.randn(num_rows, self.rank)
        self.h2 = np.random.randn(self.rank, num_cols)
        self.logger.debug(
            "Initialized h1 with shape %s and h2 with shape %s",
            self.h1.shape,
            self.h2.shape,
        )

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


MatrixCompletionSessionType = Union[
    "StandardMatrixCompletion", "SideInfoMatrixCompletion"
]


class MatrixCompletionSession:
    """
    Factory class that selects and returns an appropriate matrix completion session
    implementation based on the provided parameters.

    This class exposes a unified API for matrix completion. When creating an instance,
    if the `side_info` parameter is provided, an instance of SideInfoMatrixCompletion is returned;
    otherwise, an instance of StandardMatrixCompletion is instantiated.
    """

    def __new__(cls, *args, side_info=None, **kwargs) -> MatrixCompletionSessionType:
        """
        Creates a new instance of a matrix completion session.

        Args:
            *args: Positional arguments for the underlying session class.
            side_info (tuple or None): A tuple containing side information for genes and diseases.
                If provided, a SideInfoMatrixCompletion instance is created; if None,
                a StandardMatrixCompletion instance is returned.
            **kwargs: Keyword arguments for the underlying session class.

        Returns:
            MatrixCompletionSessionType: An instance of StandardMatrixCompletion or
                SideInfoMatrixCompletion based on the presence of side_info.
        """
        if side_info is None:
            return StandardMatrixCompletion(*args, **kwargs)
        return SideInfoMatrixCompletion(*args, side_info=side_info, **kwargs)
