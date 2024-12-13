import logging
import time
import traceback
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import mean_squared_error


class MatrixCompletion:
    def __init__(
        self,
        A: sp.csr_matrix,
        mask: sp.csr_matrix,
        test: sp.csr_matrix,
        test_mask: sp.csr_matrix,
        k: int,
        mu: float,
        iterations: int,
    ) -> None:
        """
        Initialize the MatrixCompletion class.

        Parameters:
        - A (sp.csr_matrix): Original matrix (can be sparse or dense)
        - mask (sp.csr_matrix): Mask for known entries
        - test (sp.csr_matrix): Test matrix
        - test_mask (sp.csr_matrix): Mask for test entries
        - k (int): Rank of the approximation
        - mu (float): Regularization parameter
        - iterations (int): Number of iterations
        """
        self.A = A
        self.mask = mask
        self.test_matrix = test
        self.test_mask = test_mask
        self.k = k
        self.mu = mu
        self.iterations = iterations

        # Set random seed for reproducibility
        np.random.seed(123)

        # Initialize factor matrices H1 and H2 with random values
        m, n = A.shape
        self.H1 = sp.csr_matrix(np.random.randn(m, k))
        self.H2 = sp.csr_matrix(np.random.randn(k, n))
        logging.debug(
            f"Initialized H1 and H2 with shapes {self.H1.shape} and {self.H2.shape}"
        )

    def calculate_loss(self) -> float:
        """
        Calculate the loss function.

        Returns:
        - float: The calculated loss value
        """
        # Compute the residual matrix using the mask and factorized matrices
        residual = self.mask.multiply(self.A - self.H1.dot(self.H2))
        loss = 0.5 * sp.linalg.norm(residual) ** 2
        logging.debug(f"Calculated loss: {loss}")
        return loss

    def calculate_rmse(self) -> float:
        """
        Calculate the RMSE for the test data.

        Returns:
        - float: The calculated RMSE value
        """
        # Compute the predicted matrix
        predicted_matrix = self.H1.dot(self.H2)
        # Apply the test mask to the predicted matrix
        test_predictions = self.test_mask.multiply(predicted_matrix)

        # Extract actual and predicted values for the test set
        test_values_actual = self.test_matrix[self.test_mask.nonzero()].A1
        test_values_predicted = test_predictions[self.test_mask.nonzero()].A1

        # Compute RMSE using the actual and predicted values
        rmse = np.sqrt(mean_squared_error(test_values_actual, test_values_predicted))
        logging.debug(f"Calculated RMSE: {rmse}")
        return rmse

    def MC_adaptive_2(
        self, lam: float, L: float, rho1: float, rho2: float, threshold: int = 20
    ) -> Tuple[
        sp.csr_matrix,
        sp.csr_matrix,
        sp.csr_matrix,
        List[float],
        int,
        List[float],
        float,
    ]:
        """
        Perform matrix completion using adaptive step size.

        Parameters:
        - lam (float): Regularization parameter
        - L (float): Initial step size
        - rho1 (float): Adjustment parameter for increasing step size
        - rho2 (float): Adjustment parameter for decreasing step size
        - threshold (int): Maximum iterations for inner loop

        Returns:
        - Tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, List[float], int, List[float], float]:
            - Completed matrix
            - H1 matrix
            - H2 matrix
            - Loss history
            - Number of iterations
            - RMSE history
            - Runtime
        """
        # Start measuring runtime
        start_time = time.time()
        m, n = self.A.shape

        # Initialize loss and RMSE history
        self.loss = [self.calculate_loss() / (m * n)]
        self.rmse = [self.calculate_rmse()]

        # Stack H1 and H2 for optimization
        Wk = sp.vstack([self.H1, self.H2.T])
        alpha_k = 1 / L
        tau = sp.linalg.norm(self.A) / 3
        tau1 = -(tau**2) / 3
        t1 = tau1 / 3

        logging.debug(f"Starting optimization with tau={tau}, alpha_k={alpha_k}")

        # Main optimization loop
        for i in range(self.iterations):
            try:
                logging.debug(f"Iteration {i} started")
                # Compute gradients for H1 and H2
                grad_u = (
                    self.mask.multiply((self.H1.dot(self.H2) - self.A)).dot(self.H2.T)
                    + self.mu * self.H1
                )
                grad_v = (
                    self.mask.T.multiply((self.H2.T.dot(self.H1.T) - self.A.T)).dot(
                        self.H1
                    )
                    + self.mu * self.H2.T
                )
                grad_f_Wk = sp.vstack([grad_u, grad_v])

                # Compute the gradient step
                grad = (sp.linalg.norm(Wk, ord="fro") ** 2 + tau) * Wk - (
                    alpha_k * grad_f_Wk
                )
                tau2 = (
                    (
                        -2 * (tau**3)
                        - 27 * (sp.linalg.norm(grad, ord="fro") ** 2)
                    )
                    / 27
                    / 2
                )

                # Compute step sizes
                T1 = -tau2 + np.sqrt(tau2**2 + t1**3)
                T2 = -tau2 - np.sqrt(tau2**2 + t1**3)
                t = (tau / 3) + self.nthr(T1, 3) + self.nthr(T2, 3)

                # Update Wk
                Wk1 = (1 / t) * grad
                self.H1 = sp.csr_matrix(Wk1[:m, :])
                self.H2 = sp.csr_matrix(Wk1[m:, :].T)

                # Compute the new loss
                fk1 = self.calculate_loss()
                logging.debug(f"Iteration {i}: Loss={fk1}")
                j = 0
                flag = 0

                # Inner loop to adjust step size
                while fk1 > self.loss[-1] + grad_f_Wk.T.dot(
                    Wk1 - Wk
                ).sum() + L * self.D_h(Wk1, Wk, tau):
                    flag = 1
                    j += 1
                    logging.debug(f"Adjusting step size, iteration {i}, inner loop {j}")
                    # Detect overflow and exit if necessary
                    if rho1**j > np.finfo(float).max or j == threshold:
                        raise OverflowError("Overflow detected. Exiting loop.")
                    # Adjust step size
                    L *= rho1
                    alpha_k = (1 + lam) / L
                    grad = (sp.linalg.norm(Wk, ord="fro") ** 2 + tau) * Wk - (
                        alpha_k * grad_f_Wk
                    )
                    tau2 = (
                        (
                            -2 * (tau**3)
                            - 27 * (sp.linalg.norm(grad, ord="fro") ** 2)
                        )
                        / 27
                        / 2
                    )
                    T1 = -tau2 + np.sqrt(tau2**2 + (t1 / 3) ** 3)
                    T2 = -tau2 - np.sqrt(tau2**2 + (t1 / 3) ** 3)
                    t = (tau / 3) + self.nthr(T1, 3) + self.nthr(T2, 3)
                    Wk1 = (1 / t) * grad
                    self.H1 = sp.csr_matrix(Wk1[:m, :])
                    self.H2 = sp.csr_matrix(Wk1[m:, :].T)
                    fk1 = self.calculate_loss()

                if flag == 1:
                    # Adjust step size for next iteration
                    L *= rho2

                # Update variables for the next iteration
                Wk = Wk1
                alpha_k = (1 + lam) / L
                self.loss.append(fk1 / (m * n))
                self.rmse.append(self.calculate_rmse())

                # Log iteration metrics
                logging.debug(
                    f"Iteration {i}: RMSE={self.rmse[-1]}, Loss={self.loss[-1]}"
                )

                # Break if loss becomes NaN
                if np.isnan(fk1):
                    logging.warning("NaN encountered in loss, exiting loop.")
                    break

            except Exception as e:
                logging.error(
                    f"Error encountered during iteration {i}: {e}\n{traceback.format_exc()}"
                )
                raise

        # Compute runtime
        runtime = time.time() - start_time
        logging.debug(f"Optimization completed in {runtime:.2f} seconds")
        return self.H1.dot(self.H2), self.H1, self.H2, self.loss, i, self.rmse, runtime

    @staticmethod
    def nthr(a: float, n: int) -> float:
        """
        Calculate the nth root of a number.

        Parameters:
        - a (float): The number to take the nth root of
        - n (int): The degree of the root

        Returns:
        - float: The nth root of a
        """
        return pow(a, 1 / n)

    @staticmethod
    def func_h(W: sp.csr_matrix, tau: float) -> float:
        """
        Compute the h function for a given matrix W and parameter tau.

        Parameters:
        - W (sp.csr_matrix): Input matrix
        - tau (float): Regularization parameter

        Returns:
        - float: The value of the h function
        """
        # Compute the Frobenius norm of W
        fro_norm = sp.linalg.norm(W, ord="fro")
        # Compute the h function value
        return 0.25 * fro_norm**4 + 0.5 * tau * fro_norm**2

    @staticmethod
    def D_h(W1: sp.csr_matrix, W2: sp.csr_matrix, tau: float) -> float:
        """
        Compute the difference in h function values between two matrices.

        Parameters:
        - W1 (sp.csr_matrix): First input matrix
        - W2 (sp.csr_matrix): Second input matrix
        - tau (float): Regularization parameter

        Returns:
        - float: The difference in h function values
        """
        # Compute the h function values for W1 and W2
        hw1 = MatrixCompletion.func_h(W1, tau)
        hw2 = MatrixCompletion.func_h(W2, tau)
        # Compute the gradient of h at W2
        grad_h_w2 = (sp.linalg.norm(W2, ord="fro") ** 2 + tau) * W2
        # Compute the difference in h values
        return hw1 - hw2 - grad_h_w2.T.dot(W1 - W2).sum()
