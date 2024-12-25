# pylint: disable=R0913,R0914,R0902,R0801
"""
NEGTrainer module
=================

Facilitates the training and evaluation of Non-Euclidean Gradient (NEG)-based predictive models
for gene prioritization. The trainer supports workflows involving train-test splits and 
cross-validation, with optional integration of side information. It provides methods for training
models, generating predictions, and computing evaluation metrics.
"""

import logging
from pathlib import Path
from typing import Dict, Union

import numpy as np
import optuna
import scipy.sparse as sp

from NEGradient_GenePriority.compute_models.smc import (
    MatrixCompletionResult,
    MatrixCompletionSession,
)
from NEGradient_GenePriority.preprocessing.dataloader import DataLoader
from NEGradient_GenePriority.preprocessing.side_information_loader import (
    SideInformationLoader,
)
from NEGradient_GenePriority.trainer.base import BaseTrainer
from NEGradient_GenePriority.utils import mask_sparse_containing_0s


class NEGTrainer(BaseTrainer):
    """
    NEGTrainer Class

    Facilitates the training, evaluation, and prediction of Non-Euclidean Gradient (NEG)-based
    models for gene prioritization. This class integrates side information, supports both
    cross-validation and predefined train-test splits, and computes evaluation metrics for
    performance monitoring.

    Attributes:
        dataloader (DataLoader): The data loader containing training and testing data.
        path (str): Directory path where model snapshots and results will be saved.
        optim_reg (float): Regularization parameter for the optimization.
        iterations (int): Maximum number of optimization iterations.
        lam (float): Regularization parameter for gradient adjustments.
        step_size (float): Initial step size for optimization updates.
        rho_increase (float): Factor for increasing step size dynamically.
        rho_decrease (float): Factor for decreasing step size dynamically.
        threshold (int): Maximum number of iterations allowed for the inner loop.
        seed (int): Random seed for reproducibility.
        side_info_loader (SideInformationLoader, optional): Loader for additional side information.
        logger (logging.Logger): Logger instance for tracking progress and debugging.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        path: str,
        optim_reg: float,
        iterations: int,
        lam: float,
        step_size: float,
        rho_increase: float,
        rho_decrease: float,
        threshold: int,
        seed: int,
        side_info_loader: SideInformationLoader = None,
        logger: logging.Logger = None,
    ) -> None:
        """
        Initializes the NEGTrainer class with the provided configuration.

        Args:
            dataloader (DataLoader): Data loader containing all necessary training and testing
                data.
            path (str): Directory path where model snapshots and results will be saved.
            optim_reg (float): Regularization parameter for optimization.
            iterations (int): Maximum number of iterations for the optimization process.
            lam (float): Regularization parameter for gradient adjustments.
            step_size (float): Initial step size for the optimization.
            rho_increase (float): Multiplicative factor for increasing step size dynamically.
            rho_decrease (float): Multiplicative factor for decreasing step size dynamically.
            threshold (int): Maximum number of iterations allowed for the inner loop.
            seed (int): Random seed to ensure reproducibility.
            side_info_loader (SideInformationLoader, optional): Loader for additional side
                information. Defaults to None.
            logger (logging.Logger, optional): Logger instance for tracking progress.
                Defaults to None.
        """
        super().__init__(
            dataloader=dataloader,
            path=path,
            seed=seed,
            side_info_loader=side_info_loader,
            logger=logger,
        )
        self.optim_reg = optim_reg
        self.iterations = iterations
        self.lam = lam
        self.step_size = step_size
        self.rho_increase = rho_increase
        self.rho_decrease = rho_decrease
        self.threshold = threshold

    @property
    def neg_session_kwargs(self) -> Dict[str, any]:
        """
        Generates keyword arguments for configuring the NEG session.

        Returns:
            Dict[str, any]: A dictionary containing the parameters required
                to initialize a `MatrixCompletionSession`, such as
                regularization, and step size.
        """
        return {
            "optim_reg": self.optim_reg,
            "iterations": self.iterations,
            "lam": self.lam,
            "step_size": self.step_size,
            "rho_increase": self.rho_increase,
            "rho_decrease": self.rho_decrease,
            "threshold": self.threshold,
        }

    def predict(
        self,
        session: MatrixCompletionSession,
    ) -> np.ndarray:
        """
        Extracts predictions from the trained NEG model.

        Args:
            session (MatrixCompletionSession): The trained session
                containing the model's factor matrices.

        Returns:
            np.ndarray: Predicted values as a NumPy array, computed by
                averaging over the reconstructed matrix.
        """
        y_pred = np.mean(session.predict_all(), axis=0)
        return y_pred

    def create_session(
        self,
        iteration: int,
        matrix: sp.csr_matrix,
        train_mask: sp.csr_matrix,
        test_mask: sp.csr_matrix,
        num_latent: int,
        save_name: Union[str, Path],
    ) -> MatrixCompletionSession:
        """
        Create a session for model training and evaluation.

        Args:
            iteration (int): The current iteration or fold index.
            matrix (sp.csr_matrix): The data matrix.
            train_mask (sp.csr_matrix): The training mask.
            test_mask (sp.csr_matrix): The test mask.
            num_latent (int): The number of latent dimensions for the model.
            save_name (Union[str, Path]): Filename or path for saving model snapshots.

        Returns:
            MatrixCompletionSession: A configured session object for model training and evaluation.
        """
        training_data = mask_sparse_containing_0s(matrix, train_mask)
        self.log_data("training", training_data)

        testing_data = mask_sparse_containing_0s(matrix, test_mask)
        self.log_data("testing", testing_data)
        return MatrixCompletionSession(
            **self.neg_session_kwargs,
            rank=num_latent,
            matrix=matrix,
            train_mask=train_mask,
            test_mask=test_mask,
            save_name=str(self.path / f"{iteration}:{save_name}"),
        )

    def log_training_info(training_status: MatrixCompletionResult):
        """
        Logs training information for monitoring and debugging purposes.

        Args:
            training_status (MatrixCompletionResult): The results from
                training.
        """
        pass

    def fine_tune(
        self,
        matrix: sp.csr_matrix,
        train_mask: sp.csr_matrix,
        test_mask: sp.csr_matrix,
        num_latent: int,
        load_if_exists: bool,
        n_trials: int,
    ):
        """
        Fine-tunes the hyperparameters for sparse matrix completion using Optuna.

        This method performs hyperparameter optimization for the matrix completion task 
        using the Optuna framework. It defines an objective function that evaluates 
        the performance of the matrix completion algorithm based on different 
        hyperparameter configurations and optimizes for minimal RMSE.

        Args:
            matrix (sp.csr_matrix): The sparse matrix to be completed.
            train_mask (sp.csr_matrix): Sparse matrix representing the training mask.
            test_mask (sp.csr_matrix): Sparse matrix representing the testing mask.
            num_latent (int): Number of latent factors for the matrix completion model.
            load_if_exists (bool): If True, loads an existing study with the same name.
            n_trials (int): Number of optimization trials to perform.

        Returns:
            optuna.study.Study: The study object containing the results of the optimization.
        """
        def objective(trial: optuna.Trial) -> float:
            optim_reg = trial.suggest_float("optim_reg", 0, 1)
            iterations = trial.suggest_int("iterations", 0, 10)
            lam = trial.suggest_float("lam", 0, 1)
            step_size = trial.suggest_float("step_size", 0, 1)
            rho_increase = trial.suggest_float("rho_increase", 0, 1)
            rho_decrease = trial.suggest_float("rho_decrease", 0, 1)
            threshold = trial.suggest_int("threashold", 0, 10)
            session = MatrixCompletionSession(
                optim_reg=optim_reg,
                iterations=iterations,
                lam=lam,
                step_size=step_size,
                rho_increase=rho_increase,
                rho_decrease=rho_decrease,
                threshold=threshold,
                rank=num_latent,
                matrix=matrix,
                train_mask=train_mask,
                test_mask=test_mask,
            )
            training_status = session.run()
            trial.set_user_attr("rmse on test set", training_status.rmse_history)
            trial.set_user_attr("loss on training set", training_status.loss_history)
            return training_status.rmse_history[-1]

        study = optuna.create_study(
            study_name="SMC hyper-parameters optimization",
            direction="minimize",
            load_if_exists=load_if_exists,
        )
        study.optimize(
            objective, n_trials=n_trials, n_jobs=-1, show_progress_bar=True
        )
        return study
