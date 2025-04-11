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
from typing import Dict, Tuple, Union

import numpy as np
import optuna
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf

from genepriority.compute_models.matrix_completion_result import \
    MatrixCompletionResult
from genepriority.compute_models.smc import MatrixCompletionSession
from genepriority.preprocessing.dataloader import DataLoader
from genepriority.preprocessing.side_information_loader import \
    SideInformationLoader
from genepriority.preprocessing.train_val_test_mask import TrainValTestMasks
from genepriority.trainer.base import BaseTrainer
from genepriority.utils import create_tb_dir


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
        regularization_parameter (float): Regularization parameter.
        iterations (int): Maximum number of optimization iterations.
        symmetry_parameter (float): Regularization parameter for gradient adjustments.
        smoothness_parameter (float): Initial smoothness parameter.
        rho_increase (float): Factor for increasing step size dynamically.
        rho_decrease (float): Factor for decreasing step size dynamically.
        threshold (int): Maximum number of iterations allowed for the inner loop.
        flip_fraction (float, optional): The fraction of observed positive entries
            (ones) in the training mask that will be flipped to negatives (zeros) to simulate
            label noise. Must be between 0 and 1. Default is None, meaning no label flipping
            is performed.
        seed (int): Random seed for reproducibility.
        side_info_loader (SideInformationLoader): Loader for additional side information.
        logger (logging.Logger): Logger instance for tracking progress and debugging.
        tensorboard_dir (Path): The base directory path where
            TensorBoard log files are saved.
        writer (tf.summary.SummaryWriter): A tensorflow log writer.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        path: str,
        seed: int,
        regularization_parameter: float = None,
        iterations: int = None,
        symmetry_parameter: float = None,
        smoothness_parameter: float = None,
        rho_increase: float = None,
        rho_decrease: float = None,
        threshold: int = None,
        flip_fraction: float = None,
        side_info_loader: SideInformationLoader = None,
        tensorboard_dir: Path = None,
    ):
        """
        Initializes the NEGTrainer class with the provided configuration.

        Args:
            dataloader (DataLoader): Data loader containing all necessary training and testing
                data.
            path (str): Directory path where model snapshots and results will be saved.
            seed (int): Random seed to ensure reproducibility.
            regularization_parameter (float, optional): Regularization parameter.
                Defaults to None.
            iterations (int), optional: Maximum number of iterations for the optimization process.
                Defaults to None.
            symmetry_parameter (float, optional): Regularization parameter for gradient adjustments.
                Defaults to None.
            smoothness_parameter (float, optional): Initial smoothness parameter.
                Defaults to None.
            rho_increase (float, optional): Multiplicative factor for increasing step
                size dynamically. Defaults to None.
            rho_decrease (float, optional): Multiplicative factor for decreasing step
                size dynamically. Defaults to None.
            threshold (int, optional): Maximum number of iterations allowed for the inner loop.
                Defaults to None.
            flip_fraction (float, optional): The fraction of observed positive entries
                (ones) in the training mask that will be flipped to negatives (zeros) to simulate
                label noise. Must be between 0 and 1. Default is None, meaning no label flipping
                is performed.
            side_info_loader (SideInformationLoader, optional): Loader for additional side
                information. Defaults to None.
           tensorboard_dir (Path, optional): The base directory path where
                TensorBoard log files are saved. If None, TensorBoard logging is
                disabled. Defaults to None.
        """
        super().__init__(
            dataloader=dataloader,
            path=path,
            seed=seed,
            side_info_loader=side_info_loader,
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        self.regularization_parameter = regularization_parameter
        self.iterations = iterations
        self.symmetry_parameter = symmetry_parameter
        self.smoothness_parameter = smoothness_parameter
        self.rho_increase = rho_increase
        self.rho_decrease = rho_decrease
        self.threshold = threshold
        self.flip_fraction = flip_fraction
        self.tensorboard_dir = tensorboard_dir
        self.writer = None

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
            "regularization_parameter": self.regularization_parameter,
            "iterations": self.iterations,
            "symmetry_parameter": self.symmetry_parameter,
            "smoothness_parameter": self.smoothness_parameter,
            "rho_increase": self.rho_increase,
            "rho_decrease": self.rho_decrease,
            "threshold": self.threshold,
            "flip_fraction": self.flip_fraction,
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
        y_pred = session.predict_all()
        return y_pred

    def create_session(
        self,
        iteration: int,
        train_mask: sp.csr_matrix,
        test_mask: sp.csr_matrix,
        num_latent: int,
        save_name: Union[str, Path],
        side_info: Tuple[sp.csr_matrix, sp.csr_matrix],
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
            side_info (Tuple[sp.csr_matrix, sp.csr_matrix]): The side information
                for both genes and diseases.

        Returns:
            MatrixCompletionSession: A configured session object for model training and evaluation.
        """
        return MatrixCompletionSession(
            **self.neg_session_kwargs,
            rank=num_latent,
            matrix=self.dataloader.omim,
            train_mask=train_mask,
            test_mask=test_mask,
            save_name=str(self.path / f"{iteration}:{save_name}"),
            side_info=side_info,
        )

    def pre_training_callback(
        self,
        session: MatrixCompletionResult,
        run_name: str,
    ):
        """
        Pre training callback used for monitoring and debugging purposes.

        Args:
            session (smurff.MacauSession): Model session to train.
            run_name (str): Custom run name for this training session.
        """
        if self.tensorboard_dir is not None:
            if self.flip_fraction is not None:
                run_name += "-flip_pos_labels"
            run_log_dir = self.tensorboard_dir / run_name
            self.writer = create_tb_dir(run_log_dir)
            with self.writer.as_default():
                data = [
                    [
                        session.rank,
                        session.regularization_parameter,
                        session.symmetry_parameter,
                        session.smoothness_parameter,
                        session.rho_increase,
                        session.rho_decrease,
                        session.threshold,
                    ]
                ]
                columns = [
                    "Rank",
                    "Regularization Parameter",
                    "Symmetry Parameter",
                    "Smoothness Parameter",
                    "Rho Increase Factor",
                    "Rho Decrease Factor",
                    "Inner Loop Threshold",
                ]
                if session.flip_labels is not None:
                    data[0].append(session.flip_labels.fraction)
                    columns.append("Positive Flip Fraction")
                hyperparameter_table = pd.DataFrame(
                    data,
                    columns=columns,
                    index=["Value"],
                ).to_markdown()
                tf.summary.text("hyperparameters", hyperparameter_table, step=0)
                tf.summary.flush()
                session.writer = self.writer

    def post_training_callback(
        self,
        training_status: MatrixCompletionResult,
        session: MatrixCompletionSession,
        test_mask: sp.csr_matrix,
    ):
        """
        Post training callback used for monitoring and debugging purposes.

        Args:
            training_status (MatrixCompletionResult): The results from training.
                Contains loss histories, runtime, etc.
            session (MatrixCompletionSession): The session object of the trained
                matrix completion model, which contains model parameters such as
                rank, regularization parameter, etc.
            test_mask (sp.csr_matrix): Sparse matrix serving as a mask to identify
                the test set entries.
        """
        if self.tensorboard_dir is not None:
            with self.writer.as_default():
                # Log final runtime
                tf.summary.text(
                    name="Run Time",
                    data=f"{training_status.runtime}s",
                    step=training_status.iterations,
                )
                tf.summary.flush()

    def fine_tune(self, n_trials: int, timeout: float, num_latent: int, **kwargs):
        """
        Fine-tunes the hyperparameters for sparse matrix completion using Optuna.

        This method performs hyperparameter optimization for the matrix completion task
        using the Optuna framework. It defines an objective function that evaluates
        the performance of the matrix completion algorithm based on different
        hyperparameter configurations and optimizes for minimal RMSE.

        Args:
            n_trials (int): Number of optimization trials to perform.
            timeout (float): Stop study after the given number of second(s).
                None represents no limit in terms of elapsed time.
            num_latent (int): The number of latent dimensions for the model.

        Returns:
            optuna.study.Study: The study object containing the results of the optimization.
        """

        def objective(
            trial: optuna.Trial,
            rank: int,
            iterations: int,
            threshold: int,
            matrix: sp.csr_matrix,
            train_mask: sp.csr_matrix,
            test_mask: sp.csr_matrix,
        ) -> float:
            regularization_parameter = trial.suggest_float(
                "Regularization parameter for the optimization.", 1e-5, 1e-1, log=True
            )
            symmetry_parameter = trial.suggest_float(
                "Symmetry parameter for the gradient adjustment.",
                1e-8,
                1e-0,
                log=True,
            )
            smoothness_parameter = trial.suggest_float(
                "Initial smoothness parameter.", 0.001, 0.01, step=0.001
            )
            rho_increase = trial.suggest_float(
                "Factor for increasing the step size dynamically.", 2.0, 10.0, step=1.0
            )
            rho_decrease = trial.suggest_float(
                "Factor for decreasing the step size dynamically.", 0.1, 0.9, step=0.1
            )
            session = MatrixCompletionSession(
                regularization_parameter=regularization_parameter,
                iterations=iterations,
                symmetry_parameter=symmetry_parameter,
                smoothness_parameter=smoothness_parameter,
                rho_increase=rho_increase,
                rho_decrease=rho_decrease,
                threshold=threshold,
                rank=rank,
                matrix=matrix,
                train_mask=train_mask,
                test_mask=test_mask,
            )
            try:
                training_status = session.run()
                trial.set_user_attr("rmse on test set", training_status.rmse_history)
                trial.set_user_attr(
                    "loss on training set", training_status.loss_history
                )
            except ValueError as error_msg:
                self.logger.error(error_msg)
                raise optuna.TrialPruned()
            return training_status.rmse_history[-1]

        if not isinstance(self.dataloader.omim1_masks, TrainValTestMasks):
            raise TypeError(
                "No validation set is available. Ensure `omim1_masks` "
                "is an instance of `TrainValTestMasks`."
            )

        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(
                str(self.path / "optuna_journal_storage.log")
            ),
        )

        study = optuna.create_study(
            study_name="SMC hyper-parameters optimization",
            direction="minimize",
            storage=storage,
            **kwargs,
        )
        optuna.logging.enable_propagation()  # Propagate logs to the root logger.
        optuna.logging.disable_default_handler()
        optuna.logging.set_verbosity(optuna.logging.DEBUG)

        matrix = self.dataloader.omim

        self.logger.debug("Set `iter_over_validation` to True.")

        train_mask, _, val_mask = next(iter(self.dataloader.splits))

        study.optimize(
            lambda trial: objective(
                trial,
                num_latent,
                self.iterations,
                self.threshold,
                matrix,
                train_mask,
                val_mask,
            ),
            n_trials=n_trials,
            n_jobs=-1,
            show_progress_bar=True,
            timeout=timeout,
        )

        self.logger.debug("Set `iter_over_validation` to False.")

        return study
