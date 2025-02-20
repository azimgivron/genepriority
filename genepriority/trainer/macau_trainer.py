# pylint: disable=R0913,R0914,R0902,R0801
"""
MACAUTrainer module
=================

Facilitates the training and evaluation of MACAU-based predictive models for gene
prioritization. Supports workflows with train-test splits and cross-validation, 
integrates side information, and computes performance metrics. Evaluation results 
and model snapshots can be saved for reproducibility.
"""

import logging
from pathlib import Path
from typing import Dict, Literal, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf

from genepriority.compute_models.macau import MacauSession
from genepriority.compute_models.matrix_completion_result import MatrixCompletionResult
from genepriority.preprocessing.dataloader import DataLoader
from genepriority.preprocessing.side_information_loader import SideInformationLoader
from genepriority.trainer.base import BaseTrainer
from genepriority.utils import mask_sparse_containing_0s


class MACAUTrainer(BaseTrainer):
    """
    MACAUTrainer Class

    This class extends BaseTrainer to enable the training and evaluation
    MACAU models.

    Attributes:
        dataloader (DataLoader): The data loader containing all data for training
            and testing.
        side_info_loader (SideInformationLoader): The data loader for the
            side information.
        path (str): The path to the directory where model snapshots will be saved.
        num_samples (int): The number of posterior samples to draw during training.
        burnin_period (int): The number of burn-in iterations before collecting
            posterior samples.
        direct (bool): Whether to use a Cholesky or conjugate gradient (CG) solver.
        univariate (bool): Whether to use univariate or multivariate sampling.
        seed (int): The random seed for reproducibility.
        save_freq (int): Frequency at which model state is saved (e.g., every N samples).
        verbose (Literal[0, 1, 2]): Verbosity level of the algorithm
            (0: Silent, 1: Minimal, 2: Detailed).
        logger (logging.Logger): Logger instance for debug and info messages.
        tensorboard_dir (Path): The base directory path where
            TensorBoard log files are saved.
        writer (tf.summary.SummaryWriter): A tensorflow log writer.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        path: str,
        num_samples: int,
        burnin_period: int,
        direct: bool,
        univariate: bool,
        seed: int,
        save_freq: int,
        verbose: Literal[0, 1, 2],
        side_info_loader: SideInformationLoader = None,
        tensorboard_dir: Path = None,
    ):
        """
        Initialize the MACAUTrainer class with the given configuration.

        Args:
            dataloader (DataLoader): The data loader containing all data for
                training and testing.
            path (str): The path to the directory where model snapshots will be saved.
            num_samples (int): The number of posterior samples to draw during training.
            burnin_period (int): The number of burn-in iterations before collecting
                posterior samples.
            direct (bool): Whether to use a Cholesky or CG solver.
            univariate (bool): Whether to use univariate or multivariate sampling.
            seed (int): The random seed for reproducibility.
            save_freq (int): The frequency at which the model state is saved.
            verbose (Literal[0, 1, 2]): The verbosity level of the algorithm.
            side_info_loader (SideInformationLoader, optional): The data loader for the
                side information.
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
        self.num_samples = num_samples
        self.burnin_period = burnin_period
        self.direct = direct
        self.univariate = univariate
        self.save_freq = save_freq
        self.verbose = verbose
        self.tensorboard_dir = tensorboard_dir
        self.writer = None

    @property
    def macau_session_kwargs(self) -> Dict[str, any]:
        """
        Generate keyword arguments for configuring the SMURFF Macau session.

        Returns:
            Dict[str, any]: A dictionary of parameters for the Macau session,
                including solver type, burn-in iterations, posterior sampling
                parameters, and verbosity settings.
        """
        return {
            "is_scarce": True,
            "direct": self.direct,
            "univariate": self.univariate,
            "burnin": self.burnin_period,
            "nsamples": self.num_samples,
            "seed": self.seed,
            "save_freq": self.save_freq,
            "verbose": self.verbose,
        }

    def predict(
        self,
        session: MacauSession,
    ) -> np.ndarray:
        """
        Extract predictions from the trained model.

        Args:
            session (MacauSession): The trained session.

        Returns:
            (np.ndarray): The predictions.
        """
        predict_session = session.makePredictSession()
        y_pred = np.mean(predict_session.predict_all(), axis=0)
        return y_pred

    def create_session(
        self,
        iteration: int,
        matrix: sp.csr_matrix,
        train_mask: sp.csr_matrix,
        test_mask: sp.csr_matrix,
        num_latent: int,
        save_name: Union[str, Path],
    ) -> MacauSession:
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
            MacauSession: A configured session object for model training and evaluation.
        """
        training_data = mask_sparse_containing_0s(matrix, train_mask)
        self.log_data("training", training_data)

        testing_data = matrix.multiply(test_mask)
        self.log_data("testing", testing_data)
        side_info = (
            self.side_info_loader.side_info
            if self.side_info_loader is not None
            else None
        )
        return MacauSession(
            **self.macau_session_kwargs,
            num_latent=num_latent,
            Ytrain=training_data,
            Ytest=testing_data,
            save_name=str(self.path / f"{iteration}:{save_name}"),
            side_info=side_info,
        )

    def pre_training_callback(
        self,
        session: MacauSession,
        run_name: str,
    ):
        """
        Pre training callback used for monitoring and debugging purposes.

        Args:
            session (MacauSession): Model session to train.
            run_name (str): Custom run name for this training session.
        """
        if self.tensorboard_dir is not None:
            run_log_dir = self.tensorboard_dir / run_name
            run_log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = tf.summary.create_file_writer(str(run_log_dir))
            with self.writer.as_default():
                hyperparameter_table = pd.DataFrame(
                    [
                        [
                            session.num_latent,
                            session.nsamples,
                            session.burnin,
                            session.direct,
                            session.univariate,
                        ]
                    ],
                    columns=[
                        "Rank",
                        "Number of samples to keep",
                        "Number of burnin samples to discard",
                        "Cholesky",
                        "Univariate sampling",
                    ],
                    index=["Value"],
                ).to_markdown()
                tf.summary.text("hyperparameters", hyperparameter_table, step=0)
                tf.summary.flush()
            session.writer = self.writer

    def post_training_callback(
        self,
        training_status: MatrixCompletionResult,
        session: MacauSession,
        run_name: str,
    ):
        """
        Post training callback used for monitoring and debugging purposes.

        Args:
            training_status (MatrixCompletionResult): The predictions on
                the test set during training.
            session (MacauSession): Trained model session.
            run_name (str): Custom run name for this training session.
        """
        if self.tensorboard_dir is not None:
            with self.writer.as_default():
                # Log final runtime
                tf.summary.text(
                    name="Run Time", data=f"{training_status.runtime}s", step=0
                )
                tf.summary.flush()
