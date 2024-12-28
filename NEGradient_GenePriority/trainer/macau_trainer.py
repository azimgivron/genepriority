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
from typing import Dict, List, Literal, Union

import numpy as np
import scipy.sparse as sp
import smurff
from NEGradient_GenePriority.preprocessing.dataloader import DataLoader
from NEGradient_GenePriority.preprocessing.side_information_loader import (
    SideInformationLoader,
)
from NEGradient_GenePriority.trainer.base import BaseTrainer
from NEGradient_GenePriority.utils import mask_sparse_containing_0s


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
        logger: logging.Logger = None,
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
            logger (logging.Logger, optional): Logger instance for debug messages.
                If None, a default logger is created.
        """
        super().__init__(
            dataloader=dataloader,
            path=path,
            seed=seed,
            side_info_loader=side_info_loader,
            logger=logger,
        )
        self.num_samples = num_samples
        self.burnin_period = burnin_period
        self.direct = direct
        self.univariate = univariate
        self.save_freq = save_freq
        self.verbose = verbose

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
        session: smurff.MacauSession,
    ) -> np.ndarray:
        """
        Extract predictions from the trained model.

        Args:
            session (smurff.MacauSession): The trained SMURFF session.

        Returns:
            (np.ndarray): The predictions.
        """
        predict_session = session.makePredictSession()
        y_pred = np.mean(predict_session.predict_all(), axis=0)
        return y_pred

    def add_side_info(self, session: smurff.MacauSession):
        """
        Add side information to the SMURFF Macau session.

        Args:
            session (smurff.MacauSession): The SMURFF Macau session to which
                the side information will be added.
        """
        for disease_side_info in self.side_info_loader.disease_side_info:
            # The direct method is only feasible for a small (< 100K) number of
            # features.
            session.addSideInfo(mode=1, Y=disease_side_info, direct=False)
        for gene_side_info in self.side_info_loader.gene_side_info:
            session.addSideInfo(mode=0, Y=gene_side_info, direct=False)

    def create_session(
        self,
        iteration: int,
        matrix: sp.csr_matrix,
        train_mask: sp.csr_matrix,
        test_mask: sp.csr_matrix,
        num_latent: int,
        save_name: Union[str, Path],
    ) -> smurff.MacauSession:
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
            smurff.MacauSession: A configured session object for model training and evaluation.
        """
        training_data = mask_sparse_containing_0s(matrix, train_mask)
        self.log_data("training", training_data)

        testing_data = mask_sparse_containing_0s(matrix, test_mask)
        self.log_data("testing", testing_data)
        return smurff.MacauSession(
            **self.macau_session_kwargs,
            num_latent=num_latent,
            Ytrain=training_data,
            Ytest=testing_data,
            save_name=str(self.path / f"{iteration}:{save_name}"),
            side_info=self.side_info_loader.side_info
            if self.side_info_loader
            else None,
        )

    def log_training_info(self, training_status: List[smurff.Prediction]):
        """
        Logs training information for monitoring and debugging purposes.

        Args:
            training_status (List[smurff.Prediction]): The predictions on
                the test set during training.
        """
