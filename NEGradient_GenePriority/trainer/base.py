# pylint: disable=R0913,R0914,R0902,R0801
"""
BaseTrainer module
=================

This module facilitates the training and evaluation of predictive models for gene
prioritization. It supports workflows with train-test splits and cross-validation,
integrates preprocessing with `DataLoader` and `SideInformationLoader`, and computes
and logs performance metrics. Evaluation results and model snapshots can be saved
for reproducibility and analysis.
"""
import abc
import logging
import pickle
from abc import ABCMeta
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import scipy.sparse as sp
import smurff
from tqdm import tqdm

from NEGradient_GenePriority.evaluation.evaluation import Evaluation
from NEGradient_GenePriority.evaluation.results import Results
from NEGradient_GenePriority.preprocessing.dataloader import DataLoader
from NEGradient_GenePriority.preprocessing.side_information_loader import (
    SideInformationLoader,
)


class BaseTrainer(metaclass=ABCMeta):
    """
    BaseTrainer Class

    This abstract class implements a template for the training and evaluation of
    predictive models for gene prioritization tasks. It supports evaluation across
    train-test splits or cross-validation folds and integrates preprocessing
    and metrics functionalities to compute and log performance.

    Attributes:
        dataloader (DataLoader): The data loader containing all data for training
            and testing.
        side_info_loader (SideInformationLoader): The data loader for the
            side information.
        path (str): The path to the directory where model snapshots will be saved.
        seed (int): The random seed for reproducibility.
        logger (logging.Logger): Logger instance for debug and info messages.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        path: str,
        seed: int,
        side_info_loader: SideInformationLoader = None,
        logger: logging.Logger = None,
    ) -> None:
        """
        Initialize the Trainer class with the given configuration.

        Args:
            dataloader (DataLoader): The data loader containing all data for
                training and testing.
            path (str): The path to the directory where model snapshots will be saved.
            seed (int): The random seed for reproducibility.
            side_info_loader (SideInformationLoader, optional): The data loader for the
                side information.
            logger (logging.Logger, optional): Logger instance for debug messages.
                If None, a default logger is created.
        """
        self.dataloader = dataloader
        self.side_info_loader = side_info_loader
        self.path = path
        self.seed = seed

        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def __call__(
        self,
        latent_dimensions: List[int],
        save_results: bool,
        omim2_filename: str = "omim2_results.pickle",
        omim1_filename: str = "omim1_results.pickle",
    ) -> Tuple[Dict[str, Evaluation], Dict[str, Evaluation]]:
        """
        Execute training for different latent dimensions and optionally save results.

        Args:
            latent_dimensions (List[int]): List of latent dimensions to evaluate.
            save_results (bool): Whether to save the results to file.
            omim2_filename (str, optional): Filename for saving OMIM2 results.
                Defaults to "omim2_results.pickle".
            omim1_filename (str, optional): Filename for saving OMIM1 results.
                Defaults to "omim1_results.pickle".

        Returns:
            Tuple[Dict[str, Evaluation], Dict[str, Evaluation]]: Evaluation results
                for OMIM1 and OMIM2 datasets.
        """
        omim1_results = {}
        omim2_results = {}
        for num_latent in tqdm(latent_dimensions, desc="Latent dimensions"):
            self.logger.debug(
                "Running training and evaluation for model with %d latent dimensions",
                num_latent,
            )
            omim1_results[f"latent dim={num_latent}"] = self.train_test_splits(
                num_latent=num_latent,
                save_name=f"latent={num_latent}:model-omim1.hdf5",
            )
            omim2_results[
                f"latent dim={num_latent}"
            ] = self.train_test_cross_validation(
                num_latent=num_latent,
                save_name=f"latent={num_latent}:model-omim2.hdf5",
            )
        self.logger.debug("Model trainings and evaluations completed successfully")
        if save_results:
            omim2_results_path = self.path / omim2_filename
            omim1_results_path = self.path / omim1_filename
            save_evaluations(omim2_results, omim2_results_path)
            save_evaluations(omim1_results, omim1_results_path)
            self.logger.debug("Results serialization completed successfully")
        return omim1_results, omim2_results

    @abc.abstractmethod
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
        raise NotImplementedError

    @abc.abstractmethod
    def create_session(
        self,
        iteration: int,
        y_train: sp.csr_matrix,
        y_test_1s: sp.csr_matrix,
        num_latent: int,
        save_name: Union[str, Path],
    ) -> Any:
        """
        Abstract method for creating a session for model training and evaluation.

        Args:
            iteration (int): The current iteration or fold index.
            y_train (sp.csr_matrix): The training matrix, containing both observed and
                unobserved entries.
            y_test_1s (sp.csr_matrix): The test matrix, containing only positive class
                labels.
            num_latent (int): The number of latent dimensions for the model.
            save_name (Union[str, Path]): The base filename or path for saving model snapshots.

        Returns:
            Any: A configured session object for model training and evaluation.
        """
        raise NotImplementedError

    def train_test_cross_validation(
        self,
        num_latent: int,
        save_name: str,
    ) -> Evaluation:
        """
        Trains and evaluates the model using cross-validation.

        For each fold, the method trains the model on the training data, evaluates
        its performance on the test set, and saves the predictions and evaluation
        metrics.

        Args:
            num_latent (int): Number of latent dimensions for the model.
            save_name (str): Base filename for saving model snapshots.

        Returns:
            Evaluation: An object containing aggregated evaluation results across all folds.
        """
        results = []
        for i, (y_train, y_test_1s) in tqdm(
            enumerate(zip(*self.dataloader.folds)), desc="Folds", leave=False
        ):
            self.logger.debug("Initiating training on fold %d", i + 1)
            self.logger.debug(
                "Number of 1s in the training set %s",
                len(y_train.data[y_train.data == 1]),
            )
            self.logger.debug(
                "Number of 0s in the training set %s",
                len(y_train.data[y_train.data == 0]),
            )
            session = self.create_session(i, y_train, y_test_1s, num_latent, save_name)
            session.run()

            y_pred = self.predict(session)
            results.append(Results(y_true=self.dataloader.omim2.tocsr(), y_pred=y_pred))
        return Evaluation(results)

    def train_test_splits(
        self,
        num_latent: int,
        save_name: str,
    ) -> Evaluation:
        """
        Trains and evaluates the model using predefined train-test splits.

        For each split, the method trains the model on the training data, evaluates
        its performance on the test set, and saves the predictions and evaluation metrics.

        Args:
            num_latent (int): Number of latent dimensions for the model.
            save_name (str): Base filename for saving model snapshots.

        Returns:
            Evaluation: An object containing aggregated evaluation results across
                all splits.
        """
        results = []
        for i, (y_train, y_test_1s) in tqdm(
            enumerate(zip(*self.dataloader.splits)), desc="Splits", leave=False
        ):
            self.logger.debug("Initiating training on split %d", i + 1)
            self.logger.debug(
                "Number of 1s in the training set %s",
                len(y_train.data[y_train.data == 1]),
            )
            self.logger.debug(
                "Number of 0s in the training set %s",
                len(y_train.data[y_train.data == 0]),
            )

            session = self.create_session(i, y_train, y_test_1s, num_latent, save_name)
            session.run()

            y_pred = self.predict(session)
            results.append(
                Results(y_true=self.dataloader.omim1[i].tocsr(), y_pred=y_pred)
            )
        return Evaluation(results)


def save_evaluations(results: Dict[str, Evaluation], output_path: str):
    """
    Save evaluation results to a file in binary format using pickle.

    Args:
        results (Dict[str, Evaluation]): A dictionary where keys are descriptive strings
            (e.g., latent dimensions or other identifiers) and values are `Evaluation` objects
            containing evaluation metrics and results.
        output_path (str): The file path where the results should be saved.
    """
    with open(output_path, "wb") as handler:
        pickle.dump(results, handler)
