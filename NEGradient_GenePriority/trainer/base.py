# pylint: disable=R0913,R0914,R0902,R0801,E1121
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
from abc import ABCMeta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, Union

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
from NEGradient_GenePriority.utils import serialize


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
    ):
        """
        Initialize the Trainer class with the given configuration.

        Args:
            dataloader (DataLoader): The data loader containing all data for
                training and testing.
            path (str): The path to the directory where model snapshots will be saved.
            seed (int): The random seed for reproducibility.
            side_info_loader (SideInformationLoader, optional): The data loader for the
                side information.
        """
        self.dataloader = dataloader
        self.side_info_loader = side_info_loader
        self.path = path
        self.seed = seed

        self.logger = logging.getLogger(self.__class__.__name__)

    def __call__(
        self,
        latent_dimensions: List[int],
        save_results: bool,
        omim1_filename: str,
        omim2_filename: str,
    ) -> Tuple[Dict[str, Evaluation], Dict[str, Evaluation]]:
        """
        Execute training for different latent dimensions and optionally save results.

        Args:
            latent_dimensions (List[int]): List of latent dimensions to evaluate.
            save_results (bool): Whether to save the results to file.
            omim1_filename (str, optional): Filename for saving OMIM1 results.
            omim2_filename (str, optional): Filename for saving OMIM2 results.

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
            serialize(omim2_results, omim2_results_path)
            serialize(omim1_results, omim1_results_path)
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
            np.ndarray: The predictions generated by the model.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def create_session(
        self,
        iteration: int,
        matrix: sp.csr_matrix,
        train_mask: sp.csr_matrix,
        test_mask: sp.csr_matrix,
        num_latent: int,
        save_name: Union[str, Path],
    ) -> Any:
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
            Any: A session object configured for training and evaluation.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def pre_training_callback(self, session: Any, run_name: str) -> None:
        """
        Invoked before the training process starts for monitoring and debugging.

        Subclasses must implement this method to log or record relevant details
        about the training setup, such as metrics, hyperparameters, or configurations.
        Typical use cases include storing logs in files, databases, or visualization
        tools like TensorBoard.

        Args:
            session (Any): Represents the model's session.
            run_name (str): Unique identifier for the training session, used to
                organize logs or outputs.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def post_training_callback(
        self, training_status: Any, session: Any, run_name: str
    ) -> None:
        """
        Invoked after the training process completes for monitoring and debugging.

        Subclasses must implement this method to log or record the outcomes of
        the training process, such as metrics, losses, and durations. Typical
        use cases include saving results to files, databases, or visualization
        tools like TensorBoard.

        Args:
            training_status (Any): Contains information about the training outcomes,
                such as performance metrics or final losses. The structure depends
                on the specific framework or implementation.
            session (Any): Represents the model's session, including configurations
                used during training.
            run_name (str): Unique identifier for the training session, used to
                organize logs or outputs.
        """
        raise NotImplementedError

    def train_test(
        self,
        matrix: sp.csr_matrix,
        num_latent: int,
        save_name: str,
        desc: str,
        splitted_data: Iterator[Tuple[sp.csr_matrix, sp.csr_matrix]],
    ) -> Evaluation:
        """
        Train and evaluate the model using predefined train-test splits.

        For each split, the method trains the model on training data, evaluates
        its performance on test data, and saves predictions and evaluation metrics.

        Args:
            matrix (sp.csr_matrix): The input matrix containing gene-disease associations.
            num_latent (int): Number of latent dimensions for the model.
            save_name (str): Filename for saving model snapshots.
            desc (str): Description of the dataset (e.g., "split" or "fold").
            splitted_data (Iterator[Tuple[sp.csr_matrix, sp.csr_matrix]]): Iterator
                yielding train-test masks for each split.

        Returns:
            Evaluation: Aggregated evaluation results across all splits.
        """
        results = []
        for i, (train_mask, test_mask) in tqdm(
            enumerate(splitted_data), desc=desc, leave=False
        ):
            self.logger.debug("Initiating training on %s %d", desc, i + 1)

            session = self.create_session(
                i, matrix, train_mask, test_mask, num_latent, save_name
            )

            run_name = f"{desc}{i+1}-latent{num_latent}"
            if self.side_info_loader is None:
                run_name += "-no-side-info"
            if not self.dataloader.with_0s:
                run_name += "-no-0s"
            run_name += f"-{self.__class__.__name__}"
            self.pre_training_callback(session, run_name)

            training_status = session.run()

            self.post_training_callback(training_status, session, run_name)

            y_pred = self.predict(session)
            results.append(Results(y_true=matrix.tocsr(), y_pred=y_pred))
        return Evaluation(results)

    def log_data(self, set_name: str, data: sp.csr_matrix):
        """
        Logs detailed information about a given sparse matrix dataset.

        This method logs the following details for a dataset represented as a
        sparse matrix in Compressed Sparse Row (CSR) format:
        - The number of non-zero elements (`nnz`).
        - The number of elements with a value of 1.
        - The number of elements with a value of 0.

        Args:
            set_name (str): The name of the dataset (e.g., "train", "validation", or "test").
            data (sp.csr_matrix): The dataset represented as a sparse matrix
                                in CSR format.
        """
        self.logger.debug("%s data nnz %s", set_name.capitalize(), f"{data.nnz:_}")
        self.logger.debug(
            "Number of 1s in the %s set %s",
            set_name,
            f"{np.sum(data.data == 1):_}",
        )
        self.logger.debug(
            "Number of 0s in the %s set %s",
            set_name,
            f"{np.sum(data.data == 0):_}",
        )

    def train_test_cross_validation(
        self,
        num_latent: int,
        save_name: str,
    ) -> Evaluation:
        """
        Train and evaluate the model using cross-validation.

        For each fold, the method trains the model on training data, evaluates
        its performance on test data, and saves predictions and evaluation metrics.

        Args:
            num_latent (int): Number of latent dimensions for the model.
            save_name (str): Filename for saving model snapshots.

        Returns:
            Evaluation: Aggregated evaluation results across all folds.
        """
        return self.train_test(
            matrix=self.dataloader.omim2,
            num_latent=num_latent,
            save_name=save_name,
            desc="fold",
            splitted_data=self.dataloader.folds,
        )

    def train_test_splits(
        self,
        num_latent: int,
        save_name: str,
    ) -> Evaluation:
        """
        Train and evaluate the model using predefined train-test splits.

        For each split, the method trains the model on training data, evaluates
        its performance on test data, and saves predictions and evaluation metrics.

        Args:
            num_latent (int): Number of latent dimensions for the model.
            save_name (str): Filename for saving model snapshots.

        Returns:
            Evaluation: Aggregated evaluation results across all splits.
        """
        return self.train_test(
            matrix=self.dataloader.omim1,
            num_latent=num_latent,
            save_name=save_name,
            desc="split",
            splitted_data=self.dataloader.splits,
        )
