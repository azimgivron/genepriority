"""
Baseline Module
================

This module defines BaselineTrainer, a subclass of BaseTrainer, which manages
initialization, training, prediction, and callbacks for baseline models using
BaselineSession.
"""

from pathlib import Path
from typing import Any, Tuple, Union

import numpy as np
import scipy.sparse as sp

from genepriority.models.baseline import BaselineSession
from genepriority.preprocessing.dataloader import DataLoader
from genepriority.trainer.base import BaseTrainer


class BaselineTrainer(BaseTrainer):
    """
    Trainer for baseline gene–disease association models.

    BaselineTrainer wraps BaselineSession to handle the end-to-end training,
    prediction, and logging of baseline gene–disease association methods.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        path: str,
        seed: int,
    ):
        """
        Initialize the BaselineTrainer.

        Args:
            dataloader (DataLoader): Data loader providing gene–disease data and masks.
            path (str): Directory path where model checkpoints and logs will be saved.
            seed (int): Random seed for reproducibility of training.
        """
        super().__init__(dataloader, path, seed)

    def predict(
        self,
        session: Any,
    ) -> np.ndarray:
        """
        Extract predictions from the trained model session.

        Args:
            session (Any): A trained BaselineSession instance.

        Returns:
            np.ndarray: Array of predicted scores for all gene–disease pairs.
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
    ) -> BaselineSession:
        """
        Create and configure a BaselineSession for a specific training fold.

        Args:
            iteration (int): Current fold or iteration index.
            train_mask (sp.csr_matrix): Sparse mask indicating training entries.
            test_mask (sp.csr_matrix): Sparse mask indicating test entries.
            num_latent (int): Number of latent dimensions for the baseline model.
            save_name (Union[str, Path]): Filename or path prefix for saving the session.
            side_info (Tuple[sp.csr_matrix, sp.csr_matrix]): Optional side information
                for genes and diseases (unused by BaselineSession).

        Returns:
            BaselineSession: Configured session ready for training and evaluation.
        """
        return BaselineSession(
            self.dataloader.omim,
            self.dataloader.zero_sampling_factor,
            self.seed,
            save_name=str(self.path / f"{iteration}:{save_name}"),
        )

    def pre_training_callback(self, session: Any, run_name: str):
        """
        Callback invoked before training starts to set up logging or monitoring.

        Subclasses should implement logging of hyperparameters, data statistics,
        or any other setup actions for observability.

        Args:
            session (Any): The BaselineSession instance to be trained.
            run_name (str): Unique identifier for this training run.
        """

    def post_training_callback(
        self, training_status: Any, session: Any, test_mask: sp.csr_matrix
    ):
        """
        Callback invoked after training completes to record results and metrics.

        Subclasses should implement persistence of evaluation metrics, model
        snapshots, or dashboards for analysis.

        Args:
            training_status (Any): Object containing training outcomes (loss,
                metrics, durations, etc.).
            session (Any): The trained BaselineSession instance.
            test_mask (sp.csr_matrix): Sparse mask indicating test entries used for
                final evaluation.
        """
