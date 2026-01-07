# pylint: disable=R0913,R0914,R0902,R0801,E1101
"""
IMCTrainer module
=================

Facilitates the training and evaluation of Inductive Matrix Completion (IMC)-based predictive
models for gene prioritization. The trainer supports cross-validation,
optional label-flipping noise, early stopping, TensorBoard logging, and Optuna-based
hyperparameter tuning.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import optuna
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf
from negaWsi import Result
from negaWsi.early_stopping import EarlyStopping
from negaWsi.flip_labels import FlipLabels

from genepriority.models.imc_session import IMCSession
from genepriority.preprocessing.dataloader import DataLoader
from genepriority.preprocessing.side_information_loader import \
    SideInformationLoader
from genepriority.trainer.base import BaseTrainer
from genepriority.utils import create_tb_dir


class IMCTrainer(BaseTrainer):
    """
    IMCTrainer Class

    Manages IMC model training, evaluation, and prediction while integrating side information,
    early stopping, and TensorBoard logging. Designed to closely follow ``NEGTrainer`` so that the
    same workflows and utilities apply.

    Attributes:
        dataloader (DataLoader): Data loader providing training and testing splits.
        path (str): Directory path for model checkpoints and Optuna artifacts.
        seed (int): Random seed to pass to IMC runs.
        regularization_parameters (Dict[str, float]): Regularization parameters (e.g. λg, λd).
        iterations (int): Maximum number of alternating minimization iterations.
        max_inner_iter (int): Maximum number of iterations for the inner optimization loop.
        svd_init (bool): Whether to initialize factors using SVD warm-start.
        flip_fraction (float): The fraction of observed positive entries
            (ones) in the training mask that will be flipped to negatives (zeros) to simulate
            label noise. Must be between 0 and 1. If None, no label flipping is performed.
        flip_frequency (int): The frequency at which to resample the observed
            positive entries in the training mask to be flipped to negatives.
        patience (int): The number of recent epochs/iterations to consider when
            evaluating the stopping condition. If None, no early stopping is used.
        side_info_loader (SideInformationLoader): Loader with gene/disease features.
        tensorboard_dir (Path): TensorBoard log root directory.
        seed (int): Random seed for reproducibility.
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
        regularization_parameters: Dict[str, float] = None,
        iterations: int = None,
        max_inner_iter: int = None,
        svd_init: bool = False,
        flip_fraction: float = None,
        flip_frequency: int = None,
        patience: int = None,
        side_info_loader: SideInformationLoader = None,
        tensorboard_dir: Path = None,
    ):
        """
        Initialize the IMCTrainer class with the given configuration.

        Args:
            dataloader (DataLoader): Data loader providing training and testing splits.
            path (str): Directory path for model checkpoints and Optuna artifacts.
            seed (int): Random seed to pass to IMC runs.
            regularization_parameters (Dict[str, float], optional): Regularization parameters (e.g. λg, λd).
            iterations (int, optional): Maximum number of alternating minimization iterations.
            max_inner_iter (int, optional): Maximum number of iterations for the inner optimization loop.
            svd_init (bool, optional): Whether to initialize factors using SVD warm-start.
            flip_fraction (float, optional): Fraction of positives to flip to negatives (if any).
            flip_frequency (int, optional): Frequency for regenerating flipped labels.
            patience (int, optional): Patience parameter for early stopping.
            side_info_loader (SideInformationLoader, optional): Loader with gene/disease features.
            tensorboard_dir (Path, optional): TensorBoard log root directory.
        """
        super().__init__(
            dataloader=dataloader,
            path=path,
            seed=seed,
            side_info_loader=side_info_loader,
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        self.regularization_parameters = regularization_parameters
        self.iterations = iterations
        self.max_inner_iter = max_inner_iter
        self.svd_init = svd_init
        if (flip_fraction is None) ^ (flip_frequency is None):
            raise ValueError(
                "Invalid flip-label configuration: "
                "`flip_fraction` and `flip_frequency` must be provided"
                f" together (or both left as None). Got flip_fraction={flip_fraction!r},"
                f" flip_frequency={flip_frequency!r}."
            )
        self.flip_fraction = flip_fraction
        self.flip_frequency = flip_frequency
        self.patience = patience
        self.tensorboard_dir = tensorboard_dir
        self.writer: tf.summary.SummaryWriter | None = None

    @property
    def imc_session_kwargs(self) -> Dict[str, Any]:
        """
        Generate keyword arguments for configuring the IMC session.

        Returns:
            Dict[str, Any]: Parameters forwarded to ``IMCSession``.
        """
        return {
            "regularization_parameters": self.regularization_parameters,
            "iterations": self.iterations,
            "max_inner_iter": self.max_inner_iter,
            "svd_init": self.svd_init,
        }

    def predict(
        self,
        session: IMCSession,
    ) -> np.ndarray:
        """
        Extract predictions from the trained IMC model.

        Args:
            session (IMCSession): The trained session, exposing ``predict_all``.

        Returns:
            np.ndarray: Predicted values as a NumPy array.
        """
        return session.predict_all()

    def create_session(
        self,
        iteration: int,
        train_mask: sp.csr_matrix,
        test_mask: sp.csr_matrix,
        num_latent: int,
        save_name: Union[str, Path],
        side_info: Tuple[np.ndarray, np.ndarray],
    ) -> IMCSession:
        """
        Create a session for model training and evaluation.

        Args:
            iteration (int): Current fold iteration.
            train_mask (sp.csr_matrix): Training mask.
            test_mask (sp.csr_matrix): Validation mask (see ``BaseTrainer`` docstring).
            num_latent (int): Latent dimensionality.
            save_name (Union[str, Path]): Filename (relative to ``self.path``) for serialization.
            side_info (Tuple[np.ndarray, np.ndarray]): Tuple of gene/disease features.

        Returns:
            IMCSession: Configured session for training and evaluation.
        """
        if side_info is None:
            raise ValueError(
                "IMCTrainer requires side information but none was provided. "
                "Ensure `side_info_loader` is configured."
            )
        kwargs = dict(self.imc_session_kwargs)
        if self.flip_fraction is not None:
            ones_indices = np.argwhere(self.dataloader.omim.multiply(train_mask))
            kwargs["flip_labels"] = FlipLabels(
                self.flip_fraction, self.flip_frequency, ones_indices
            )
        if self.patience is not None:
            kwargs["early_stopping"] = EarlyStopping(self.patience)
        return IMCSession(
            **kwargs,
            rank=num_latent,
            matrix=self.dataloader.omim.toarray(),
            train_mask=train_mask.toarray().astype(bool),
            test_mask=test_mask.toarray().astype(bool),
            save_name=str(self.path / f"{iteration}:{save_name}"),
            side_info=side_info,
            seed=iteration
        )

    def pre_training_callback(
        self,
        session: IMCSession,
        run_name: str,
    ):
        """
        Pre training callback used for monitoring and debugging purposes.

        Args:
            session (IMCSession): Model session to train.
            run_name (str): Custom run name for this training session.
        """
        if self.tensorboard_dir is None:
            return

        if self.flip_fraction is not None:
            run_name += "-flip_pos_labels"
        run_log_dir = self.tensorboard_dir / run_name
        self.writer = create_tb_dir(run_log_dir)
        with self.writer.as_default():
            data = [
                [
                    session.rank,
                    session.iterations,
                    session.max_inner_iter,
                    self.svd_init,
                ]
            ]
            columns = [
                "Rank",
                "Iterations",
                "Max Inner Iterations",
                "svd_init",
            ]
            for reg_name, reg_val in session.regularization_parameters.items():
                data[0].append(reg_val)
                columns.append(f"Reg. Param.: {reg_name}")
            if session.flip_labels is not None:
                data[0].append(session.flip_labels.fraction)
                columns.append("Positive Flip Fraction")
            if session.early_stopping is not None:
                data[0].append(session.early_stopping.patience)
                columns.append("Patience")
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
        training_status: Result,
        session: IMCSession,
        test_mask: sp.csr_matrix,
    ):
        """
        Post training callback used for monitoring and debugging purposes.

        Args:
            training_status (Result): Training summary (loss/RMSE/runtime).
            session (IMCSession): Trained model session.
            test_mask (sp.csr_matrix): Validation mask used during training.
        """
        if self.tensorboard_dir is None or self.writer is None:
            return
        with self.writer.as_default():
            tf.summary.text(
                name="Run Time",
                data=f"{training_status.runtime}s",
                step=training_status.iterations,
            )
            tf.summary.flush()

    def fine_tune(
        self,
        n_trials: int,
        timeout: float,
        num_latent: int,
        load_if_exists: bool = False,
        **_,
    ):
        """
        Fine-tunes IMC hyperparameters using Optuna.

        Args:
            n_trials (int): Number of Optuna trials.
            timeout (float): Overall time budget in seconds.
            num_latent (int): Latent dimensionality to evaluate.
            load_if_exists (bool): Reuse previous Optuna storage if present.

        Returns:
            optuna.study.Study: Resulting Optuna study.
        """
        if self.side_info_loader is None:
            raise ValueError(
                "IMC fine-tuning requires side information. "
                "Please provide a `SideInformationLoader`."
            )

        def objective(
            trial: optuna.Trial,
            rank: int,
            iterations: int,
            max_inner_iter: int,
            matrix: sp.csr_matrix,
            side_info: Tuple[np.ndarray, np.ndarray],
            train_mask: sp.csr_matrix,
            test_mask: sp.csr_matrix,
        ) -> float:
            reg = {
                "λg": trial.suggest_float("λg", low=1e-4, high=1e2, log=True),
                "λd": trial.suggest_float("λd", low=1e-4, high=1e2, log=True),
            }
            kwargs: Dict[str, Any] = {
                "regularization_parameters": reg,
                "iterations": iterations,
                "rank": rank,
                "matrix": matrix.toarray(),
                "side_info": side_info,
                "train_mask": train_mask.toarray().astype(bool),
                "test_mask": test_mask.toarray().astype(bool),
                "svd_init": self.svd_init,
                "seed": trial._trial_id,
                "max_inner_iter": max_inner_iter,
            }
            if self.flip_fraction is not None:
                observed = matrix.multiply(train_mask).toarray()
                ones_indices = np.argwhere(observed)
                kwargs["flip_labels"] = FlipLabels(
                    self.flip_fraction,
                    self.flip_frequency,
                    ones_indices,
                )
            if self.patience is not None:
                kwargs["early_stopping"] = EarlyStopping(self.patience)
            session = IMCSession(**kwargs)
            training_status = session.run()
            trial.set_user_attr("rmse on test set", training_status.rmse_history)
            trial.set_user_attr("loss on training set", training_status.loss_history)
            if max(training_status.rmse_history) > 1e8:
                return np.inf
            return training_status.rmse_history[-1]

        storage_path = self.path / "optuna_imc_storage.log"
        load = storage_path.exists()
        if load and not load_if_exists:
            storage_path.unlink()
            load = False

        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(str(storage_path))
        )

        study = optuna.create_study(
            study_name="IMC hyper-parameters optimization",
            direction="minimize",
            storage=storage,
            load_if_exists=load,
        )
        optuna.logging.enable_propagation()
        optuna.logging.disable_default_handler()
        optuna.logging.set_verbosity(optuna.logging.DEBUG)

        matrix = self.dataloader.omim
        train_mask, _, _, val_mask = next(iter(self.dataloader.omim_masks))
        side_info = self.side_info_loader.side_info
        study.optimize(
            lambda trial: objective(
                trial,
                num_latent,
                self.iterations,
                self.max_inner_iter,
                matrix,
                side_info,
                train_mask,
                val_mask,
            ),
            n_trials=n_trials,
            n_jobs=1,
            show_progress_bar=True,
            timeout=timeout,
        )
        return study
