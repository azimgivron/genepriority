# pylint: disable=R0913,R0914
"""
Trainer module
=================

This module orchestrates model training and evaluation across train-test splits
or cross-validation folds. It integrates preprocessing and metrics functionalities
to compute and log performance metrics. Key components include the `Evaluation`
data class to store metrics and functions like `extract_results` and
`train_test_cross_validation` to streamline the evaluation process.
"""

import logging
import pickle
from typing import List, Literal, Optional, Dict, Tuple

import numpy as np
import smurff

from NEGradient_GenePriority.evaluation.evaluation import Evaluation
from NEGradient_GenePriority.evaluation.results import Results

class Trainer:
    def __init__(
        self,
        path: str,
        num_samples: int,
        burnin_period: int,
        direct: bool,
        univariate: bool,
        num_latent: int,
        seed: int,
        save_freq: int,
        verbose: Literal[0, 1, 2],
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the Trainer class.

        Args:
            path (str): The path to the directory where the snapshots will be saved.
            num_samples (int): The number of posterior samples to draw during training.
            burnin_period (int): The number of burn-in iterations before collecting posterior samples.
            direct (bool): Whether to use a Cholesky instead of conjugate gradient (CG) solver.
                Cholesky is recommended up to $dim(F_e) \approx 20,000$.
            univariate (bool): Whether to use univariate or multivariate sampling.
                Multivariate sampling requires computing the whole precision matrix
                $D \cdot F_e \times D \cdot F_e$ where $D$ is the latent vector size and $F_e$
                is the dimensionality of the entity features. If True, it uses a Gibbs sampler.
            num_latent (int): The number of latent factors to be used by the model.
            seed (int): The random seed to ensure reproducibility in stochastic operations.
            save_freq (int): The frequency at which the model state is saved (e.g., every N samples).
            verbose (Literal[0,1,2]): The verbosity level of the algorithm (0: Silent, 1: Minimal, 2: Detailed).
            logger (Optional[logging.Logger], optional): Logger instance for debug and info messages. Defaults to None.
        """
        self.path = path
        self.num_samples = num_samples
        self.burnin_period = burnin_period
        self.direct = direct
        self.univariate = univariate
        self.num_latent = num_latent
        self.seed = seed
        self.save_freq = save_freq
        self.verbose = verbose

        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    @property
    def macau_session_kwargs(self) -> Dict[str, any]:
        """Generate keyword arguments for the Macau session.

        Returns:
            Dict[str, any]: A dictionary of parameters for configuring the Macau session.
        """
        return {
            "is_scarce": True,
            "direct": self.direct,
            "univariate": self.univariate,
            "num_latent": self.num_latent,
            "burnin": self.burnin_period,
            "nsamples": self.num_samples,
            "seed": self.seed,
            "save_freq": self.save_freq,
            "verbose": self.verbose,
        }

    def train(self) -> None:
        """Train the model and serialize results to files."""
        omim1_results, omim2_results = self()
        omim2_results_path = self.path / "omim2_results.pickle"
        omim1_results_path = self.path / "omim1_results.pickle"
        self.to_file(omim2_results, omim2_results_path)
        self.to_file(omim1_results, omim1_results_path)
        self.logger.debug("Results serialization completed successfully")

    def __call__(self, latent_dimensions: List[int]) -> Tuple[Dict[str, Evaluation], Dict[str, Evaluation]]:
        """Execute training for different latent dimensions.

        Args:
            latent_dimensions (List[int]): List of latent dimensions to evaluate.

        Returns:
            Tuple[Dict[str, Evaluation], Dict[str, Evaluation]]: Results for OMIM1 and OMIM2 datasets.
        """
        omim1_results = {}
        omim2_results = {}
        for num_latent in latent_dimensions:
            self.logger.debug("Running MACAU for %d latent dimensions", num_latent)
            self.logger.debug("Starting training on OMIM1")
            omim1_results[f"latent dim={num_latent}"] = self.train_test_splits(
                save_name=f"latent={num_latent}:macau-omim1.hdf5",
            )
            self.logger.debug("Starting training on OMIM2")
            omim2_results[f"latent dim={num_latent}"] = self.train_test_cross_validation(
                save_name=f"latent={num_latent}:macau-omim2.hdf5",
            )
        self.logger.debug("MACAU session completed successfully")
        return omim1_results, omim2_results

    def to_file(self, results: Dict[str, Evaluation], output_path: str) -> None:
        """Serialize results to a file.

        Args:
            results (Dict[str, Evaluation]): The results to serialize.
            output_path (str): Path to the output file.
        """
        with open(output_path, "wb") as handler:
            pickle.dump(results, handler)

    def predict(
        self,
        session: smurff.MacauSession,
        mask: np.ndarray,
    ) -> Results:
        """Extract predictions from the trained model for the specified testing indices.

        Args:
            session (smurff.MacauSession): The smurff session.
            mask (np.ndarray): Mask for keeping test values only.

        Returns:
            Results: Contains `y_true` (ground truth) and `y_pred` (predictions).
        """
        predict_session = session.makePredictSession()
        y_pred = np.mean(predict_session.predict_all(), axis=0)[mask]
        return y_pred 

    def train_test_cross_validation(
        self,
        save_name: str,
    ) -> Evaluation:
        r"""
        Train and evaluate the model across multiple folds for cross-validation.

        This function performs cross-validation using a provided list of train-test
        folds (`folds_list`). For each fold, the algorithm trains a model, makes predictions,
        and extracts performance metrics.

        Args:
            save_name (str): The base filename to use when saving model snapshots.

        Returns:
            Evaluation: The results of the evaluation.
        """
        results = []
        for i, (y_train, y_true, mask) in enumerate(self.dataloader.folds):
            self.logger.debug("Initiating training on fold %s", i + 1)
            session = smurff.MacauSession(
                **self.macau_session_kwargs,
                Ytrain=y_train,
                save_name=str(self.path / f"{i}:{save_name}"),
            )
            session.run()  # Run training
            self.logger.debug("Training on fold %s ended successfully.", i + 1)

            y_pred = self.predict(session, mask)
            result = y_pred = Results(y_true, y_pred)

            self.logger.debug("Evaluation on fold %s ended successfully.", i + 1)
            results.append(result)
        return Evaluation(results)

    def train_test_splits(
        self,
        save_name: str,
    ) -> Evaluation:
        r"""
        Train and evaluate the model across multiple splits.

        For each split, the algorithm trains a model, makes predictions,
        and extracts performance metrics.

        Args:
            save_name (str): The base filename to use when saving model snapshots.

        Returns:
            Evaluation: The results of the evaluation.
        """
        results = []
        for i, (y_train, y_true, mask) in enumerate(self.dataloader.splits):
            self.logger.debug("Initiating training on split %s", i + 1)

            session = smurff.MacauSession(
                **self.macau_session_kwargs,
                Ytrain=y_train,
                save_name=str(self.path / f"{i}:{save_name}"),
            )
            session.run()  # Run training
            self.logger.debug("Training on split %s ended successfully.", i + 1)

            y_pred = self.predict(session, mask)
            result = y_pred = Results(y_true, y_pred)

            self.logger.debug("Evaluation on split %s ended successfully.", i + 1)
            results.append(result)
        return Evaluation(results)
