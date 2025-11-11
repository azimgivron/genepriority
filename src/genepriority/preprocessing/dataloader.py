# pylint: disable=R0902,R0913
"""
DataLoader module
=================

This module contains the `DataLoader` class for preprocessing and preparing gene-disease
association data for gene prioritization tasks. It includes functionality for handling
sparse matrices, random splits, cross-validation folds, and filtering based on association
thresholds.

Features:
- Load and preprocess gene-disease association data from CSV files.
- Create sparse matrices for gene-disease associations.
- Sample negative associations to balance datasets.
- Generate random splits and cross-validation folds for model training and evaluation.
- Filter data based on a minimum number of associations per disease or gene.
- Log detailed statistics for debugging and tracking preprocessing steps.

"""
import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp

from genepriority.preprocessing.preprocessing import (
    compute_statistics, convert_dataframe_to_sparse_matrix, sample_zeros)
from genepriority.preprocessing.train_val_test_mask import TrainValTestMasks


class DataLoader:
    """
    DataLoader Class

    A utility class to handle the preprocessing and preparation of gene-disease
    association data for gene prioritization tasks. It provides methods for
    sparse matrix creation, random sampling and data splitting for cross-validation.

    Attributes:
        nb_genes (int): Number of genes in the dataset.
        nb_diseases (int): Number of diseases in the dataset.
        path (str): Path to the CSV file containing gene-disease associations.
        seed (int): Random seed for reproducibility in sampling and splitting operations.
        zero_sampling_factor (int, optional): Factor determining the number of negative
            associations (zeros) to sample relative to positive associations (ones).
        num_folds (int): Number of folds for cross-validation in the OMIM2 dataset.
        omim (sp.csr_matrix): Sparse matrix combining positive and sampled negative
            associations for OMIM.
        omim_masks (TrainValTestMasks): The masks.
        logger (logging.Logger): Logger instance for tracking and debugging the
            preprocessing steps.
        validation_size (float, optional): Proportion of data used for validation in splits.
    """

    def __init__(
        self,
        nb_genes: int,
        nb_diseases: int,
        path: str,
        seed: int,
        num_folds: int,
        validation_size: float,
        zero_sampling_factor: int = None,
    ):
        """
        Initialize the DataLoader with configuration settings for data processing.

        Args:
            nb_genes (int): Total number of genes in the dataset.
            nb_diseases (int): Total number of diseases in the dataset.
            path (str): Path to the CSV file containing gene-disease association data.
            seed (int): Random seed to ensure reproducibility in data splitting.
            num_folds (int): Number of folds to create for cross-validation in OMIM dataset.
            validation_size (float): Proportion of the data to be used for validation in splits.
            zero_sampling_factor (int, optional): Multiplier for generating negative associations.
        """
        self.nb_genes = nb_genes
        self.nb_diseases = nb_diseases
        self.omim = None
        self.omim_masks = None
        self.path = path
        self.seed = seed
        self.zero_sampling_factor = zero_sampling_factor
        self.num_folds = num_folds
        self.validation_size = validation_size
        self.logger = logging.getLogger(self.__class__.__name__)
        self()

    @property
    def with_0s(self) -> bool:
        """Whether there 0s in the data.

        Returns:
            (bool): True if data contains zeros.
        """
        return self.zero_sampling_factor is not None and self.zero_sampling_factor > 0

    def load_data(self) -> pd.DataFrame:
        """
        Load the gene-disease association data from a CSV file.

        Returns:
            pd.DataFrame: The loaded gene-disease data as a Pandas DataFrame.
        """
        self.logger.debug("Loading gene-disease data from %s", self.path)
        dataframe = pd.read_csv(self.path)
        self.logger.debug(
            "Loaded gene-disease data with %d rows and %d columns", *dataframe.shape
        )
        return dataframe

    def __call__(self):
        """
        Process and prepare the OMIM dataset by creating a sparse matrix, optionally
        sampling negative associations, and generating random splits for training and testing.
        """
        gene_disease = self.load_data()
        self.omim = convert_dataframe_to_sparse_matrix(
            gene_disease, shape=(self.nb_genes, self.nb_diseases)
        )

        if self.with_0s:
            self.omim = sample_zeros(
                self.omim, self.zero_sampling_factor, seed=self.seed
            )
        self._log_matrix_stats(self.omim)
        self.omim_masks = TrainValTestMasks(
            data=self.omim,
            seed=self.seed,
            validation_size=self.validation_size,
            num_folds=self.num_folds,
        )
        train_size = (self.num_folds - 1) / self.num_folds * (1 - self.validation_size)
        self.logger.debug(
            "Random splits created: %.2f%% for validation (50%% for validation, 50%% for "
            "fine tuning), %.2f%% for training, %.2f%% for testing.",
            self.validation_size * 100,
            train_size * 100,
            (1 - train_size - self.validation_size) * 100,
        )
        self.logger.debug(
            "%.2ipts for training (avg), %.2ipts for validation, "
            "%.2ipts for finetuning, %.2i nnz pts and %.2i pts with zeros for testing (avg).",
            np.mean([mask.data.sum() for mask in self.omim_masks.training_masks]),
            self.omim_masks.validation_mask.data.sum(),
            self.omim_masks.finetuning_mask.data.sum(),
            np.mean(
                [
                    self.omim.toarray()[mask].sum()
                    for mask in self.omim_masks.testing_masks
                ]
            ),
            np.mean([mask.sum() for mask in self.omim_masks.testing_masks]),
        )
        self.logger.debug("Processed OMIM dataset. Shape: %s", self.omim.shape)
        counts = compute_statistics(self.omim, self.omim_masks)
        self.logger.debug("Disease count statistics:\n%s", counts)

    def _log_matrix_stats(self, matrix: sp.csr_matrix):
        """
        Log statistics about the sparse matrix for debugging purposes.

        Args:
            matrix (sp.csr_matrix): Sparse matrix to log statistics for.
        """
        self.logger.debug(
            "Number of 0s in OMIM: %s",
            f"{np.sum(matrix.data == 0):_}",
        )
        self.logger.debug(
            "Number of 1s in OMIM: %s",
            f"{np.sum(matrix.data == 1):_}",
        )
        self.logger.debug("Non-zero data in OMIM: %s", f"{matrix.nnz:_}")
        self.logger.debug(
            "Sparsity in OMIM: %s%%",
            f"{matrix.sum()/(matrix.shape[0]*matrix.shape[1])*100:.3f}",
        )
