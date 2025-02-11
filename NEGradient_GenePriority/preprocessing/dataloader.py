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
from typing import Union

import numpy as np
import pandas as pd
import scipy.sparse as sp

from NEGradient_GenePriority.preprocessing.preprocessing import (
    compute_statistics,
    convert_dataframe_to_sparse_matrix,
    filter_by_number_of_association,
    sample_zeros,
)
from NEGradient_GenePriority.preprocessing.train_test_masks import TrainTestMasks
from NEGradient_GenePriority.preprocessing.train_val_test_mask import TrainValTestMasks


class DataLoader:
    """
    DataLoader Class

    A utility class to handle the preprocessing and preparation of gene-disease
    association data for gene prioritization tasks. It supports two types of
    datasets (OMIM1 and OMIM2) and provides methods for sparse matrix creation,
    random sampling, and data splitting for cross-validation.

    Attributes:
        nb_genes (int): Number of genes in the dataset.
        nb_diseases (int): Number of diseases in the dataset.
        path (str): Path to the CSV file containing gene-disease associations.
        seed (int): Random seed for reproducibility in sampling and splitting operations.
        num_splits (int): Number of random splits to create for the OMIM1 dataset.
        zero_sampling_factor (int, optional): Factor determining the number of negative
            associations (zeros) to sample relative to positive associations (ones).
        num_folds (int): Number of folds for cross-validation in the OMIM2 dataset.
        omim1 (List[sp.csr_matrix]): List of sparse matrices representing splits for
            OMIM1.
        omim2 (sp.csr_matrix): Sparse matrix combining positive and sampled negative
            associations for OMIM2.
        omim1_masks (List[dict]): Masks for random splits of OMIM1 for
            training and testing.
        omim2_masks (List[dict]): Masks for cross-validation folds of OMIM2.
        logger (logging.Logger): Logger instance for tracking and debugging the
            preprocessing steps.
        min_associations (int): Minimum number of associations required for filtering in OMIM2.
        train_size (float): Proportion of data used for training in splits.
        validation_size (float, optional): Proportion of data used for validation in splits.
    """

    def __init__(
        self,
        nb_genes: int,
        nb_diseases: int,
        path: str,
        seed: int,
        num_splits: int,
        num_folds: int,
        train_size: float,
        min_associations: int,
        validation_size: float = None,
        zero_sampling_factor: int = None,
    ):
        """
        Initialize the DataLoader with configuration settings for data processing.

        Args:
            nb_genes (int): Total number of genes in the dataset.
            nb_diseases (int): Total number of diseases in the dataset.
            path (str): Path to the CSV file containing gene-disease association data.
            seed (int): Random seed to ensure reproducibility in data splitting.
            num_splits (int): Number of random splits to create for the OMIM1 dataset.
            num_folds (int): Number of folds to create for cross-validation in OMIM2 dataset.
            train_size (float): Proportion of the data to be used for training in splits.
            validation_size (float, optional): Proportion of the data to be used for validation in
                splits.
            min_associations (int): Minimum number of associations required for filtering in OMIM2.
            zero_sampling_factor (int, optional): Multiplier for generating negative associations.
        """
        self.nb_genes = nb_genes
        self.nb_diseases = nb_diseases
        self.omim1 = None
        self.omim2 = None
        self.omim1_masks = None
        self.omim2_masks = None
        self.path = path
        self.seed = seed
        self.num_splits = num_splits
        self.zero_sampling_factor = zero_sampling_factor
        self.num_folds = num_folds
        if not 0 <= train_size <= 1:
            raise ValueError(
                "`train_size` must be a fraction between 0 and 1, inclusive. "
                f"Received: {train_size}."
            )

        self.train_size = train_size
        self.validation_size = validation_size
        self.min_associations = min_associations
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @property
    def with_0s(self) -> bool:
        """Whether there 0s in the data.
        
        Returns:
            (bool): True if data contains zeros.
        """
        return self.zero_sampling_factor is not None and self.zero_sampling_factor > 0

    @property
    def iter_over_validation(self) -> bool:
        """
        Get the `iter_over_validation` flag from the `omim1_masks` object.

        Returns:
            bool: The value of `iter_over_validation` from `omim1_masks`.
        """
        if not isinstance(self.omim1_masks, TrainValTestMasks):
            raise TypeError(
                "`omim1_masks` must be an instance of `TrainValTestMasks` to get "
                f"`iter_over_validation`. Current type: {type(self.omim1_masks).__name__}."
            )
        return self.omim1_masks.iter_over_validation

    @iter_over_validation.setter
    def iter_over_validation(self, value: bool):
        """
        Set the `iter_over_validation` flag on the `omim1_masks` object.

        Args:
            value (bool): The value to set for `iter_over_validation`.
        """
        if not isinstance(self.omim1_masks, TrainValTestMasks):
            raise TypeError(
                "`omim1_masks` must be an instance of `TrainValTestMasks` to set "
                f"`iter_over_validation`. Current type: {type(self.omim1_masks).__name__}."
            )
        self.omim1_masks.iter_over_validation = value

    def load_data(self) -> pd.DataFrame:
        """
        Load the gene-disease association data from a CSV file.

        Returns:
            pd.DataFrame: The loaded gene-disease data as a Pandas DataFrame.
        """
        self.logger.debug("Loading gene-disease data from %s", self.path)
        return pd.read_csv(self.path)

    def __call__(self, filter_column: str):
        """
        Entry point for processing the input dataset to generate OMIM1 and OMIM2 datasets
        along with their respective splits and folds.

        Args:
            filter_column (str): Column name in the dataset to use for filtering associations
                for the OMIM2 dataset.
        """
        gene_disease = self.load_data()
        self.logger.debug(
            "Loaded gene-disease data with %d rows and %d columns", *gene_disease.shape
        )
        self.load_omim1(gene_disease)
        self.load_omim2(gene_disease, filter_column=filter_column)

    def load_omim1(self, gene_disease: pd.DataFrame):
        """
        Process and prepare the OMIM1 dataset by creating a sparse matrix, optionally
        sampling negative associations, and generating random splits for training and testing.

        Args:
            gene_disease (pd.DataFrame): DataFrame containing the gene-disease associations.
        """
        self.omim1 = convert_dataframe_to_sparse_matrix(
            gene_disease, shape=(self.nb_genes + 1, self.nb_diseases + 1)
        )

        if self.with_0s:
            self.omim1 = sample_zeros(
                self.omim1, self.zero_sampling_factor, seed=self.seed
            )
        self._log_matrix_stats(self.omim1, "omim1")

        mask = self.omim1.copy()
        mask.data = np.ones(mask.nnz, dtype=bool)

        if self.validation_size is None:
            self.omim1_masks = TrainTestMasks(seed=self.seed)
            self.omim1_masks.split(
                mask, train_size=self.train_size, num_splits=self.num_splits
            )
            self.logger.debug(
                "Random splits created: %.2f%% for training, %.2f%% for testing.",
                self.train_size * 100,
                (1 - self.train_size) * 100,
            )
        else:
            self.omim1_masks = TrainValTestMasks(seed=self.seed)
            self.omim1_masks.split(
                mask,
                train_size=self.train_size,
                num_splits=self.num_splits,
                validation_size=self.validation_size,
            )
            self.logger.debug(
                "Random splits created: %.2f%% for validation, %.2f%% for "
                "training, %.2f%% for testing.",
                self.validation_size * 100,
                self.train_size * (1 - self.validation_size) * 100,
                (
                    1
                    - self.train_size * (1 - self.validation_size)
                    - self.validation_size
                )
                * 100,
            )

        self.logger.debug("Processed OMIM1 dataset. Shape: %s", self.omim1.shape)
        counts = compute_statistics(self.omim1, self.omim1_masks)
        self.logger.debug("Disease count statistics:\n%s", counts)

    def load_omim2(self, gene_disease: pd.DataFrame, filter_column: str):
        """
        Process and prepare the OMIM2 dataset by filtering based on association count,
        creating a sparse matrix, sampling negative associations, and generating
        cross-validation folds.

        Args:
            gene_disease (pd.DataFrame): DataFrame containing the gene-disease associations.
            filter_column (str): Column name used to filter associations based on a threshold.
        """
        self.logger.debug("Filtering gene-disease data by association threshold")
        filtered_gene_disease = filter_by_number_of_association(
            gene_disease, threshold=self.min_associations, col_name=filter_column
        )
        self.logger.debug(
            "Filtered gene-disease data contains %d genes-disease associations",
            len(filtered_gene_disease),
        )

        self.omim2 = convert_dataframe_to_sparse_matrix(
            filtered_gene_disease, shape=(self.nb_genes + 1, self.nb_diseases + 1)
        )

        if self.with_0s:
            self.omim2 = sample_zeros(
                self.omim2, self.zero_sampling_factor, seed=self.seed
            )
        self._log_matrix_stats(self.omim2, "omim2")

        mask = self.omim2.copy()
        mask.data = np.ones(mask.nnz, dtype=bool)

        self.omim2_masks = TrainTestMasks(seed=self.seed)
        self.omim2_masks.fold(mask, num_folds=self.num_folds)

        self.logger.debug("Processed OMIM2 dataset. Shape: %s", self.omim2.shape)

    def _log_matrix_stats(self, matrix: sp.csr_matrix, dataset_name: str):
        """
        Log statistics about the sparse matrix for debugging purposes.

        Args:
            matrix (sp.csr_matrix): Sparse matrix to log statistics for.
            dataset_name (str): The name of the dataset being processed.
        """
        self.logger.debug(
            "Number of 0s in %s: %s",
            dataset_name.upper(),
            f"{np.sum(matrix.data == 0):_}",
        )
        self.logger.debug(
            "Number of 1s in %s: %s",
            dataset_name.upper(),
            f"{np.sum(matrix.data == 1):_}",
        )
        self.logger.debug(
            "Non-zero data in %s: %s", dataset_name.upper(), f"{matrix.nnz:_}"
        )

    @property
    def splits(
        self,
    ) -> Union[TrainTestMasks, TrainValTestMasks]:
        """
        Retrieve the random split masks for the OMIM1 dataset.

        Returns:
            Union[TrainTestMasks, TrainValTestMasks]: The random split masks for OMIM1.
        """
        return self.omim1_masks

    @property
    def folds(
        self,
    ) -> TrainTestMasks:
        """
        Retrieve the cross-validation fold masks for the OMIM2 dataset.

        Returns:
            TrainTestMasks: The cross-validation fold masks for OMIM2.
        """
        return self.omim2_masks
