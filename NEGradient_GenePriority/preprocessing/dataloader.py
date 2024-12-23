# pylint: disable=R0902,R0913
"""
DataLoader module
=================

This module contains the `DataLoader` class for preprocessing and preparing gene-disease 
association data for gene prioritization tasks. It includes functionality for handling 
sparse matrices, random splits, and cross-validation folds.

"""
import logging

import numpy as np
import pandas as pd
from NEGradient_GenePriority.preprocessing.preprocessing import (
    compute_statistics,
    convert_dataframe_to_sparse_matrix,
    filter_by_number_of_association,
    sample_zeros,
)
from NEGradient_GenePriority.preprocessing.train_test_masks import TrainTestMasks


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
        zero_sampling_factor (int): Factor determining the number of negative associations
            (zeros) to sample relative to positive associations (ones).
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
        zero_sampling_factor: int = None,
        logger: logging.Logger = None,
    ) -> None:
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
            min_associations (int): Minimum number of associations required for filtering in OMIM2.
            zero_sampling_factor (int, optional): Multiplier for generating negative associations.
            logger (logging.Logger, optional): Logger for debugging. Defaults to a standard logger.
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
                "`train_size` is a fraction, hence it must contained between 0 and 1"
            )
        self.train_size = train_size
        self.min_associations = min_associations

        if logger is None:
            logger = logging.getLogger(__name__)

        self.logger = logger

    def load_data(self) -> pd.DataFrame:
        """
        Load the gene-disease association data from a CSV file.

        Returns:
            pd.DataFrame: The loaded gene-disease data as a Pandas DataFrame.
        """
        self.logger.debug("Loading gene-disease data from %s", self.path)
        return pd.read_csv(self.path)

    def __call__(self, filter_column: str) -> None:
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

    def load_omim1(self, gene_disease: pd.DataFrame) -> None:
        """
        Process and prepare the OMIM1 dataset by creating a sparse matrix, optionally
        sampling negative associations, and generating random splits for training and testing.

        Args:
            gene_disease (pd.DataFrame): DataFrame containing the gene-disease associations.
        """
        self.omim1 = convert_dataframe_to_sparse_matrix(
            gene_disease, shape=(self.nb_genes + 1, self.nb_diseases + 1)
        )
        sparsity = self.omim1.nnz / (self.omim1.shape[0] * self.omim1.shape[1])
        self.logger.debug("Data sparsity: %.2f%%", sparsity * 100)

        if self.zero_sampling_factor is not None:
            self.omim1 = sample_zeros(
                self.omim1, self.zero_sampling_factor, seed=self.seed
            )
            self.logger.debug("Number of 0s: %s", f"{np.sum(self.omim1.data == 0):_}")
            self.logger.debug("Number of 1s: %s", f"{np.sum(self.omim1.data == 1):_}")
            self.logger.debug("Non-zero data: %s", f"{self.omim1.nnz:_}")

        mask = self.omim1.copy()
        mask.data = np.ones(mask.nnz, dtype=bool)

        self.omim1_masks = TrainTestMasks(seed=self.seed)
        self.omim1_masks.split(
            mask, train_size=self.train_size, num_splits=self.num_splits
        )

        self.logger.debug(
            "Combined sparse matrix for OMIM1 created. Shape is %s", self.omim1.shape
        )
        self.logger.debug("Generated random splits for OMIM1 data")

        counts = compute_statistics(self.omim1, self.omim1_masks)
        self.logger.debug("Disease count statistics:\n%s", counts)

    def load_omim2(self, gene_disease: pd.DataFrame, filter_column: str) -> None:
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
        if self.zero_sampling_factor is not None:
            self.omim2 = sample_zeros(
                self.omim2, self.zero_sampling_factor, seed=self.seed
            )
            self.logger.debug("Number of 0s: %s", f"{np.sum(self.omim2.data == 0):_}")
            self.logger.debug("Number of 1s: %s", f"{np.sum(self.omim2.data == 1):_}")
            self.logger.debug("Non-zero data: %s", f"{self.omim2.nnz:_}")

        mask = self.omim2.copy()
        mask.data = np.ones(mask.nnz, dtype=bool)

        self.omim2_masks = TrainTestMasks(seed=self.seed)
        self.omim2_masks.fold(mask, num_folds=self.num_folds)
        self.logger.debug("Created folds for OMIM2 data")

    @property
    def splits(
        self,
    ) -> TrainTestMasks:
        """
        Retrieve the random split masks for the OMIM1 dataset.

        Returns:
            TrainTestMasks
        """
        return self.omim1_masks

    @property
    def folds(
        self,
    ) -> TrainTestMasks:
        """
        Retrieve the cross-validation fold masks for the OMIM2 dataset.

        Returns:
            TrainTestMasks
        """
        return self.omim2_masks
