# pylint: disable=R0902,R0913
"""
DataLoader module
=================

This module contains the `DataLoader` class for preprocessing and preparing gene-disease 
association data for gene prioritization tasks. It includes functionality for handling 
sparse matrices, random splits, and cross-validation folds.
"""
import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from NEGradient_GenePriority.preprocessing.preprocessing import (
    combine_matrices,
    combine_splits,
    compute_statistics,
    convert_dataframe_to_sparse_matrix,
    create_folds,
    create_random_splits_from_matrices,
    create_random_splits_from_matrix,
    filter_by_number_of_association,
    sample_zeros,
)


class DataLoader:
    """ "
    DataLoader Class

    A utility class to handle the preprocessing and preparation of gene-disease association data
    for gene prioritization tasks. The DataLoader prepares datasets for two types of matrices
    (OMIM1 and OMIM2), including creating sparse matrices, random splits, and folds for
    training and testing.

    Attributes:
        path (str): Path to the CSV file containing gene-disease associations.
        seed (int): Seed for reproducibility in random operations.
        num_splits (int): Number of splits to generate for training and testing data in OMIM1.
        zero_sampling_factor (int): Factor to determine the number of zeros to sample for
            negative associations.
        num_folds (int): Number of folds to generate for cross-validation in OMIM2.
        logger (logging.Logger): Logger instance for logging the processing steps.
        nb_genes (int): The number of genes.
        nb_diseases (int): The number of diseases.

    """

    def __init__(
        self,
        nb_genes: int,
        nb_diseases: int,
        path: str,
        seed: int,
        num_splits: int,
        zero_sampling_factor: int,
        num_folds: int,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize the DataLoader with configuration.

        Args:
            nb_genes (int): The number of genes.
            nb_diseases (int): The number of diseases.
            path (str): Path to the input gene-disease data CSV.
            seed (int): Seed for random number generation.
            num_splits (int): Number of random splits for the data.
            zero_sampling_factor (int): Factor for sampling zero associations.
            num_folds (int): Number of folds for cross-validation.
            logger (Optional[logging.Logger]): Logger for debug messages.
                If None, a default logger is created.
        """
        self.nb_genes = nb_genes
        self.nb_diseases = nb_diseases
        self.omim1 = None
        self.omim2 = None
        self.omim1_splits_indices = None
        self.omim2_folds_indices = None
        self.path = path
        self.seed = seed
        self.num_splits = num_splits
        self.zero_sampling_factor = zero_sampling_factor
        self.num_folds = num_folds

        if logger is None:
            logger = logging.getLogger(__name__)

        self.logger = logger

    def __call__(self, filter_column: str) -> None:
        """
        This method serves as the entry point for processing the input dataset. It
        loads the data from the specified path, preprocesses it, and generates
        the OMIM1 and OMIM2 datasets along with their respective random splits
        and cross-validation folds.

        Args:
            filter_column (str): Column name in the dataset used for filtering
                associations based on a threshold for OMIM2.

        """
        self.logger.debug("Loading gene-disease data from %s", self.path)
        gene_disease = pd.read_csv(self.path)

        self.logger.debug(
            "Loaded gene-disease data with %d rows and %d columns", *gene_disease.shape
        )
        self.load_omim1(gene_disease)
        self.load_omim2(gene_disease, filter_column=filter_column)

    def load_omim1(self, gene_disease: pd.DataFrame) -> None:
        """
        Process the OMIM1 dataset.

        Args:
            gene_disease (pd.DataFrame): Gene-disease association data as a
                DataFrame.
        """
        omim1_1s = convert_dataframe_to_sparse_matrix(
            gene_disease, shape=(self.nb_genes + 1, self.nb_diseases + 1)
        )
        omim1_0s = [
            sample_zeros(omim1_1s, self.zero_sampling_factor, seed=self.seed)
            for _ in range(self.num_splits)
        ]
        self.omim1 = [
            combine_matrices(omim1_1s, omim1_0s_per_split)
            for omim1_0s_per_split in omim1_0s
        ]
        self.logger.debug(
            "Combined sparse matrix for OMIM1 created. Shape is %s", omim1_1s.shape
        )
        omim1_1s_splits_indices = create_random_splits_from_matrix(
            omim1_1s, num_splits=self.num_splits
        )
        omim1_0s_splits_indices = create_random_splits_from_matrices(omim1_0s)
        self.omim1_splits_indices = combine_splits(
            omim1_1s_splits_indices, omim1_0s_splits_indices
        )
        self.logger.debug("Generated random splits for OMIM1 data")

        sparsity = omim1_1s.count_nonzero() / (omim1_1s.shape[0] * omim1_1s.shape[1])
        self.logger.debug("Data sparsity: %.2f%%", sparsity * 100)

        counts = compute_statistics(omim1_1s, omim1_1s_splits_indices)
        self.logger.debug("Disease count statistics:\n%s", counts)

    def load_omim2(self, gene_disease: pd.DataFrame, filter_column: str) -> None:
        """
        This method processes the OMIM2 dataset by filtering associations based
        on the number of occurrences, creating sparse matrices, sampling zeros
        for negative associations, and generating cross-validation folds.

        Args:
            gene_disease (pd.DataFrame): Gene-disease association data as a
                DataFrame containing the full dataset.
            filter_column (str): Column name used to filter gene-disease
                associations based on a threshold of occurrences.
        """
        self.logger.debug("Filtering gene-disease data by association threshold")
        filtered_gene_disease = filter_by_number_of_association(
            gene_disease, threshold=10, col_name=filter_column
        )
        self.logger.debug(
            "Filtered gene-disease data contains %d genes-disease associations",
            len(filtered_gene_disease),
        )

        omim2_1s = convert_dataframe_to_sparse_matrix(
            filtered_gene_disease, shape=(self.nb_genes + 1, self.nb_diseases + 1)
        )
        omim2_0s = sample_zeros(omim2_1s, self.zero_sampling_factor, seed=self.seed)
        self.omim2 = combine_matrices(omim2_1s, omim2_0s)
        self.logger.debug("Combined sparse matrix for OMIM2 created")
        omim2_1s_folds_indices = create_folds(omim2_1s, num_folds=self.num_folds)
        omim2_0s_folds_indices = create_folds(omim2_0s, num_folds=self.num_folds)
        self.omim2_folds_indices = combine_splits(
            omim2_1s_folds_indices, omim2_0s_folds_indices
        )
        self.logger.debug("Created folds for OMIM2 data")

    @property
    def splits(
        self,
    ) -> Tuple[List[sp.csr_matrix], List[sp.csr_matrix], List[np.ndarray]]:
        """
        Get splits for the OMIM1 dataset.

        Returns:
            Tuple[List[sp.csr_matrix], List[sp.csr_matrix], List[np.ndarray]]:
                - List of training matrices (sp.csr_matrix).
                - List of testing matrices (sp.csr_matrix).
                - List of masks for testing data (np.ndarray).
        """
        ys_train = [
            fold.training_indices.get_data(omim1)
            for fold, omim1 in zip(self.omim2_folds_indices, self.omim1)
        ]
        ys_test = [
            fold.testing_indices.get_data(omim1)
            for fold, omim1 in zip(self.omim2_folds_indices, self.omim1)
        ]
        masks = [fold.testing_indices.mask for fold in self.omim2_folds_indices]
        return ys_train, ys_test, masks

    @property
    def folds(
        self,
    ) -> Tuple[List[sp.csr_matrix], List[sp.csr_matrix], List[np.ndarray]]:
        """
        Get folds for the OMIM2 dataset.

        Returns:
            Tuple[List[sp.csr_matrix], List[sp.csr_matrix], List[np.ndarray]]:
                - List of training matrices (sp.csr_matrix).
                - List of testing matrices (sp.csr_matrix).
                - List of masks for testing data (np.ndarray).
        """
        ys_train = [
            fold.training_indices.get_data(self.omim2)
            for fold in self.omim2_folds_indices
        ]
        ys_test = [
            fold.testing_indices.get_data(self.omim2)
            for fold in self.omim2_folds_indices
        ]
        masks = [fold.testing_indices.mask for fold in self.omim2_folds_indices]
        return ys_train, ys_test, masks
