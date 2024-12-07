"""
Side Information Loader Module
==============================

Provides utilities for loading, processing, and converting side information datasets 
into sparse matrix representations. The module includes methods for handling gene and 
disease-related data, adding implicit information, and converting datasets to COO sparse matrices.
"""
import logging
from typing import List

import numpy as np
import pandas as pd
import scipy.sparse as sp


class SideInformationLoader:
    """
    A class to handle loading, processing, and converting side information datasets
    into sparse matrix representations.

    Attributes:
        gene_side_information (List[sp.coo_matrix]): List of COO matrices for
            gene-side information.
        disease_side_information (List[sp.coo_matrix]): List of COO matrices for
            disease-side information.
        logger (logging.Logger): Logger for logging messages.
    """

    def __init__(self, logger: logging.Logger):
        """
        Initialize the SideInformationLoader class with a logger instance.

        Args:
            logger (logging.Logger): Logger instance for logging.
        """
        self.logger = logger

        # Initialize attributes for processed data
        self.gene_side_information = None
        self.disease_side_information = None

    @staticmethod
    def to_coo(dataframe: pd.DataFrame) -> sp.coo_matrix:
        """
        Convert a DataFrame to a COO sparse matrix representation.

        Args:
            dataframe (pd.DataFrame): Input DataFrame with at least three columns.

        Returns:
            sp.coo_matrix: Sparse matrix in COO format.
        """
        mat = dataframe.to_numpy()
        return sp.coo_matrix(
            (mat[:, 2], (mat[:, 0], mat[:, 1])),
            shape=(int(mat[:, 0].max()) + 1, int(mat[:, 1].max()) + 1),
        )

    @staticmethod
    def add_implicit_ones(
        dataframe: pd.DataFrame, implicit_1s_name: str = "implicit 1s"
    ) -> pd.DataFrame:
        """
        Add a column of ones to a DataFrame.

        Args:
            dataframe (pd.DataFrame): Input DataFrame.
            implicit_1s_name (str): Name of the new column.

        Returns:
            pd.DataFrame: DataFrame with an additional column of ones.
        """
        dataframe[implicit_1s_name] = np.ones(len(dataframe))
        return dataframe

    def __call__(
        self, side_information_dataframes: List[pd.DataFrame], add_1s_list: List[bool]
    ) -> List[sp.coo_matrix]:
        """
        Process and convert datasets to sparse matrices.

        Args:
            side_information_dataframes (List[pd.DataFrame]): List of DataFrames
                to process.
            add_1s_list (List[bool]): List indicating whether to add implicit ones to
                corresponding DataFrames.

        Returns:
            List[sp.coo_matrix]: List of COO matrices for side information.
        """
        side_information = []
        for dataframe, add_1s in zip(side_information_dataframes, add_1s_list):
            if add_1s:
                dataframe = self.add_implicit_ones(dataframe)
            side_information.append(self.to_coo(dataframe))
        return side_information

    def process_side_information(
        self,
        gene_side_information_paths: List[str],
        gene_add_1s: List[bool],
        disease_side_information_paths: List[str],
        disease_add_1s: List[bool],
    ):
        """
        Process the datasets and store the results as attributes.

        Args:
            gene_side_information_paths (List[str]): List of file paths for gene-related
                side information.
            gene_add_1s (List[bool]): List indicating whether to add implicit ones to
                gene-related DataFrames.
            disease_side_information_paths (List[str]): List of file paths for
                disease-related side information.
            disease_add_1s (List[bool]): List indicating whether to add implicit ones to
                disease-related DataFrames.
        """
        gene_dataframes = [pd.read_csv(path) for path in gene_side_information_paths]
        disease_dataframes = [
            pd.read_csv(path) for path in disease_side_information_paths
        ]
        shapes = [dataframe.shape for dataframe in gene_dataframes + disease_dataframes]
        log_df = pd.DataFrame(
            shapes, columns=["number of rows", "number of columns"]
        ).applymap(lambda x: f"{x:_}")
        self.logger.debug(
            "Side information dataframes loaded successfully. \n%s\n",
            log_df.to_markdown(),
        )
        self.gene_side_information = self(gene_dataframes, gene_add_1s)
        self.disease_side_information = self(disease_dataframes, disease_add_1s)
        self.logger.debug(
            "Processed gene-side information and disease-side information successfully."
        )
