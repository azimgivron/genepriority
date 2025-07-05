# pylint : disable=R0913
"""
Side Information Loader Module
==============================

Provides utilities for loading, processing, and converting side information datasets 
into sparse matrix representations. The module includes methods for handling gene and 
disease-related data, adding implicit information, and converting datasets to COO sparse matrices.
"""
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD


class SideInformationLoader:
    """
    SideInformationLoader Class

    A utility class for loading, processing, and converting side information datasets
    related to genes and diseases into sparse matrix representations. This class is
    designed to handle side information, such as gene-gene or disease-disease relationships,
    and transform them into a format suitable for machine learning tasks like gene prioritization.

    Attributes:
        gene_side_info (sp.csr_matrix): The gene-related side information.
        disease_side_info (List[sp.coo_matrix]): The disease-related side information.
        logger (logging.Logger): Logger instance for recording information, warnings,
            and errors.
        nb_genes (int): Number of genes defining the dimensions of gene-side sparse matrices.
        nb_diseases (int): Number of diseases defining the dimensions of disease-side
            sparse matrices.
        max_dims (int, optional): Maximum number of dimensions of the side
            information.

    """

    def __init__(self, nb_genes: int, nb_diseases: int, max_dims: int):
        """
        Initialize the SideInformationLoader class with a logger instance.

        Args:
            nb_genes (int): Number of genes to define the shape of gene-side matrices.
            nb_diseases (int): Number of diseases to define the shape of
                disease-side matrices.
            max_dims (int): Maximum number of dimensions of the side
                information.
        """
        self.nb_genes = nb_genes
        self.nb_diseases = nb_diseases
        self.max_dims = max_dims
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize attributes for processed data
        self.gene_side_info = None
        self.disease_side_info = None

    @property
    def side_info(self) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
        """Getter of the side information for each dimensions.

        Returns:
            Tuple[sp.csr_matrix, sp.csr_matrix]: The side information
                for both genes and diseases.
        """
        return [self.gene_side_info, self.disease_side_info]

    @staticmethod
    def to_coo(dataframe: pd.DataFrame, rows: int) -> sp.coo_matrix:
        """
        Convert a DataFrame to a COO sparse matrix representation.

        Args:
            dataframe (pd.DataFrame): Input DataFrame containing at least three columns:
                - The first column indicates row indices.
                - The second column indicates column indices.
                - The third column (optional) contains the score or value for each entry.
            rows (int): Number of rows for the resulting matrix.

        Returns:
            sp.coo_matrix: Sparse matrix in COO format.
        """
        mat = dataframe.to_numpy()
        return sp.coo_matrix(
            (mat[:, 2], (mat[:, 0], mat[:, 1])),
            shape=(rows, int(mat[:, 1].max() + 1)),
        )

    @staticmethod
    def add_implicit_ones(
        dataframe: pd.DataFrame, implicit_1s_name: str = "implicit 1s"
    ) -> pd.DataFrame:
        """
        Add a column of implicit ones to a DataFrame for default scoring.

        Args:
            dataframe (pd.DataFrame): Input DataFrame containing at least two columns:
                - The first column indicates row indices.
                - The second column indicates column indices.
            implicit_1s_name (str): Name of the new column to be added.
                Default is "implicit 1s".

        Returns:
            pd.DataFrame: Modified DataFrame with an additional column of
                ones for scoring.
        """
        dataframe[implicit_1s_name] = np.ones(len(dataframe))
        return dataframe

    def __call__(
        self, side_info_dataframes: List[pd.DataFrame], rows: int
    ) -> sp.csr_matrix:
        """
        Process and convert datasets into sparse matrices in CSR format.

        Args:
            side_info_dataframes (List[pd.DataFrame]): List of DataFrames
                representing side information.
                - Each DataFrame must contain either two or three columns.
                - The first column represents row indices.
                - The second column represents column indices.
                - The optional third column contains scores or values
                (default: 1 if absent).
            rows (int): Number of rows for the resulting sparse matrices.

        Returns:
            sp.csr_matrix: A CSR matrix representing processed normalized side information.

        """
        side_info = []
        for dataframe in side_info_dataframes:
            if dataframe.shape[1] == 2:
                dataframe = self.add_implicit_ones(dataframe)
            elif dataframe.shape[1] != 3:
                raise ValueError(
                    "DataFrame must have 3 or 2 columns. First column is the row "
                    "index, the second is the column index and the optional third "
                    "column is the score. By default the score is 1."
                )
            side_info_mat = self.to_coo(dataframe, rows).tocsr()
            norm = sp.linalg.norm(side_info_mat, ord="fro")
            side_info.append(side_info_mat / norm)
        stacked = sp.hstack(side_info)
        normalized = stacked / sp.linalg.norm(stacked, ord="fro")
        return normalized

    def process_side_info(
        self,
        gene_side_info_paths: List[Path],
        disease_side_info_paths: List[Path],
    ):
        """
        Process gene- and disease-related side information files and store the results.

        Args:
            gene_side_info_paths (List[Path]): List of file paths for gene-side
                information datasets.
            disease_side_info_paths (List[Path]): List of file paths for disease-side
                information datasets.

        """
        gene_dataframes = [pd.read_csv(path) for path in gene_side_info_paths]
        disease_dataframes = [pd.read_csv(path) for path in disease_side_info_paths]
        shapes = [dataframe.shape for dataframe in gene_dataframes + disease_dataframes]
        log_df = pd.DataFrame(
            shapes, columns=["number of rows", "number of columns"]
        ).map(lambda x: f"{x:_}")
        names = [path.stem for path in gene_side_info_paths + disease_side_info_paths]
        log_df.index = names
        self.logger.debug(
            "Side informationdataframes loaded successfully. \n%s\n",
            log_df.to_markdown(),
        )
        self.gene_side_info = self(gene_dataframes, self.nb_genes)
        self.logger.debug(
            "Gene side information matrix has shape: %s",
            self.gene_side_info.shape,
        )
        self.disease_side_info = self(disease_dataframes, self.nb_diseases)
        self.logger.debug(
            "Disease side information matrix has shape: %s",
            self.disease_side_info.shape,
        )
        if self.max_dims is not None:
            if self.gene_side_info.shape[1] > self.max_dims:
                self.logger.debug(
                    "Using TruncatedSVD to reduce gene features from %d to %d",
                    self.gene_side_info.shape[1],
                    self.max_dims,
                )
                self.gene_side_info = TruncatedSVD(
                    n_components=self.max_dims
                ).fit_transform(self.gene_side_info)

            if self.disease_side_info.shape[1] > self.max_dims:
                self.logger.debug(
                    "Using TruncatedSVD to reduce disease features from %d to %d",
                    self.disease_side_info.shape[1],
                    self.max_dims,
                )
                self.disease_side_info = TruncatedSVD(
                    n_components=self.max_dims
                ).fit_transform(self.disease_side_info)
        self.logger.debug(
            "Processed gene-side information and disease-side information successfully."
        )
