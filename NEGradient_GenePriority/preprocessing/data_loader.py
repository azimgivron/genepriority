import logging

import pandas as pd

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
    def __init__(
        self,
        path: str,
        seed: int,
        num_splits: int,
        zero_sampling_factor: int,
        num_folds: int,
        logger=None,
    ) -> None:
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

    def __call__(self):
        # Load data
        self.logger.debug("Loading gene-disease data from %s", self.path)
        gene_disease = pd.read_csv(self.path)

        self.logger.debug(
            "Loaded gene-disease data with %d rows and %d columns", *gene_disease.shape
        )
        self.load_omim1(gene_disease)
        self.load_omim2(gene_disease)

    def load_omim1(self, gene_disease: pd.DataFrame):
        # Convert gene-disease DataFrame to sparse matrix
        omim1_1s = convert_dataframe_to_sparse_matrix(gene_disease)
        omim1_0s = [
            sample_zeros(omim1_1s, self.zero_sampling_factor, seed=self.seed)
            for _ in range(self.num_splits)
        ]
        self.omim1 = [
            combine_matrices(omim1_1s, omim1_0s_per_split)
            for omim1_0s_per_split in omim1_0s
        ]
        self.logger.debug("Combined sparse matrix for OMIM1 created")
        omim1_1s_splits_indices = create_random_splits_from_matrix(
            omim1_1s, num_splits=self.num_splits
        )
        omim1_0s_splits_indices = create_random_splits_from_matrices(omim1_0s)
        self.omim1_splits_indices = combine_splits(
            omim1_1s_splits_indices, omim1_0s_splits_indices
        )
        self.logger.debug("Generated random splits for OMIM1 data")

        # Calculate sparsity
        sparsity = omim1_1s.count_nonzero() / (omim1_1s.shape[0] * omim1_1s.shape[1])
        self.logger.debug("Data sparsity: %.2f%%", sparsity * 100)

        counts = compute_statistics(omim1_1s, omim1_1s_splits_indices)
        self.logger.debug("Disease count statistics:\n%s", counts)

    def load_omim2(self, gene_disease: pd.DataFrame):
        # Filter diseases and create folds
        self.logger.debug("Filtering gene-disease data by association threshold")
        filtered_gene_disease = filter_by_number_of_association(
            gene_disease, threshold=10, col_name="disease ID"
        )
        self.logger.debug(
            "Filtered gene-disease data contains %d genes-disease associations",
            len(filtered_gene_disease),
        )

        omim2_1s = convert_dataframe_to_sparse_matrix(filtered_gene_disease)
        omim2_0s = sample_zeros(omim2_1s, self.zero_sampling_factor, seed=self.seed)
        self.omim2 = combine_matrices(omim2_1s, omim2_0s)
        self.logger.debug("Combined sparse matrix for OMIM2 created")
        omim2_1s_folds_indices = create_folds(omim2_1s, num_folds=self.num_folds)
        omim2_0s_folds_indices = create_folds(omim2_0s, num_folds=self.num_folds)
        self.omim2_folds_indices = combine_splits(
            omim2_1s_folds_indices, omim2_0s_folds_indices
        )
        self.logger.debug("Created folds for OMIM2 data")
