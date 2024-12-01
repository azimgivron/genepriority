import logging
from pathlib import Path

import numpy as np
import pandas as pd

from NEGradient_GenePriority import (
    combine_matrices,
    combine_splits,
    compute_statistics,
    convert_dataframe_to_sparse_matrix,
    create_folds,
    create_random_splits,
    filter_by_number_of_association,
    sample_zeros,
    train_and_test_folds,
    train_and_test_splits,
)


def setup_logger(log_file: str) -> logging.Logger:
    """
    Configures and returns a logger that writes to both the console and a file.
    The log file is overwritten each time the logger is set up.

    Args:
        log_file (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger("main_logger")
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create file handler with write mode
    file_handler = logging.FileHandler(
        log_file, mode="w"
    )  # 'w' ensures the file is overwritten
    file_handler.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)

    return logger


def main():
    # Setup paths and logger
    log_file = "pipeline.log"
    logger = setup_logger(log_file)
    input_path = Path("/home/TheGreatestCoder/code/data/postprocessed/").absolute()

    if not input_path.exists():
        logger.error("The input path does not exist: %s", input_path)
        raise FileNotFoundError(f"The input path does not exist: {input_path}")

    try:
        # Load data
        logger.debug("Loading gene-disease data from %s", input_path)
        gene_disease = pd.read_csv(input_path / "gene-disease.csv")
        logger.debug(
            "Loaded gene-disease data with %d rows and %d columns", *gene_disease.shape
        )

        # Set parameters
        logger.debug("Setting parameters for splits and BPMF")
        alphas = [228.5, 160.9, 32.2, 16.1, 5.3]
        latent_dimensions = [25, 30, 40]
        num_splits = 6
        zero_sampling_factor = 5

        # Convert gene-disease DataFrame to sparse matrix
        omim1_1s = convert_dataframe_to_sparse_matrix(gene_disease)
        omim1_0s = sample_zeros(omim1_1s, zero_sampling_factor, seed=42)
        omim1 = combine_matrices(omim1_1s, omim1_0s)
        logger.debug("Combined sparse matrix for OMIM1 created")
        omim1_1s_splits_indices = create_random_splits(omim1_1s, num_splits=num_splits)
        omim1_0s_splits_indices = create_random_splits(omim1_0s, num_splits=num_splits)
        omim1_splits_indices = combine_splits(
            omim1_1s_splits_indices, omim1_0s_splits_indices
        )
        logger.debug("Generated random splits for OMIM1 data")

        # Calculate sparsity
        sparsity = omim1_1s.count_nonzero() / (omim1_1s.shape[0] * omim1_1s.shape[1])
        logger.debug("Data sparsity: %.2f%%", sparsity * 100)

        counts = compute_statistics(omim1, omim1_splits_indices)
        logger.debug("Disease count statistics:\n%s", counts)

        # Filter diseases and create folds
        logger.debug("Filtering gene-disease data by association threshold")
        filtered_gene_disease = filter_by_number_of_association(
            gene_disease, threshold=10, col_name="disease ID"
        )
        logger.debug(
            "Filtered gene-disease data contains %d rows", len(filtered_gene_disease)
        )

        omim2_1s = convert_dataframe_to_sparse_matrix(filtered_gene_disease)
        omim2_0s = sample_zeros(omim2_1s, zero_sampling_factor, seed=42)
        omim2 = combine_matrices(omim2_1s, omim2_0s)
        logger.debug("Combined sparse matrix for OMIM2 created")
        omim2_1s_splits_indices = create_folds(omim2_1s, num_folds=num_splits)
        omim2_0s_splits_indices = create_folds(omim2_0s, num_folds=num_splits)
        omim2_splits_indices = combine_splits(
            omim2_1s_splits_indices, omim2_0s_splits_indices
        )
        logger.debug("Created folds for OMIM2 data")

        # Configure and run BPMF
        logger.debug("Configuring BPMF session")
        num_samples = 3500
        burnin_period = 500

        omim1_results = {}
        omim2_results = {}
        # for num_latent in latent_dimensions:
        #     logger.debug("Running BPMF for %d latent dimensions", num_latent)
        #     omim1_results[num_latent] = train_and_test_splits(
        #         omim1, omim1_splits_indices, num_samples, burnin_period, num_latent, alphas
        #     )
        #     omim2_results[num_latent] = train_and_test_folds(
        #         omim2, omim2_splits_indices, num_samples, burnin_period, num_latent, alphas
        #     )
        logger.debug("BPMF session completed successfully")

    except Exception as e:
        logger.exception("An error occurred during processing")
        raise


if __name__ == "__main__":
    main()
