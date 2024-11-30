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

    Args:
        log_file (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger("main_logger")
    logger.setLevel(logging.DEBUG)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def main():
    # Setup paths and logger
    log_file = "pipeline.log"
    logger = setup_logger(log_file)
    input_path = Path("data/postprocessed/").absolute()

    if not input_path.exists():
        logger.error("The input path does not exist: %s", input_path)
        raise FileNotFoundError(f"The input path does not exist: {input_path}")

    try:
        # Load data
        logger.info("Loading gene-disease data from %s", input_path)
        gene_disease = pd.read_csv(input_path / "gene-disease.csv")
        logger.debug(
            "Loaded gene-disease data with %d rows and %d columns", *gene_disease.shape
        )

        # Convert gene-disease DataFrame to sparse matrix
        logger.info("Converting gene-disease data to a sparse matrix")
        omim1_1s = convert_dataframe_to_sparse_matrix(gene_disease)
        logger.debug(
            "Sparse matrix shape: %s, non-zero entries: %d",
            omim1_1s.shape,
            omim1_1s.count_nonzero(),
        )

        # Calculate sparsity
        sparsity = omim1_1s.count_nonzero() / (omim1_1s.shape[0] * omim1_1s.shape[1])
        logger.info("Data sparsity: %.2f%%", sparsity * 100)

        # Set parameters
        alphas = [228.5, 160.9, 32.2, 16.1, 5.3]
        latent_dimensions = [25, 30, 40]
        num_splits = 6
        zero_sampling_factor = 5

        # Sample zeros and generate random splits
        logger.info(
            "Sampling zeros with factor %d and creating random splits",
            zero_sampling_factor,
        )
        omim1_1s_splits_indices = create_random_splits(
            np.vstack((omim1_1s.row, omim1_1s.col)).T, num_splits=num_splits
        )
        omim1_0s = sample_zeros(omim1_1s, zero_sampling_factor)
        logger.debug(
            "Sampled zeros shape: %s, non-zero entries: %d",
            omim1_0s.shape,
            omim1_0s.count_nonzero(),
        )

        omim1_0s_splits_indices = create_random_splits(
            np.vstack((omim1_0s.row, omim1_0s.col)).T, num_splits=num_splits
        )
        omim1 = combine_matrices(omim1_1s, omim1_0s)
        omim1_splits_indices = combine_splits(
            omim1_1s_splits_indices, omim1_0s_splits_indices
        )
        logger.info("Generated combined splits for OMIM1 data")

        # Compute disease count statistics
        disease_stats = compute_statistics(omim1_1s_splits_indices)
        logger.info(
            "Disease count statistics (on the 1s):\n%s", disease_stats.to_markdown()
        )

        # Filter diseases and create folds
        logger.info("Filtering gene-disease data by association threshold")
        filtered_gene_disease = filter_by_number_of_association(
            gene_disease, threshold=10, col_name="disease ID"
        )
        logger.debug(
            "Filtered gene-disease data contains %d rows", len(filtered_gene_disease)
        )

        omim2_1s = convert_dataframe_to_sparse_matrix(filtered_gene_disease)
        omim2_1s_splits_indices = create_folds(omim2_1s, num_folds=num_splits)
        omim2_0s = sample_zeros(omim2_1s, zero_sampling_factor)
        omim2_0s_splits_indices = create_folds(omim2_0s, num_folds=num_splits)
        omim2 = combine_matrices(omim2_1s, omim2_0s)
        omim2_splits_indices = combine_splits(
            omim2_1s_splits_indices, omim2_0s_splits_indices
        )
        logger.info("Processed OMIM2 data and created folds")

        # Configure and run BPMF
        logger.info("Configuring BPMF session")
        num_samples = 3500
        burnin_period = 500

        omim1_results = {}
        omim2_results = {}
        for num_latent in latent_dimensions:
            logger.info("Running BPMF for %d latent dimensions", num_latent)
            omim1_results[num_latent] = train_and_test_splits(
                omim1, omim1_splits_indices, num_samples, burnin_period, num_latent, alphas
            )
            omim2_results[num_latent] = train_and_test_folds(
                omim2, omim2_splits_indices, num_samples, burnin_period, num_latent, alphas
            )
        logger.info("BPMF session completed successfully")

    except Exception as e:
        logger.exception("An error occurred during processing")
        raise


if __name__ == "__main__":
    main()
