# pylint: disable=R0914, R0915
"""Main module"""
import logging
import os
import pickle
import traceback
from pathlib import Path

import pandas as pd

from NEGradient_GenePriority import (
    ModelEvaluationCollection,
    combine_matrices,
    combine_splits,
    compute_statistics,
    convert_dataframe_to_sparse_matrix,
    create_folds,
    create_random_splits,
    filter_by_number_of_association,
    generate_auc_loss_table,
    plot_bedroc_boxplots,
    plot_roc_curves,
    sample_zeros,
    train_and_test,
)


def setup_logger(log_file: str) -> None:
    """
    Configures the root logger to write to a log file and console.
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
        ],
    )


def main():
    """Main"""
    # Setup paths
    input_path = Path("/home/TheGreatestCoder/code/data/postprocessed/").absolute()
    output_path = Path("/home/TheGreatestCoder/code/output/").absolute()
    os.makedirs(output_path, exist_ok=True)

    # Setup logger
    log_file = output_path / "pipeline.log"
    setup_logger(log_file)
    logger = logging.getLogger(__name__)

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
        logger.debug("Setting parameters for splits and MACAU")
        latent_dimensions = [25, 30, 40]
        num_splits = 6
        num_folds = 5
        zero_sampling_factor = 5
        seed = 42

        # Convert gene-disease DataFrame to sparse matrix
        omim1_1s = convert_dataframe_to_sparse_matrix(gene_disease)
        omim1_0s = sample_zeros(omim1_1s, zero_sampling_factor, seed=seed)
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
            "Filtered gene-disease data contains %d genes-disease associations",
            len(filtered_gene_disease),
        )

        omim2_1s = convert_dataframe_to_sparse_matrix(filtered_gene_disease)
        omim2_0s = sample_zeros(omim2_1s, zero_sampling_factor, seed=seed)
        omim2 = combine_matrices(omim2_1s, omim2_0s)
        logger.debug("Combined sparse matrix for OMIM2 created")
        omim2_1s_folds_indices = create_folds(omim2_1s, num_folds=num_folds)
        omim2_0s_folds_indices = create_folds(omim2_0s, num_folds=num_folds)
        omim2_folds_indices = combine_splits(
            omim2_1s_folds_indices, omim2_0s_folds_indices
        )
        logger.debug("Created folds for OMIM2 data")

        # Configure and run MACAU
        logger.debug("Configuring MACAU session")
        num_samples = 1500
        burnin_period = 100
        save_freq = 100
        # Whether to use a Cholesky instead of conjugate gradient (CG) solver.
        # Keep false until the column features side information (F_e) reaches ~20,000.
        direct = False
        univariate = False  # Whether to use univariate or multivariate sampling.
        verbose = 0

        omim1_results = {}
        omim2_results = {}
        for num_latent in latent_dimensions:
            logger.debug("Running MACAU for %d latent dimensions", num_latent)
            logger.debug("Starting training on OMIM1")
            omim1_results[f"latent dim={num_latent}"] = train_and_test(
                omim1,
                omim1_splits_indices,
                num_samples,
                burnin_period,
                direct,
                univariate,
                num_latent,
                seed=seed,
                save_freq=save_freq,
                output_path=output_path,
                save_name=f"latent={num_latent}:macau-omim1.hdf5",
                verbose=verbose,
            )
            logger.debug("Starting training on OMIM2")
            omim2_results[f"latent dim={num_latent}"] = train_and_test(
                omim2,
                omim2_folds_indices,
                num_samples,
                burnin_period,
                direct,
                univariate,
                num_latent,
                seed=seed,
                save_freq=save_freq,
                output_path=output_path,
                save_name=f"latent={num_latent}:macau-omim2.hdf5",
                verbose=verbose,
            )
        logger.debug("MACAU session completed successfully")

        with open(output_path / "omim1_results.pickle", "wb") as handler:
            pickle.dump(omim1_results, handler)
        with open(output_path / "omim2_results.pickle", "wb") as handler:
            pickle.dump(omim2_results, handler)
        logger.debug("Results serialization completed successfully")

        logger.debug("Starting figures and tables creation.")
        omim1_collection = ModelEvaluationCollection(omim1_results)
        plot_roc_curves(
            evaluation_collection=omim1_collection,
            output_file=(output_path / "roc_curve_omim1.png"),
        )
        auc_loss_dataframe_omim1 = generate_auc_loss_table(
            omim1_collection.compute_auc_losses(),
            model_names=omim1_collection.model_names,
        )
        auc_loss_dataframe_omim1.to_csv(output_path / "auc_loss_omim1.csv")
        plot_bedroc_boxplots(
            omim1_collection.compute_bedroc_scores(),
            model_names=omim1_collection.model_names,
            output_file=(output_path / "bedroc_omim1.png"),
        )

        omim2_collection = ModelEvaluationCollection(omim2_results)
        plot_roc_curves(
            evaluation_collection=omim2_collection,
            output_file=(output_path / "roc_curve_omim2.png"),
        )
        auc_loss_dataframe_omim2 = generate_auc_loss_table(
            omim2_collection.compute_auc_losses(),
            model_names=omim2_collection.model_names,
        )
        auc_loss_dataframe_omim2.to_csv(output_path / "auc_loss_omim2.csv")
        plot_bedroc_boxplots(
            omim2_collection.compute_bedroc_scores(),
            model_names=omim2_collection.model_names,
            output_file=(output_path / "bedroc_omim2.png"),
        )
        logger.debug("Figures and tables creation completed successfully")
    except Exception as exception:
        logger.error("An error occurred during processing: %s", exception)
        logger.error("%s", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
