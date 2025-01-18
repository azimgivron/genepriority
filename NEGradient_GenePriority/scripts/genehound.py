# pylint: disable=R0914, R0915
"""
Reproduce GeneHound results using a MACAU-based approach.

This script:
1. Loads gene-disease association data.
2. Loads side information for genes and diseases.
3. Trains multiple MACAU models (with different latent dimensions).
4. Evaluates the models and produces:
   - ROC curves
   - AUC/loss tables
   - BEDROC scores

Usage:
    python genehound_pipeline.py \
        --input-path /path/to/data \
        --output-path /path/to/results
"""
import argparse
import logging
import os
import sys
import traceback
from pathlib import Path

from NEGradient_GenePriority import (
    DataLoader,
    Evaluation,
    MACAUTrainer,
    ModelEvaluationCollection,
    SideInformationLoader,
    generate_auc_loss_table,
    generate_bedroc_table,
    plot_bedroc_boxplots,
    plot_roc_curves,
)


def setup_logger(log_file: Path) -> None:
    """
    Configures the root logger to write to a log file and also to the console.

    Args:
        log_file (Path): Path to the log file where logs will be written.
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - [%(funcName)s] - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main() -> None:
    """
    Main entry point for the GeneHound reproduction pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Reproduce GeneHound results using MACAU-based approach."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="/home/TheGreatestCoder/code/data/postprocessed/",
        help="Path to the directory containing input data, including 'gene-disease.csv'.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/home/TheGreatestCoder/code/genehounds/",
        help="Path to the directory where output results will be saved.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path).absolute()
    output_path = Path(args.output_path).absolute()
    os.makedirs(output_path, exist_ok=True)

    # Setup logger
    log_file = output_path / "pipeline.log"
    setup_logger(log_file)
    logger = logging.getLogger(__name__)

    # Validate input path
    if not input_path.exists():
        logger.error("The input path does not exist: %s", input_path)
        raise FileNotFoundError(f"The input path does not exist: {input_path}")

    try:
        ############################
        # LOAD DATA
        ############################
        logger.debug("Setting parameters for splits and MACAU")
        latent_dimensions = [25, 30, 40]
        num_splits = 6
        num_folds = 5
        zero_sampling_factor = 5
        seed = 42
        nb_genes = 14_195
        nb_diseases = 314
        train_size = 0.9
        min_associations = 10

        # Load gene-disease data
        dataloader = DataLoader(
            nb_genes=nb_genes,
            nb_diseases=nb_diseases,
            path=input_path / "gene-disease.csv",
            seed=seed,
            num_splits=num_splits,
            zero_sampling_factor=zero_sampling_factor,
            num_folds=num_folds,
            train_size=train_size,
            min_associations=min_associations,
        )
        dataloader(filter_column="Disease ID")

        # Load side information
        interpro_path = input_path / "interpro.csv"
        uniprot_path = input_path / "uniprot.csv"
        go_path = input_path / "go.csv"
        phenotype_path = input_path / "phenotype.csv"
        side_info_loader = SideInformationLoader(
            logger=logger, nb_genes=nb_genes, nb_diseases=nb_diseases
        )
        side_info_loader.process_side_info(
            gene_side_info_paths=[interpro_path, uniprot_path, go_path],
            disease_side_info_paths=[phenotype_path],
            names=["interpro", "uniprot", "GO", "phenotype"],
        )

        ############################
        # RUN TRAINING AND PREDICT
        ############################
        logger.debug("Configuring MACAU session")
        num_samples = 3_500
        burnin_period = 500
        save_freq = 100
        direct = False  # Whether to use Cholesky solver
        univariate = True  # Whether to use univariate or multivariate sampling
        verbose = 0

        # Initialize MACAUTrainer
        trainer = MACAUTrainer(
            dataloader=dataloader,
            side_info_loader=side_info_loader,
            path=output_path,
            num_samples=num_samples,
            burnin_period=burnin_period,
            direct=direct,
            univariate=univariate,
            seed=seed,
            save_freq=save_freq,
            verbose=verbose,
            logger=logger,
        )

        # Run training for multiple latent dimensions
        omim1_results, omim2_results = trainer(
            latent_dimensions=latent_dimensions, save_results=True
        )

        ############################
        # POST PROCESSING RESULTS
        ############################
        logger.debug("Starting figures and tables creation.")
        alphas = [228.5, 160.9, 32.2, 16.1, 5.3]
        alpha_map = {228.5: "100", 160.9: "1%", 32.2: "5%", 16.1: "10%", 5.3: "30%"}
        Evaluation.alphas = alphas
        Evaluation.alpha_map = alpha_map

        collections = [
            ModelEvaluationCollection(omim1_results),
            ModelEvaluationCollection(omim2_results),
        ]

        # Plot ROC curves for OMIM1 and OMIM2
        for i, collection in enumerate(collections, start=1):
            plot_roc_curves(
                evaluation_collection=collection,
                output_file=output_path / f"roc_curve_omim{i}",
            )

        # Generate AUC/Loss table for OMIM1
        collection_omim1 = collections[0]
        auc_loss_dataframe = generate_auc_loss_table(
            collection_omim1.compute_auc_losses(),
            model_names=collection_omim1.model_names,
        )
        auc_loss_csv_path = output_path / "auc_loss_omim1.csv"
        auc_loss_dataframe.to_csv(auc_loss_csv_path)
        logger.info("AUC/Loss table saved: %s", auc_loss_csv_path)

        # Generate BEDROC plots for OMIM2
        collection_omim2 = collections[1]
        bedroc_scores = collection_omim2.compute_bedroc_scores()
        bedroc_plot_path = output_path / "bedroc_omim2.png"
        plot_bedroc_boxplots(
            bedroc_scores, model_names=latent_dimensions, output_file=bedroc_plot_path
        )
        logger.info("BEDROC boxplots saved: %s", bedroc_plot_path)

        # Generate BEDROC table for OMIM2
        bedroc_df = generate_bedroc_table(
            bedroc_scores,
            model_names=collection_omim2.model_names,
            alpha_map=alpha_map,
        )
        bedroc_csv_path = output_path / "bedroc_omim2.csv"
        bedroc_df.to_csv(bedroc_csv_path)
        logger.info("BEDROC table saved: %s", bedroc_csv_path)

        logger.debug("Figures and tables creation completed successfully")

    except Exception as exception:
        logger.error("An error occurred during processing: %s", exception)
        logger.error("%s", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
