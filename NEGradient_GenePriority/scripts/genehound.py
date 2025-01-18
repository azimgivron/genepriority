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
    python genehound_pipeline.py --input-path /path/to/data --output-path /path/to/results
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
        help=("Path to the directory containing input data, including "
              "'gene-disease.csv' (default: %(default)s)."),
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/home/TheGreatestCoder/code/genehounds/",
        help="Path to the directory where output results will be saved (default: %(default)s).",
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
        side_info_loader = SideInformationLoader(
            logger=logger, nb_genes=nb_genes, nb_diseases=nb_diseases
        )
        side_info_loader.process_side_info(
            gene_side_info_paths=[
                input_path / "interpro.csv",
                input_path / "uniprot.csv",
                input_path / "go.csv",
            ],
            disease_side_info_paths=[input_path / "phenotype.csv"],
            names=["interpro", "uniprot", "GO", "phenotype"],
        )

        ############################
        # RUN TRAINING AND PREDICT
        ############################
        logger.debug("Configuring MACAU session")
        trainer = MACAUTrainer(
            dataloader=dataloader,
            side_info_loader=side_info_loader,
            path=output_path,
            num_samples=3_500,
            burnin_period=500,
            direct=False,
            univariate=True,
            seed=seed,
            save_freq=100,
            verbose=0,
            logger=logger,
        )

        omim1_results, omim2_results = trainer(
            latent_dimensions=latent_dimensions, save_results=True
        )

        ############################
        # POST PROCESSING RESULTS
        ############################
        logger.debug("Starting figures and tables creation.")
        Evaluation.alphas = [228.5, 160.9, 32.2, 16.1, 5.3]
        Evaluation.alpha_map = {
            228.5: "100",
            160.9: "1%",
            32.2: "5%",
            16.1: "10%",
            5.3: "30%",
        }

        collections = [
            ModelEvaluationCollection(omim1_results),
            ModelEvaluationCollection(omim2_results),
        ]

        for i, collection in enumerate(collections, start=1):
            plot_roc_curves(
                evaluation_collection=collection,
                output_file=output_path / f"roc_curve_omim{i}",
            )

        auc_loss_csv_path = output_path / "auc_loss_omim1.csv"
        generate_auc_loss_table(
            collections[0].compute_auc_losses(),
            model_names=collections[0].model_names,
        ).to_csv(auc_loss_csv_path)
        logger.info("AUC/Loss table saved: %s", auc_loss_csv_path)

        bedroc_plot_path = output_path / "bedroc_omim2.png"
        plot_bedroc_boxplots(
            collections[1].compute_bedroc_scores(),
            model_names=latent_dimensions,
            output_file=bedroc_plot_path,
        )
        logger.info("BEDROC boxplots saved: %s", bedroc_plot_path)

        bedroc_csv_path = output_path / "bedroc_omim2.csv"
        generate_bedroc_table(
            collections[1].compute_bedroc_scores(),
            model_names=collections[1].model_names,
            alpha_map=Evaluation.alpha_map,
        ).to_csv(bedroc_csv_path)
        logger.info("BEDROC table saved: %s", bedroc_csv_path)

        logger.debug("Figures and tables creation completed successfully")

    except Exception as exception:
        logger.error("An error occurred during processing: %s", exception)
        logger.error("%s", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
