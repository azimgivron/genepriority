# pylint: disable=R0914, R0915
"""Main module"""
import logging
import os
import traceback
from pathlib import Path

from NEGradient_GenePriority import (
    DataLoader,
    Evaluation,
    ModelEvaluationCollection,
    Trainer,
    generate_auc_loss_table,
    plot_bedroc_boxplots,
    plot_roc_curves,
)


def setup_logger(log_file: str):
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
        ############################
        # LOAD DATA
        ############################
        # Set parameters
        logger.debug("Setting parameters for splits and MACAU")
        latent_dimensions = [25, 30, 40]
        num_splits = 6
        num_folds = 5
        zero_sampling_factor = 5
        seed = 42

        dataloader = DataLoader(
            input_path / "gene-disease.csv",
            seed,
            num_splits,
            zero_sampling_factor,
            num_folds,
        )
        dataloader()  # load the data

        ############################
        # RUN TRAINING AND PREDICT
        ############################
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

        trainer = Trainer(
            dataloader=dataloader,
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
        for i, collection in enumerate(collections, 1):
            plot_roc_curves(
                evaluation_collection=collection,
                output_file=(output_path / f"roc_curve_omim{i}"),
            )
            auc_loss_dataframe = generate_auc_loss_table(
                collection.compute_auc_losses(),
                model_names=collection.model_names,
            )
            auc_loss_dataframe.to_csv(output_path / f"auc_loss_omim{i}.csv")
            plot_bedroc_boxplots(
                collection.compute_bedroc_scores(),
                model_names=latent_dimensions,
                output_file=(output_path / f"bedroc_omim{i}.png"),
            )
        logger.debug("Figures and tables creation completed successfully")
    except Exception as exception:
        logger.error("An error occurred during processing: %s", exception)
        logger.error("%s", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
