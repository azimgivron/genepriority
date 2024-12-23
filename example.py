
# pylint: disable=R0914, R0915
"""Main module"""
import logging
import os
import traceback
from pathlib import Path

from NEGradient_GenePriority import (DataLoader, Evaluation,
                                     ModelEvaluationCollection,
                                     SideInformationLoader, Trainer,
                                     generate_auc_loss_table,
                                     generate_bedroc_table,
                                     plot_bedroc_boxplots, plot_roc_curves)


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
        logger.debug("Setting parameters for splits and NEG")
        latent_dimensions = [25, 30, 40]
        num_splits = 6
        num_folds = 5
        seed = 42
        nb_genes = 14_195
        nb_diseases = 314

        # load data
        dataloader = DataLoader(
            nb_genes=nb_genes,
            nb_diseases=nb_diseases,
            path=input_path / "gene-disease.csv",
            seed=seed,
            num_splits=num_splits,
            num_folds=num_folds,
        )
        dataloader(filter_column="Disease ID")  # load the data

        ############################
        # RUN TRAINING AND PREDICT
        ############################
        # Configure and run model
        logger.debug("Configuring NEG session")
        omim1_results = {}
        omim2_results = {}
        
        trainer = 
        
        MatrixCompletion(
            A: sp.csr_matrix,
            mask: sp.csr_matrix,
            test: sp.csr_matrix,
            test_mask: sp.csr_matrix,
            k: int,
            mu: float,
            iterations: int,
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
        
        collection = collections[0]
        auc_loss_dataframe = generate_auc_loss_table(
            collection.compute_auc_losses(),
            model_names=collection.model_names,
        )
        auc_loss_dataframe.to_csv(output_path / f"auc_loss_omim1.csv")

        collection = collections[1]
        plot_bedroc_boxplots(
            collection.compute_bedroc_scores(),
            model_names=latent_dimensions,
            output_file=(output_path / f"bedroc_omim2.png"),
        )
        bedroc_dataframe  = generate_bedroc_table(
            collection.compute_bedroc_scores(),
            model_names=collection.model_names,
            alpha_map=alpha_map
        )
        bedroc_dataframe.to_csv(output_path / f"bedroc_omim2.csv")
        logger.debug("Figures and tables creation completed successfully")
    except Exception as exception:
        logger.error("An error occurred during processing: %s", exception)
        logger.error("%s", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()