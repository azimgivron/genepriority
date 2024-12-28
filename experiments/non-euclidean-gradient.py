# pylint: disable=R0914, R0915
"""Run non-euclidean gradient based method for gene prioritization."""
import logging
import os
import pickle
import traceback
from pathlib import Path

import pint
from NEGradient_GenePriority import DataLoader, NEGTrainer


def setup_logger(log_file: str):
    """
    Configures the root logger to write to a log file and console.
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - [%(funcName)s] - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
        ],
    )


def main():
    """Main"""
    # Setup paths
    input_path = Path("/home/TheGreatestCoder/code/data/postprocessed/").absolute()
    output_path = Path("/home/TheGreatestCoder/code/neg/").absolute()
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
        num_splits = 1
        seed = 42
        nb_genes = 14_195
        nb_diseases = 314
        validation_size = 0.1 #10% of the whole data
        train_size = 0.8 #80% of the 90% remaining data, hence train size = 72% and test size = 18%
        min_associations = 10

        # load data
        dataloader = DataLoader(
            nb_genes=nb_genes,
            nb_diseases=nb_diseases,
            path=input_path / "gene-disease.csv",
            seed=seed,
            num_splits=num_splits,
            num_folds=None,
            train_size=train_size,
            min_associations=min_associations,
            validation_size=validation_size,
        )
        gene_disease = dataloader.load_data()
        dataloader.load_omim1(gene_disease)

        ############################
        # RUN TRAINING AND PREDICT
        ############################
        # Configure and run NEG
        logger.debug("Configuring NEG session")

        train_mask, test_mask = next(iter(dataloader.splits))
        trainer = NEGTrainer(
            dataloader=dataloader,
            path=output_path,
            seed=seed,
            logger=logger,
        )
    
        optuna_study = trainer.fine_tune(
            matrix=dataloader.omim1,
            train_mask=train_mask,
            test_mask=test_mask,
            load_if_exists=False,
            n_trials=10_000,
            timeout=pint.Quantity(2, "d").to("s").m,
        )

        with open(output_path / "study.pickle", "wb") as handler:
            pickle.dump(optuna_study, handler)

    except Exception as exception:
        logger.error("An error occurred during processing: %s", exception)
        logger.error("%s", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
