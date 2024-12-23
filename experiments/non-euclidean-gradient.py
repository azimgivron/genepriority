# pylint: disable=R0914, R0915
"""Main module"""
import logging
import os
import traceback
from pathlib import Path

from NEGradient_GenePriority import DataLoader, NEGTrainer


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
            num_folds=None,
        )
        dataloader(filter_column="Disease ID")  # load the data

        ############################
        # RUN TRAINING AND PREDICT
        ############################
        # Configure and run MACAU
        logger.debug("Configuring MACAU session")
        
        optim_reg=.1
        iterations=100
        lam=.4
        step_size=.3
        rho_increase=.4
        rho_decrease=.3
        threshold=20

        trainer = NEGTrainer(
            dataloader=dataloader,
            path=output_path,
            seed=seed,
            optim_reg=optim_reg,
            iterations=iterations,
            lam=lam,
            step_size=step_size,
            rho_increase=rho_increase,
            rho_decrease=rho_decrease,
            threshold=threshold,
            logger=logger,
        )
        
    except Exception as exception:
        logger.error("An error occurred during processing: %s", exception)
        logger.error("%s", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
