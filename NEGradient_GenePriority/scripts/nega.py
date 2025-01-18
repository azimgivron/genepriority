# pylint: disable=R0914, R0915
"""
Run non-euclidean gradient-based method for gene prioritization.

This script can perform cross-validation (for hyperparameter tuning) or
train-evaluation (for final model training/testing) using a Non-Euclidean Gradient
approach for gene prioritization. It reads a gene-disease association file, splits
the data into train/validation/test sets, and either runs an Optuna hyperparameter
search (cross-validation) or a single train/test cycle (train-eval).
"""
import argparse
import logging
import os
import pickle
import sys
import traceback
from pathlib import Path

import pint

from NEGradient_GenePriority.preprocessing import DataLoader
from NEGradient_GenePriority.trainer import NEGTrainer
from NEGradient_GenePriority.utils import serialize


def setup_logger(log_file: Path) -> None:
    """
    Configures the root logger to write to a log file and the console.

    Args:
        log_file (Path): The file path where logs should be written.
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - [%(funcName)s] - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def cross_validation(
    logger: logging.Logger, input_path: Path, output_path: Path
) -> None:
    """
    Runs a cross-validation procedure (via Optuna) to perform hyperparameter tuning.

    Args:
        logger (logging.Logger): Logger for logging messages.
        input_path (Path): Path to the folder containing 'gene-disease.csv'.
        output_path (Path): Path to the folder where output files (logs, models, etc.)
            should be saved.
    """
    try:
        ############################
        # LOAD DATA
        ############################
        logger.debug("Setting parameters for splits and NEG")
        num_splits = 1
        seed = 42
        nb_genes = 14_195
        nb_diseases = 314
        validation_size = 0.1  # 10% of the whole data
        train_size = 0.8  # 80% of the 90% remaining data, hence total train size = 72% of full data
        min_associations = 10
        rank = 25
        iterations = 500
        threshold = 10
        n_trials = 5

        # Load data
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
        logger.debug("Configuring NEG session for cross-validation")

        trainer = NEGTrainer(
            dataloader=dataloader,
            path=output_path,
            seed=seed,
            logger=logger,
            iterations=iterations,
            threshold=threshold,
        )

        # Perform hyperparameter tuning with Optuna
        optuna_study = trainer.fine_tune(
            load_if_exists=False,
            n_trials=n_trials,
            timeout=pint.Quantity(8, "h").to("s").m,  # 8-hour timeout
            num_latent=rank,
        )

        # Save the Optuna study
        study_file = output_path / f"study-rank{rank}-it{iterations}.pickle"
        serialize(optuna_study, study_file)
        logger.info(
            "Cross-validation completed successfully. Optuna study saved at %s",
            study_file,
        )

    except Exception as exception:
        logger.error("An error occurred during cross-validation: %s", exception)
        logger.error("%s", traceback.format_exc())
        raise


def train_eval(logger: logging.Logger, input_path: Path, output_path: Path) -> None:
    """
    Trains a model on the training set and evaluates on the test set.

    Args:
        logger (logging.Logger): Logger for logging messages.
        input_path (Path): Path to the folder containing 'gene-disease.csv'.
        output_path (Path): Path to the folder where output files (logs, models, etc.)
            should be saved.
    """
    try:
        ############################
        # LOAD DATA
        ############################
        logger.debug("Setting parameters for splits and NEG (train_eval)")
        num_splits = 1
        seed = 42
        nb_genes = 14_195
        nb_diseases = 314
        validation_size = 0.1  # 10% of the whole data
        train_size = 0.8  # 80% of the 90% remaining data, hence total train size = 72% of full data
        min_associations = 10
        rank = 25
        iterations = 500
        threshold = 10

        # Load data
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
        logger.debug("Configuring NEG session for train-eval")

        trainer = NEGTrainer(
            dataloader=dataloader,
            path=output_path,
            seed=seed,
            logger=logger,
            iterations=iterations,
            threshold=threshold,
        )

        # Train/evaluate the model on the predefined splits
        evaluation_results = trainer.train_test_splits(rank, "trained_model.pickle")

        # Save the evaluation results
        evaluation_file = output_path / "evaluation.pickle"
        serialize(evaluation_results, evaluation_file)
        logger.info(
            "Train-eval completed successfully. Results saved at %s", evaluation_file
        )

    except Exception as exception:
        logger.error("An error occurred during train-eval: %s", exception)
        logger.error("%s", traceback.format_exc())
        raise


def main() -> None:
    """
    Entry point for the script. Parses command-line arguments and runs the
    specified pipeline step (cross-validation or train-eval).
    """
    parser = argparse.ArgumentParser(
        description="Run Non-Euclidean Gradient-based method for gene prioritization."
    )
    parser.add_argument(
        "--mode",
        choices=["cross-validation", "train-eval"],
        required=True,
        help="Choose the pipeline step: cross-validation or train-eval.",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="/home/TheGreatestCoder/code/data/postprocessed/",
        help="Path to input data directory containing 'gene-disease.csv'.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/home/TheGreatestCoder/code/neg/",
        help="Path to output directory (logs, models, etc.).",
    )

    args = parser.parse_args()
    input_path = Path(args.input_path).absolute()
    output_path = Path(args.output_path).absolute()
    os.makedirs(output_path, exist_ok=True)

    # Setup logger
    log_file = output_path / "pipeline.log"
    setup_logger(log_file)
    logger = logging.getLogger(__name__)

    # Verify input path exists
    if not input_path.exists():
        logger.error("The input path does not exist: %s", input_path)
        raise FileNotFoundError(f"The input path does not exist: {input_path}")

    # Run the chosen mode
    if args.mode == "cross-validation":
        logger.info("Starting cross-validation mode.")
        cross_validation(logger, input_path, output_path)
    elif args.mode == "train-eval":
        logger.info("Starting train-eval mode.")
        train_eval(logger, input_path, output_path)


if __name__ == "__main__":
    main()
