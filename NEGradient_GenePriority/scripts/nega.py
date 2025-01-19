# pylint: disable=R0914, R0915, R0801
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
        ],
    )


def cross_validation(
    logger: logging.Logger, input_path: Path, output_path: Path
) -> None:
    """
    Runs a cross-validation procedure (via Optuna) to perform hyperparameter tuning.
    """
    try:
        # Parameters
        num_splits = 1
        seed = 42
        nb_genes = 14_195
        nb_diseases = 314
        validation_size = 0.1
        train_size = 0.8
        min_associations = 10
        rank = 100
        iterations = 700
        threshold = 10
        n_trials = 100

        # Load data
        dataloader = DataLoader(
            nb_genes=nb_genes,
            nb_diseases=nb_diseases,
            path=input_path / "gene-disease.csv",
            seed=seed,
            num_splits=num_splits,
            train_size=train_size,
            min_associations=min_associations,
            validation_size=validation_size,
            num_folds=None,
        )
        gene_disease = dataloader.load_data()
        dataloader.load_omim1(gene_disease)

        # Training
        trainer = NEGTrainer(
            dataloader=dataloader,
            path=output_path,
            seed=seed,
            logger=logger,
            iterations=iterations,
            threshold=threshold,
        )

        optuna_study = trainer.fine_tune(
            load_if_exists=False,
            n_trials=n_trials,
            timeout=pint.Quantity(8, "h").to("s").m,
            num_latent=rank,
        )

        study_file = output_path / f"study-rank{rank}-it{iterations}.pickle"
        serialize(optuna_study, study_file)
        logger.info("Cross-validation completed. Results saved at %s", study_file)

    except Exception as exception:
        logger.error("Error during cross-validation: %s", exception)
        logger.error("%s", traceback.format_exc())
        raise


def train_eval(logger: logging.Logger, input_path: Path, output_path: Path) -> None:
    """
    Trains a model on the training set and evaluates it on the test set.
    """
    tensorboard_base_log_dir = Path("/home/TheGreatestCoder/code/logs")
    try:
        # Parameters
        num_splits = 1
        seed = 42
        nb_genes = 14_195
        nb_diseases = 314
        validation_size = 0.1
        train_size = 0.8
        min_associations = 10
        rank = 20
        iterations = 1_000
        threshold = 10
        regularization_parameter = 0.003
        symmetry_parameter = 0.0002
        smoothness_parameter = 0.007
        rho_increase = 4.0
        rho_decrease = 0.6

        # Load data
        dataloader = DataLoader(
            nb_genes=nb_genes,
            nb_diseases=nb_diseases,
            path=input_path / "gene-disease.csv",
            seed=seed,
            num_splits=num_splits,
            train_size=train_size,
            min_associations=min_associations,
            validation_size=validation_size,
            num_folds=None,
        )
        gene_disease = dataloader.load_data()
        dataloader.load_omim1(gene_disease)

        # Training
        trainer = NEGTrainer(
            dataloader=dataloader,
            path=output_path,
            seed=seed,
            logger=logger,
            iterations=iterations,
            threshold=threshold,
            regularization_parameter=regularization_parameter,
            symmetry_parameter=symmetry_parameter,
            smoothness_parameter=smoothness_parameter,
            rho_increase=rho_increase,
            rho_decrease=rho_decrease,
            tensorboard_base_log_dir=tensorboard_base_log_dir,
        )

        evaluation_results = trainer.train_test_splits(rank, "trained_model.pickle")

        evaluation_file = output_path / "evaluation.pickle"
        serialize(evaluation_results, evaluation_file)
        logger.info("Train-eval completed. Results saved at %s", evaluation_file)

    except Exception as exception:
        logger.error("Error during train-eval: %s", exception)
        logger.error("%s", traceback.format_exc())
        raise


def main() -> None:
    """
    Entry point for the script.
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
        help="Path to input data directory containing 'gene-disease.csv' (default: %(default)s).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/home/TheGreatestCoder/code/neg/",
        help="Path to output directory (logs, models, etc.) (default: %(default)s).",
    )

    args = parser.parse_args()
    input_path = Path(args.input_path).absolute()
    output_path = Path(args.output_path).absolute()
    os.makedirs(output_path, exist_ok=True)

    log_file = output_path / "pipeline100.log"
    setup_logger(log_file)
    logger = logging.getLogger(__name__)

    if not input_path.exists():
        logger.error("Input path does not exist: %s", input_path)
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if args.mode == "cross-validation":
        logger.info("Starting cross-validation mode.")
        cross_validation(logger, input_path, output_path)
    elif args.mode == "train-eval":
        logger.info("Starting train-eval mode.")
        train_eval(logger, input_path, output_path)


if __name__ == "__main__":
    main()
