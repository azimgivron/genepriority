# pylint: disable=R0914, R0915, R0801, R0913
"""
Run non-Euclidean gradient-based method for gene prioritization.

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


def setup_logger(log_file: Path):
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


def load_data(
    num_splits: int,
    train_size: float,
    validation_size: float,
    input_path: Path,
    seed: int,
    zero_sampling_factor: int,
) -> DataLoader:
    """
    Loads and preprocesses gene-disease association data.

    Args:
        num_splits (int): Number of data splits.
        train_size (float): Proportion of data to use for training.
        validation_size (float): Proportion of data to use for validation.
        input_path (Path): Path to the input data directory.
        seed (int): Random seed for reproducibility.
        zero_sampling_factor (int): Multiplier for generating negative associations.

    Returns:
        DataLoader: A DataLoader instance with the loaded data.
    """
    nb_genes = 14_195
    nb_diseases = 314
    min_associations = 10

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
        zero_sampling_factor=zero_sampling_factor,
    )
    gene_disease = dataloader.load_data()
    dataloader.load_omim1(gene_disease)
    return dataloader


def cross_validation(
    logger: logging.Logger,
    output_path: Path,
    dataloader: DataLoader,
    rank: int,
    iterations: int,
    threshold: int,
    seed: int,
    n_trials: int,
):
    """
    Runs a cross-validation procedure (via Optuna) to perform hyperparameter tuning.

    Args:
        logger (logging.Logger): Logger for logging messages.
        output_path (Path): Path to the output data directory.
        dataloader (DataLoader): DataLoader instance with preprocessed data.
        rank (int): Rank of the model.
        iterations (int): Number of iterations for training.
        threshold (int): Threshold parameter for the model.
        seed (int): Random seed for reproducibility.
        n_trials (int): Number of trials for hyperparameter tuning.
    """
    try:
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


def train_eval(
    logger: logging.Logger,
    output_path: Path,
    dataloader: DataLoader,
    rank: int,
    iterations: int,
    threshold: int,
    seed: int,
    regularization_parameter: float,
    symmetry_parameter: float,
    smoothness_parameter: float,
    rho_increase: float,
    rho_decrease: float,
    tensorboard_base_log_dir: Path,
):
    """
    Trains a model on the training set and evaluates it on the test set.

    Args:
        logger (logging.Logger): Logger for logging messages.
        output_path (Path): Path to the output data directory.
        dataloader (DataLoader): DataLoader instance with preprocessed data.
        rank (int): Rank of the model.
        iterations (int): Number of iterations for training.
        threshold (int): Threshold parameter for the model.
        seed (int): Random seed for reproducibility.
        regularization_parameter (float): Regularization parameter.
        symmetry_parameter (float): Symmetry parameter.
        smoothness_parameter (float): Smoothness parameter.
        rho_increase (float): Rho increase value.
        rho_decrease (float): Rho decrease value.
        tensorboard_base_log_dir (Path): Path to the TensorBoard log directory.
    """
    try:
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


def get_args() -> argparse.Namespace:
    """
    Parses and retrieves command-line arguments.

    Returns:
        argparse.Namespace: Namespace containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run Non-Euclidean Gradient-based method for gene prioritization."
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    cv_parser = subparsers.add_parser(
        "cross-validation", help="Perform cross-validation for hyperparameter tuning."
    )
    eval_parser = subparsers.add_parser(
        "train-eval", help="Train and evaluate the model."
    )

    for subparser in [cv_parser, eval_parser]:
        subparser.add_argument(
            "--input-path",
            type=str,
            default="/home/TheGreatestCoder/code/data/postprocessed/",
            help=(
                "Path to input data directory containing "
                "'gene-disease.csv' (default: %(default)s)."
            ),
        )
        subparser.add_argument(
            "--output-path",
            type=str,
            default="/home/TheGreatestCoder/code/neg/",
            help="Path to output directory (logs, models, etc.) (default: %(default)s).",
        )
        subparser.add_argument(
            "--log-filename",
            type=str,
            default="pipeline.log",
            help="Filename of the logs. (default: %(default)s).",
        )
        subparser.add_argument(
            "--num-splits",
            type=int,
            default=1,
            help="Number of data splits (default: %(default)s).",
        )
        subparser.add_argument(
            "--rank",
            type=int,
            default=50,
            help="Rank of the model (default: %(default)s).",
        )
        subparser.add_argument(
            "--iterations",
            type=int,
            default=700,
            help="Number of iterations (default: %(default)s).",
        )
        subparser.add_argument(
            "--threshold",
            type=int,
            default=10,
            help="Threshold parameter (default: %(default)s).",
        )
        subparser.add_argument(
            "--validation-size",
            type=float,
            default=0.1,
            help="Validation set size (default: %(default)s).",
        )
        subparser.add_argument(
            "--train-size",
            type=float,
            default=0.8,
            help="Training set size (default: %(default)s).",
        )
        subparser.add_argument(
            "--seed", type=int, default=42, help="Random seed (default: %(default)s)."
        )
        subparser.add_argument(
            "--zero-sampling-factor",
            type=int,
            default=None,
            help=(
                "Factor to determine the number of zeros to sample, calculated as the "
                "specified factor multiplied by the number of ones (default: %(default)s)."
            ),
        )

    eval_parser.add_argument(
        "--tensorboard-base-log-dir",
        type=str,
        default="/home/TheGreatestCoder/code/logs",
        help="Path to the TensorBoard log directory (default: %(default)s).",
    )
    eval_parser.add_argument(
        "--regularization-parameter",
        type=float,
        default=0.003,
        help="Regularization parameter (default: %(default)s).",
    )
    eval_parser.add_argument(
        "--symmetry-parameter",
        type=float,
        default=0.08,
        help="Symmetry parameter (default: %(default)s).",
    )
    eval_parser.add_argument(
        "--smoothness-parameter",
        type=float,
        default=0.002,
        help="Smoothness parameter (default: %(default)s).",
    )
    eval_parser.add_argument(
        "--rho-increase",
        type=float,
        default=4.0,
        help="Rho increase value (default: %(default)s).",
    )
    eval_parser.add_argument(
        "--rho-decrease",
        type=float,
        default=0.9,
        help="Rho decrease value (default: %(default)s).",
    )
    cv_parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of trials for hyperparameter tuning (default: %(default)s).",
    )
    return parser.parse_args()


def main():
    """
    Entry point for the script, managing the overall execution flow.

    This function parses command-line arguments, sets up the logger, initializes
    data loading, and triggers either cross-validation or train-evaluation based
    on the selected mode.
    """
    args = get_args()

    input_path = Path(args.input_path).absolute()
    output_path = Path(args.output_path).absolute()

    os.makedirs(output_path, exist_ok=True)

    log_file = output_path / args.log_filename
    setup_logger(log_file)
    logger = logging.getLogger(__name__)

    if not input_path.exists():
        logger.error("Input path does not exist: %s", input_path)
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    dataloader = load_data(
        num_splits=args.num_splits,
        train_size=args.train_size,
        validation_size=args.validation_size,
        input_path=input_path,
        seed=args.seed,
        zero_sampling_factor=args.zero_sampling_factor,
    )

    if args.mode == "cross-validation":
        logger.info("Starting cross-validation mode.")
        cross_validation(
            logger=logger,
            output_path=output_path,
            dataloader=dataloader,
            rank=args.rank,
            iterations=args.iterations,
            threshold=args.threshold,
            seed=args.seed,
            n_trials=args.n_trials,
        )
    elif args.mode == "train-eval":
        logger.info("Starting train-eval mode.")
        tensorboard_base_log_dir = Path(args.tensorboard_base_log_dir).absolute()
        train_eval(
            logger=logger,
            output_path=output_path,
            dataloader=dataloader,
            rank=args.rank,
            iterations=args.iterations,
            threshold=args.threshold,
            seed=args.seed,
            regularization_parameter=args.regularization_parameter,
            symmetry_parameter=args.symmetry_parameter,
            smoothness_parameter=args.smoothness_parameter,
            rho_increase=args.rho_increase,
            rho_decrease=args.rho_decrease,
            tensorboard_base_log_dir=tensorboard_base_log_dir,
        )


if __name__ == "__main__":
    main()
