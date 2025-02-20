# pylint: disable=R0914, R0915, R0801, R0913
"""
Run non-Euclidean gradient-based method for gene prioritization.

This script can perform either cross-validation (for hyperparameter tuning) or
a train-evaluation cycle (for final model training/testing) using a Non-Euclidean Gradient
approach for gene prioritization. It reads a gene–disease association file, splits the data into
train/validation/test sets, and then either runs an Optuna hyperparameter search (cross-validation)
or a single train/test cycle (train-eval).
"""

import argparse
import logging
from pathlib import Path

import pint
import yaml

from genepriority.preprocessing.dataloader import DataLoader
from genepriority.scripts.utils import pre_processing
from genepriority.trainer.neg_trainer import NEGTrainer
from genepriority.utils import serialize


def cross_validation(
    logger: logging.Logger,
    output_path: Path,
    dataloader: DataLoader,
    rank: int,
    iterations: int,
    threshold: int,
    seed: int,
    n_trials: int,
    timeout: float = 8.0,
) -> None:
    """
    Runs cross-validation (using Optuna) for hyperparameter tuning of the NEGA model.

    Args:
        logger (logging.Logger): Logger for output messages.
        output_path (Path): Directory where output results will be saved.
        dataloader (DataLoader): Preprocessed DataLoader with gene–disease data.
        rank (int): Model rank (number of latent factors).
        iterations (int): Number of training iterations.
        threshold (int): Threshold parameter for the model.
        seed (int): Random seed for reproducibility.
        n_trials (int): Number of trials for the hyperparameter search.
        timeout (float, optional): Time out in hours. Default to 8.
    """
    trainer = NEGTrainer(
        dataloader=dataloader,
        path=output_path,
        seed=seed,
        iterations=iterations,
        threshold=threshold,
    )
    timeout_seconds = pint.Quantity(timeout, "h").to("s").m
    optuna_study = trainer.fine_tune(
        load_if_exists=False,
        n_trials=n_trials,
        timeout=timeout_seconds,
        num_latent=rank,
    )
    study_file = output_path / f"nega-cv-rank{rank}-it{iterations}.pickle"
    serialize(optuna_study, study_file)
    logger.info("Cross-validation completed. Results saved at %s", study_file)


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
    tensorboard_dir: Path,
    results_filename: str,
) -> None:
    """
    Trains the NEGA model on the training set and evaluates it on the test set.

    This function initializes a NEGTrainer with the specified hyperparameters, runs the
    training and evaluation cycle, and serializes the evaluation results.

    Args:
        logger (logging.Logger): Logger for output messages.
        output_path (Path): Directory to save output results.
        dataloader (DataLoader): Preprocessed DataLoader with gene–disease data.
        rank (int): Model rank (number of latent factors).
        iterations (int): Number of training iterations.
        threshold (int): Threshold parameter for the model.
        seed (int): Random seed for reproducibility.
        regularization_parameter (float): Regularization parameter for training.
        symmetry_parameter (float): Symmetry parameter for training.
        smoothness_parameter (float): Smoothness parameter for training.
        rho_increase (float): Rho increase factor.
        rho_decrease (float): Rho decrease factor.
        tensorboard_dir (Path): Directory for TensorBoard logs.
        results_filename (str): Filename to use for saving results.

    """
    trainer = NEGTrainer(
        dataloader=dataloader,
        path=output_path,
        seed=seed,
        iterations=iterations,
        threshold=threshold,
        regularization_parameter=regularization_parameter,
        symmetry_parameter=symmetry_parameter,
        smoothness_parameter=smoothness_parameter,
        rho_increase=rho_increase,
        rho_decrease=rho_decrease,
        tensorboard_dir=tensorboard_dir,
    )
    results_path = output_path / str(rank)
    results_path.mkdir(parents=True, exist_ok=True)
    trainer.path = results_path
    result = trainer.train_test_splits(
        rank,
        save_name=f"latent={rank}:model-omim1.pickle",
    )
    serialize(result, results_path / results_filename)
    logger.debug("Serialized results for latent dimension %s saved successfully.", rank)


def parse_nega(subparsers: argparse._SubParsersAction) -> None:
    """
    Adds subcommands for NEGA to the argument parser.

    Two subcommands are provided:
      - "nega-cv": Run cross-validation (for hyperparameter tuning).
      - "nega": Run a single train-evaluation cycle.

    Args:
        subparsers (argparse._SubParsersAction): The subparsers object to which
            the NEGA commands will be added.
    """
    cv_parser = subparsers.add_parser(
        "nega-cv", help="Perform cross-validation for hyperparameter tuning of NEGA."
    )
    eval_parser = subparsers.add_parser(
        "nega", help="Train and evaluate the NEGA model."
    )

    for subparser in [cv_parser, eval_parser]:
        subparser.add_argument(
            "--output-path",
            type=str,
            required=True,
            help="Directory to save output result (default: %(default)s).",
        )
        subparser.add_argument(
            "--num-splits",
            type=int,
            required=True,
            help="Number of data splits (default: %(default)s).",
        )
        subparser.add_argument(
            "--zero-sampling-factor",
            type=int,
            required=True,
            help=("Factor for zero sampling (number of zeros = factor * "
                  "number of ones) (default: %(default)s). "
            ),
        )
        subparser.add_argument(
            "--input-path",
            type=str,
            default="/home/TheGreatestCoder/code/data/postprocessed/",
            help=(
                "Path to the input data directory containing 'gene-disease.csv'"
                " (default: %(default)s)."
            ),
        )
        subparser.add_argument(
            "--omim-meta-path",
            type=str,
            default="/home/TheGreatestCoder/code/genepriority/configurations/omim.yaml",
            help="Path to the OMIM metadata file (default: %(default)s).",
        )
        subparser.add_argument(
            "--log-filename",
            type=str,
            default="pipeline.log",
            help="Filename for log output (default: %(default)s).",
        )
        subparser.add_argument(
            "--rank",
            type=int,
            default=40,
            help="Rank (number of latent factors) for the model (default: %(default)s).",
        )
        subparser.add_argument(
            "--iterations",
            type=int,
            default=200,
            help="Number of training iterations (default: %(default)s).",
        )
        subparser.add_argument(
            "--threshold",
            type=int,
            default=10,
            help="Threshold parameter for the model (default: %(default)s).",
        )
        subparser.add_argument(
            "--validation-size",
            type=float,
            default=0.1,
            help="Proportion of data to use for validation (default: %(default)s).",
        )
        subparser.add_argument(
            "--train-size",
            type=float,
            default=0.8,
            help="Proportion of data to use for training (default: %(default)s).",
        )
        subparser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed for reproducibility (default: %(default)s).",
        )

    eval_parser.add_argument(
        "--tensorboard-dir",
        type=str,
        default="/home/TheGreatestCoder/code/logs",
        help="Directory for TensorBoard logs (default: %(default)s).",
    )
    eval_parser.add_argument(
        "--config-path",
        type=str,
        default="/home/TheGreatestCoder/code/genepriority/configurations/nega/meta.yaml",
        help=(
            "Path to the YAML configuration file containing simulation parameters "
            "(default: %(default)s)."
        ),
    )
    eval_parser.add_argument(
        "--results-filename",
        type=str,
        default="results.pickle",
        help="Filename for serialized results (default: %(default)s).",
    )
    cv_parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of trials for hyperparameter tuning (default: %(default)s).",
    )


def nega(args: argparse.Namespace) -> None:
    """
    Main entry point for the NEGA script.

    This function parses command-line arguments, sets up logging, loads and preprocesses data,
    and triggers either cross-validation or train-evaluation mode based on the selected subcommand.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    input_path: Path = Path(args.input_path).absolute()
    omim_meta_path: Path = Path(args.omim_meta_path).absolute()
    output_path: Path = Path(args.output_path).absolute()
    logger: logging.Logger = logging.getLogger("NEGA")

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if not omim_meta_path.exists():
        raise FileNotFoundError(f"OMIM metadata path does not exist: {omim_meta_path}")

    dataloader, _ = pre_processing(
        input_path=input_path,
        seed=args.seed,
        omim_meta_path=omim_meta_path,
        side_info=False,
        num_splits=args.num_splits,
        zero_sampling_factor=args.zero_sampling_factor,
        num_folds=None,
        train_size=args.train_size,
        validation_size=args.validation_size,
    )

    if args.algorithm_command == "nega-cv":
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
    elif args.algorithm_command == "nega":
        config_path: Path = Path(args.config_path).absolute()
        if not config_path.exists():
            raise FileNotFoundError(
                f"The configuration path does not exist: {config_path}"
            )

        logger.debug("Loading configuration file: %s", config_path)
        with config_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)

        regularization_parameter: float = config.get("regularization_parameter")
        symmetry_parameter: float = config.get("symmetry_parameter")
        smoothness_parameter: float = config.get("smoothness_parameter")
        rho_increase: float = config.get("rho_increase")
        rho_decrease: float = config.get("rho_decrease")
        tensorboard_dir: Path = Path(args.tensorboard_dir).absolute()

        train_eval(
            logger=logger,
            output_path=output_path,
            dataloader=dataloader,
            rank=args.rank,
            iterations=args.iterations,
            threshold=args.threshold,
            seed=args.seed,
            regularization_parameter=regularization_parameter,
            symmetry_parameter=symmetry_parameter,
            smoothness_parameter=smoothness_parameter,
            rho_increase=rho_increase,
            rho_decrease=rho_decrease,
            tensorboard_dir=tensorboard_dir,
            results_filename=args.results_filename,
        )
    else:
        raise ValueError(
            "Invalid mode must be either 'cross-validation' or 'train-eval'."
        )
