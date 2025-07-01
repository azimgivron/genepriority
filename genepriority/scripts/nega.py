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
from genepriority.preprocessing.side_information_loader import \
    SideInformationLoader
from genepriority.scripts.utils import pre_processing
from genepriority.trainer.neg_trainer import NEGTrainer
from genepriority.utils import serialize


def finetune(
    logger: logging.Logger,
    output_path: Path,
    dataloader: DataLoader,
    side_info_loader: SideInformationLoader,
    rank: int,
    iterations: int,
    threshold: int,
    flip_fraction: float,
    flip_frequency: int,
    patience: int,
    seed: int,
    n_trials: int,
    timeout: float,
):
    """
    Search for hyperparameter tuning of the NEGA model.

    Args:
        logger (logging.Logger): Logger for output messages.
        output_path (Path): Directory where output results will be saved.
        dataloader (DataLoader): Preprocessed DataLoader with gene–disease data.
        side_info_loader (SideInformationLoader): The loader for side information,
            if available.
        rank (int): Model rank (number of latent factors).
        iterations (int): Number of training iterations.
        threshold (int): Threshold parameter for the model.
        flip_fraction (float): Fraction (0 to 1) of positive training entries to flip,
            simulating label noise.
        flip_frequency (int): How often to resample positive training entries for flipping.
        patience (int): Number of recent epochs/iterations considered for early stopping.
        seed (int): Random seed for reproducibility.
        n_trials (int): Number of trials for the hyperparameter search.
        timeout (float, optional): Time out in hours.

    """
    trainer = NEGTrainer(
        dataloader=dataloader,
        side_info_loader=side_info_loader,
        path=output_path,
        seed=seed,
        iterations=iterations,
        threshold=threshold,
        flip_fraction=flip_fraction,
        flip_frequency=flip_frequency,
        patience=patience,
    )
    timeout_seconds = pint.Quantity(timeout, "h").to("s").m
    optuna_study = trainer.fine_tune(
        n_trials=n_trials,
        timeout=timeout_seconds,
        num_latent=rank,
    )
    filename = f"nega-finetune-rank{rank}"
    if side_info_loader is None:
        filename += "-no-side-info"
    if not dataloader.zero_sampling_factor > 0:
        filename += "-no-0s"
    study_file = output_path / f"{filename}.pickle"
    serialize(optuna_study, study_file)
    logger.info("Fine tuning completed. Results saved at %s", study_file)


def train_eval(
    logger: logging.Logger,
    output_path: Path,
    dataloader: DataLoader,
    side_info_loader: SideInformationLoader,
    rank: int,
    iterations: int,
    threshold: int,
    flip_fraction: float,
    flip_frequency: int,
    patience: int,
    seed: int,
    regularization_parameter: float,
    symmetry_parameter: float,
    smoothness_parameter: float,
    rho_increase: float,
    rho_decrease: float,
    tensorboard_dir: Path,
    results_filename: str,
):
    """
    Trains the NEGA model on the training set and evaluates it on the test set.

    This function initializes a NEGTrainer with the specified hyperparameters, runs the
    training and evaluation cycle, and serializes the evaluation results.

    Args:
        logger (logging.Logger): Logger for output messages.
        output_path (Path): Directory to save output results.
        dataloader (DataLoader): Preprocessed DataLoader with gene–disease data.
        side_info_loader (SideInformationLoader): The loader for side information,
            if available.
        rank (int): Model rank (number of latent factors).
        iterations (int): Number of training iterations.
        threshold (int): Threshold parameter for the model.
        flip_fraction (float): Fraction (0 to 1) of positive training entries to flip,
            simulating label noise.
        flip_frequency (int): How often to resample positive training entries for flipping.
        patience (int): Number of recent epochs/iterations considered for early stopping.
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
        side_info_loader=side_info_loader,
        path=output_path,
        seed=seed,
        iterations=iterations,
        threshold=threshold,
        flip_fraction=flip_fraction,
        flip_frequency=flip_frequency,
        patience=patience,
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
    result = trainer.train_test_cross_validation(
        rank,
        save_name=f"nega:latent={rank}.pickle",
    )
    serialize(result, results_path / results_filename)
    logger.debug("Serialized results for latent dimension %s saved successfully.", rank)


def nega(args: argparse.Namespace):
    """
    Main entry point for the NEGA script.

    This function parses command-line arguments, sets up logging, loads and preprocesses data,
    and triggers either cross-validation or train-evaluation mode based on the selected subcommand.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    logger = logging.getLogger("NEGA")

    dataloader, side_info_loader = pre_processing(
        gene_disease_path=args.gene_disease_path,
        seed=args.seed,
        omim_meta_path=args.omim_meta_path,
        side_info=args.side_info,
        gene_side_info_paths=args.gene_side_info_paths,
        disease_side_info_paths=args.disease_side_info_paths,
        zero_sampling_factor=args.zero_sampling_factor,
        num_folds=args.num_folds,
        validation_size=args.validation_size,
    )

    if args.nega_command == "fine-tune":
        finetune(
            logger=logger,
            output_path=args.output_path,
            dataloader=dataloader,
            side_info_loader=side_info_loader,
            rank=args.rank,
            iterations=args.iterations,
            threshold=args.threshold,
            seed=args.seed,
            flip_fraction=args.flip_fraction,
            flip_frequency=args.flip_frequency,
            patience=args.patience,
            n_trials=args.n_trials,
            timeout=args.timeout,
        )
    elif args.nega_command == "cv":
        logger.debug("Loading configuration file: %s", args.config_path)
        with args.config_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        if "side_info" in args and args.side_info is not None and args.side_info:
            config = config["side-info"]
            logger.debug("‘side-info‘ configuration loaded.")
        elif (
            "zero_sampling_factor" in args
            and args.zero_sampling_factor is not None
            and args.zero_sampling_factor > 0
        ):
            config = config["0s"]
            logger.debug("‘0s‘ configuration loaded.")
        else:
            config = config["1s"]
            logger.debug("‘1s‘ configuration loaded.")

        regularization_parameter = config.get("regularization_parameter")
        symmetry_parameter = config.get("symmetry_parameter")
        smoothness_parameter = config.get("smoothness_parameter")
        rho_increase = config.get("rho_increase")
        rho_decrease = config.get("rho_decrease")

        train_eval(
            logger=logger,
            output_path=args.output_path,
            dataloader=dataloader,
            rank=args.rank,
            iterations=args.iterations,
            threshold=args.threshold,
            flip_fraction=args.flip_fraction,
            flip_frequency=args.flip_frequency,
            patience=args.patience,
            seed=args.seed,
            regularization_parameter=regularization_parameter,
            symmetry_parameter=symmetry_parameter,
            smoothness_parameter=smoothness_parameter,
            rho_increase=rho_increase,
            rho_decrease=rho_decrease,
            tensorboard_dir=args.tensorboard_dir,
            results_filename=args.results_filename,
            side_info_loader=side_info_loader,
        )
    else:
        raise ValueError(
            "Invalid mode must be either 'cross-validation' or 'train-eval'."
        )
