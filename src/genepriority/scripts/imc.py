# pylint: disable=R0913,R0914,R0915
"""
Run Inductive Matrix Completion (IMC) training workflows.

This launcher provides high-level commands for:
  * Hyper-parameter search via Optuna (``fine-tune`` subcommand).
  * Cross-validated training/evaluation using predefined configurations (``cv`` subcommand).
"""

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict

import pint
import yaml

from genepriority.preprocessing.dataloader import DataLoader
from genepriority.preprocessing.side_information_loader import \
    SideInformationLoader
from genepriority.scripts.utils import pre_processing
from genepriority.trainer.imc_trainer import IMCTrainer
from genepriority.utils import serialize


def finetune(
    logger: logging.Logger,
    output_path: Path,
    dataloader: DataLoader,
    side_info_loader: SideInformationLoader,
    rank: int,
    iterations: int,
    max_inner_iter: int,
    flip_fraction: float,
    flip_frequency: int,
    patience: int,
    seed: int,
    n_trials: int,
    timeout: float,
    svd_init: bool,
):
    """
    Runs Optuna-based hyperparameter tuning for IMC.

    Args:
        logger (logging.Logger): Logger for output messages.
        output_path (Path): Directory where output results will be saved.
        dataloader (DataLoader): Preprocessed DataLoader with gene-disease data.
        side_info_loader (SideInformationLoader): The loader for side information,
            if available.
        rank (int): Model rank (number of latent factors).
        iterations (int): Number of training iterations.
        max_inner_iter (int): Maximum number of iterations for the inner optimization loop.
        flip_fraction (float): Fraction (0 to 1) of positive training entries to flip,
            simulating label noise.
        flip_frequency (int): How often to resample positive training entries for flipping.
        patience (int): Number of recent epochs/iterations considered for early stopping.
        seed (int): Random seed for reproducibility.
        n_trials (int): Number of trials for the hyperparameter search.
        timeout (float): Time out in hours.
        svd_init (bool): Whether to initialize the latent matrices with SVD decomposition.
    """
    if side_info_loader is None:
        raise ValueError("IMC fine-tuning requires gene and disease side information.")

    trainer = IMCTrainer(
        dataloader=dataloader,
        side_info_loader=side_info_loader,
        path=output_path,
        seed=seed,
        iterations=iterations,
        max_inner_iter=max_inner_iter,
        flip_fraction=flip_fraction,
        flip_frequency=flip_frequency,
        patience=patience,
        svd_init=svd_init,
    )
    timeout_seconds = pint.Quantity(timeout, "h").to("s").m
    optuna_study = trainer.fine_tune(
        n_trials=n_trials,
        timeout=timeout_seconds,
        num_latent=rank,
    )
    filename = f"imc-finetune-rank{rank}"
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
    max_inner_iter: int,
    flip_fraction: float,
    flip_frequency: int,
    patience: int,
    seed: int,
    regularization_parameters: Dict[str, float],
    tensorboard_dir: Path,
    results_filename: str,
    svd_init: bool,
):
    """
    Trains the IMC model on cross-validation folds and serializes evaluation results.

    Args:
        logger (logging.Logger): Logger for output messages.
        output_path (Path): Directory to save output results.
        dataloader (DataLoader): Preprocessed DataLoader with gene-disease data.
        side_info_loader (SideInformationLoader): The loader for side information,
            if available.
        rank (int): Model rank (number of latent factors).
        iterations (int): Number of training iterations.
        max_inner_iter (int): Maximum number of iterations for the inner optimization loop.
        flip_fraction (float): Fraction (0 to 1) of positive training entries to flip,
            simulating label noise.
        flip_frequency (int): How often to resample positive training entries for flipping.
        patience (int): Number of recent epochs/iterations considered for early stopping.
        seed (int): Random seed for reproducibility.
        regularization_parameters (Dict[str, float]): Regularization parameters for training.
        tensorboard_dir (Path): Directory for TensorBoard logs.
        results_filename (str): Filename to use for saving results.
        svd_init (bool): Whether to initialize the latent
                matrices with SVD decomposition.
    """
    if side_info_loader is None:
        raise ValueError("IMC training requires gene and disease side information.")

    trainer = IMCTrainer(
        dataloader=dataloader,
        side_info_loader=side_info_loader,
        path=output_path,
        seed=seed,
        iterations=iterations,
        max_inner_iter=max_inner_iter,
        flip_fraction=flip_fraction,
        flip_frequency=flip_frequency,
        patience=patience,
        tensorboard_dir=tensorboard_dir,
        regularization_parameters=regularization_parameters,
        svd_init=svd_init,
    )
    results_path = output_path / str(rank)
    results_path.mkdir(parents=True, exist_ok=True)
    trainer.path = results_path
    result = trainer.train_test_cross_validation(
        rank,
        save_name=f"imc:latent={rank}.pickle",
    )
    serialize(result, results_path / results_filename)
    logger.debug("Serialized results for latent dimension %s saved successfully.", rank)


def imc(args: argparse.Namespace):
    """
    Entry point for the IMC launcher.
    """
    logger = logging.getLogger("IMC")
    kwargs = defaultdict(list)

    if not args.gene_side_info or not args.disease_side_info:
        raise ValueError(
            "IMC requires both gene and disease side information. "
            "Use --gene-side-info and --disease-side-info flags.",
        )
    if args.ppi:
        kwargs["gene_side_info_paths"].append(args.gene_graph_path)
    kwargs["gene_side_info_paths"].extend(args.gene_features_paths)
    kwargs["disease_side_info_paths"].extend(args.disease_side_info_paths)

    dataloader, side_info_loader = pre_processing(
        gene_disease_path=args.gene_disease_path,
        seed=args.seed,
        omim_meta_path=args.omim_meta_path,
        side_info=True,
        zero_sampling_factor=args.zero_sampling_factor,
        num_folds=args.num_folds,
        validation_size=args.validation_size,
        max_dims=args.max_dims,
        **kwargs,
    )

    if args.imc_command == "fine-tune":
        finetune(
            logger=logger,
            output_path=args.output_path,
            dataloader=dataloader,
            side_info_loader=side_info_loader,
            rank=args.rank,
            iterations=args.iterations,
            max_inner_iter=args.max_inner_iter,
            flip_fraction=args.flip_fraction,
            flip_frequency=args.flip_frequency,
            patience=args.patience,
            seed=args.seed,
            n_trials=args.n_trials,
            timeout=args.timeout,
            svd_init=args.svd_init,
        )
    elif args.imc_command == "cv":
        logger.debug("Loading IMC configuration file: %s", args.config_path)
        with args.config_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        config = config["default"]
        regularization_parameters = config.get("regularization_parameters")

        if regularization_parameters is None:
            raise ValueError("Missing 'regularization_parameters' in IMC config.")
        train_eval(
            logger=logger,
            output_path=args.output_path,
            dataloader=dataloader,
            seed=args.seed,
            side_info_loader=side_info_loader,
            rank=args.rank,
            iterations=args.iterations,
            max_inner_iter=args.max_inner_iter,
            flip_fraction=args.flip_fraction,
            flip_frequency=args.flip_frequency,
            patience=args.patience,
            regularization_parameters=regularization_parameters,
            tensorboard_dir=args.tensorboard_dir,
            results_filename=args.results_filename,
            svd_init=args.svd_init,
        )
    else:
        raise ValueError("Invalid IMC command.")
