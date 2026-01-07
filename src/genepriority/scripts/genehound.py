# pylint: disable=R0914, R0915, R0801, R0913
"""
Run GeneHound training workflows.

This launcher provides high-level commands for:
  * Hyper-parameter search via Optuna (``fine-tune`` subcommand).
  * Cross-validated training/evaluation using predefined configurations (``cv`` subcommand).
"""

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import pint
import yaml

from genepriority.preprocessing.dataloader import DataLoader
from genepriority.preprocessing.side_information_loader import \
    SideInformationLoader
from genepriority.scripts.utils import pre_processing
from genepriority.trainer.macau_trainer import MACAUTrainer
from genepriority.utils import serialize

def train_eval(
    logger: logging.Logger,
    dataloader: DataLoader,
    side_info_loader: SideInformationLoader,
    output_path: Path,
    tensorboard_dir: Path,
    seed: int,
    latent_dimension: int,
    results_filename: str,
    num_samples: int,
    burnin_period: int,
    direct: bool,
    univariate: bool,
    save_freq: int,
):
    """
    Configures and runs the MACAU training session.

    This function initializes a MACAUTrainer with the provided parameters and runs
    the training process for each specified latent dimension. The resulting models are
    serialized and saved to disk.

    Args:
        logger (logging.Logger): Logger for output messages.
        dataloader (DataLoader): The DataLoader containing gene–disease association data.
        side_info_loader (SideInformationLoader): The loader for side information,
            if available.
        output_path (Path): Directory where training outputs will be saved.
        tensorboard_dir (Path): Directory for TensorBoard logs.
        latent_dimension (int): Latent dimension of model.
        results_filename (str): Filename to use for saving results.
        num_samples (int): The number of posterior samples to draw during training.
        burnin_period (int): The number of burn-in iterations before collecting
            posterior samples.
        direct (bool): Whether to use a Cholesky or CG solver.
        univariate (bool): Whether to use univariate or multivariate sampling.
        seed (int): The random seed for reproducibility.
        save_freq (int): The frequency at which the model state is saved.
    """
    logger.debug("Configuring MACAU training session.")
    trainer = MACAUTrainer(
        dataloader=dataloader,
        side_info_loader=side_info_loader,
        path=output_path,
        num_samples=num_samples,
        burnin_period=burnin_period,
        direct=direct,
        univariate=univariate,
        seed=seed,
        save_freq=save_freq,
        verbose=0,
        tensorboard_dir=tensorboard_dir,
    )

    results_path = output_path / str(latent_dimension)
    results_path.mkdir(parents=True, exist_ok=True)
    trainer.path = results_path
    result = trainer.train_test_cross_validation(
        num_latent=latent_dimension,
        save_name=f"genehound:latent={latent_dimension}.hdf5",
    )
    serialize(result, results_path / results_filename)
    logger.debug(
        "Serialized results for latent dimension %s saved successfully.",
        latent_dimension,
    )


def genehound(args: argparse.Namespace):
    """
    Executes the GeneHound reproduction pipeline.

    This function performs the following steps:
      1. Sets up output directories and logging.
      2. Loads gene–disease association data and side information via pre_processing().
      3. Runs the MACAU training session with the specified latent dimensions.
      4. Serializes and saves the resulting models.

    Errors encountered during processing are logged and re-raised.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    logger = logging.getLogger("GeneHound")
    kwargs = defaultdict(list)

    if args.ppi:
        kwargs["gene_side_info_paths"].append(args.gene_graph_path)
    if args.gene_side_info:
        kwargs["gene_side_info_paths"].extend(args.gene_features_paths)
    if args.disease_side_info:
        kwargs["gene_side_info_paths"].extend(args.disease_side_info_paths)

    dataloader, side_info_loader = pre_processing(
        gene_disease_path=args.gene_disease_path,
        seed=args.seed,
        omim_meta_path=args.omim_meta_path,
        side_info=args.gene_side_info or args.disease_side_info or args.ppi,
        zero_sampling_factor=args.zero_sampling_factor,
        num_folds=args.num_folds,
        validation_size=args.validation_size,
        max_dims=args.max_dims,
        **kwargs,
    )

    logger.debug("Loading configuration file: %s", args.config_path)
    with args.config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    direct = config.get("direct")
    univariate = config.get("univariate")
    save_freq = config.get("save_freq")
    num_samples = config.get("num_samples")
    burnin_period = config.get("burnin_period")

    train_eval(
        logger=logger,
        dataloader=dataloader,
        side_info_loader=side_info_loader,
        seed=args.seed,
        output_path=args.output_path,
        tensorboard_dir=args.tensorboard_dir,
        latent_dimension=args.latent_dimension,
        results_filename=args.results_filename,
        direct=direct,
        univariate=univariate,
        save_freq=save_freq,
        num_samples=num_samples,
        burnin_period=burnin_period,
    )
