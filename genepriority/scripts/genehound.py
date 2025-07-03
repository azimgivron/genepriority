# pylint: disable=R0914, R0915, R0801, R0913
"""
Reproduce GeneHound results using a MACAU-based approach.

This script performs the following steps:
1. Loads gene–disease association data.
2. Loads side information for genes and diseases.
3. Trains multiple MACAU models using different latent dimensions.

Dependencies:
    - A YAML configuration file (e.g., meta.yaml) for GeneHound parameters.
    - CSV files for gene–disease associations and side information.
    - MACAUTrainer from genepriority.trainer.macau_trainer.
"""

import argparse
import logging
from pathlib import Path

import yaml

from genepriority.preprocessing.dataloader import DataLoader
from genepriority.preprocessing.side_information_loader import \
    SideInformationLoader
from genepriority.scripts.utils import pre_processing
from genepriority.trainer.macau_trainer import MACAUTrainer
from genepriority.utils import serialize


def run(
    dataloader: DataLoader,
    side_info_loader: SideInformationLoader,
    output_path: Path,
    tensorboard_dir: Path,
    seed: int,
    latent_dimension: int,
    results_filename: str,
    config_path: Path,
):
    """
    Configures and runs the MACAU training session.

    This function initializes a MACAUTrainer with the provided parameters and runs
    the training process for each specified latent dimension. The resulting models are
    serialized and saved to disk.

    Args:
        dataloader (DataLoader): The DataLoader containing gene–disease association data.
        side_info_loader (SideInformationLoader): The loader for side information,
            if available.
        output_path (Path): Directory where training outputs will be saved.
        tensorboard_dir (Path): Directory for TensorBoard logs.
        seed (int): Seed for reproducibility.
        latent_dimension (int): Latent dimension of model.
        results_filename (str): Filename to use for saving results.
        config_path (Path): Path to the YAML configuration file.
    """
    logger = logging.getLogger("run")
    logger.debug("Loading configuration file: %s", config_path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    direct = config.get("direct")
    univariate = config.get("univariate")
    save_freq = config.get("save_freq")
    num_samples = config.get("num_samples")
    burnin_period = config.get("burnin_period")

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
    seed = args.seed

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
        max_dims=args.max_dims,
    )
    run(
        dataloader=dataloader,
        side_info_loader=side_info_loader,
        output_path=args.output_path,
        tensorboard_dir=args.tensorboard_dir,
        seed=seed,
        latent_dimension=args.latent_dimension,
        results_filename=args.results_filename,
        config_path=args.config_path,
    )
