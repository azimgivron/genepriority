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
from typing import List, Optional

import yaml

from genepriority.preprocessing.dataloader import DataLoader
from genepriority.preprocessing.side_information_loader import SideInformationLoader
from genepriority.scripts.utils import pre_processing
from genepriority.trainer.macau_trainer import MACAUTrainer
from genepriority.utils import serialize


def run(
    dataloader: DataLoader,
    side_info_loader: Optional[SideInformationLoader],
    output_path: Path,
    tensorboard_dir: Path,
    seed: int,
    latent_dimensions: List[int],
    results_filename: str,
    config_path: Path,
    is_omim1: bool,
) -> None:
    """
    Configures and runs the MACAU training session.

    This function initializes a MACAUTrainer with the provided parameters and runs
    the training process for each specified latent dimension. The resulting models are
    serialized and saved to disk.

    Args:
        dataloader (DataLoader): The DataLoader containing gene–disease association data.
        side_info_loader (Optional[SideInformationLoader]): The loader for side information,
            if available.
        output_path (Path): Directory where training outputs will be saved.
        tensorboard_dir (Path): Directory for TensorBoard logs.
        seed (int): Seed for reproducibility.
        latent_dimensions (List[int]): List of latent dimensions for model training.
        results_filename (str): Filename to use for saving results.
        config_path (Path): Path to the YAML configuration file.
        is_omim1 (bool): True if using OMIM1 (multiple splits), False if using OMIM2
            (cross-validation).
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
    train_method = (
        trainer.train_test_splits if is_omim1 else trainer.train_test_cross_validation
    )

    for latent in latent_dimensions:
        results_path = output_path / str(latent)
        results_path.mkdir(parents=True, exist_ok=True)
        trainer.path = results_path
        result = train_method(
            num_latent=latent,
            save_name=f"latent={latent}:model-omim{int(not is_omim1) + 1}.hdf5",
        )
        serialize(result, results_path / results_filename)
        logger.debug(
            "Serialized results for latent dimension %s saved successfully.", latent
        )


def genehound(args: argparse.Namespace) -> None:
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
    output_path: Path = Path(args.output_path).absolute()
    output_path.mkdir(parents=True, exist_ok=True)

    latent_dimensions: List[int] = args.latent_dimensions

    tensorboard_dir: Path = Path(args.tensorboard_dir).absolute()
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    seed: int = args.seed

    input_path: Path = Path(args.input_path).absolute()
    if not input_path.exists():
        raise FileNotFoundError(f"The input path does not exist: {input_path}")

    config_path: Path = Path(args.config_path).absolute()
    if not config_path.exists():
        raise FileNotFoundError(f"The configuration path does not exist: {config_path}")

    omim_meta_path: Path = Path(args.omim_meta_path).absolute()
    if not omim_meta_path.exists():
        raise FileNotFoundError(f"OMIM metadata path does not exist: {omim_meta_path}")

    dataloader, side_info_loader = pre_processing(
        input_path=input_path,
        seed=seed,
        omim_meta_path=omim_meta_path,
        side_info=args.side_info,
        num_splits=args.num_splits if "num_splits" in args else None,
        zero_sampling_factor=args.zero_sampling_factor,
        num_folds=args.num_folds if "num_folds" in args else None,
        train_size=args.train_size,
        validation_size=args.validation_size,
    )
    run(
        dataloader=dataloader,
        side_info_loader=side_info_loader,
        output_path=output_path,
        tensorboard_dir=tensorboard_dir,
        seed=seed,
        latent_dimensions=latent_dimensions,
        results_filename=args.results_filename,
        config_path=config_path,
        is_omim1=args.algorithm_command == "genehound-omim1",
    )
