"""
Baseline Training Script
========================

This script preprocesses gene–disease association data and runs a baseline training session
using `BaselineTrainer`.
"""
import argparse
import logging
from pathlib import Path

from genepriority.preprocessing.dataloader import DataLoader
from genepriority.scripts.utils import pre_processing
from genepriority.trainer.baseline_trainer import BaselineTrainer


def run(
    dataloader: DataLoader,
    output_path: Path,
    seed: int,
):
    """
    Execute a baseline training session with cross-validation.

    Args:
        dataloader (DataLoader): The DataLoader containing gene–disease association data.
        output_path (Path): Directory where training outputs will be saved.
        seed (int): Seed for reproducibility.

    Returns:
        None
    """
    logger = logging.getLogger("run")
    logger.debug("Configuring Baseline training session.")
    trainer = BaselineTrainer(
        dataloader=dataloader,
        path=output_path,
        seed=seed,
    )
    results_path = output_path
    results_path.mkdir(parents=True, exist_ok=True)
    trainer.path = results_path
    _ = trainer.train_test_cross_validation(
        num_latent=None,
        save_name=f"baseline:{dataloader.zero_sampling_factor}0s.hdf5",
    )


def baseline(args: argparse.Namespace):
    """
    Main entry point for the baseline script.

    This function validates input, configuration, and metadata paths, processes the raw data
    via `pre_processing`, and invokes the `run` function to perform baseline training.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Raises:
        FileNotFoundError: If any of the specified paths (input, config, or metadata)
        do not exist.

    Returns:
        None
    """
    output_path = Path(args.output_path).absolute()
    output_path.mkdir(parents=True, exist_ok=True)

    seed = args.seed

    input_path = Path(args.input_path).absolute()
    if not input_path.exists():
        raise FileNotFoundError(f"The input path does not exist: {input_path}")

    config_path = Path(args.config_path).absolute()
    if not config_path.exists():
        raise FileNotFoundError(f"The configuration path does not exist: {config_path}")

    omim_meta_path = Path(args.omim_meta_path).absolute()
    if not omim_meta_path.exists():
        raise FileNotFoundError(f"OMIM metadata path does not exist: {omim_meta_path}")

    dataloader, _ = pre_processing(
        input_path=input_path,
        seed=seed,
        omim_meta_path=omim_meta_path,
        side_info=args.side_info,
        zero_sampling_factor=args.zero_sampling_factor,
        num_folds=args.num_folds,
        validation_size=args.validation_size,
    )
    run(
        dataloader=dataloader,
        output_path=output_path,
        seed=seed,
    )
