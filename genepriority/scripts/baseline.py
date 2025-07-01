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
from genepriority.utils import serialize


def run(
    dataloader: DataLoader,
    output_path: Path,
    seed: int,
    results_filename: str,
):
    """
    Execute a baseline training session with cross-validation.

    Args:
        dataloader (DataLoader): The DataLoader containing gene–disease association data.
        output_path (Path): Directory where training outputs will be saved.
        seed (int): Seed for reproducibility.
        results_filename (str): Filename to use for saving results.

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
    result = trainer.train_test_cross_validation(
        num_latent=None,
        save_name=f"baseline:{dataloader.zero_sampling_factor}0s.pickle",
    )
    serialize(result, results_path / results_filename)


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
    seed = args.seed

    dataloader, _ = pre_processing(
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
    run(
        dataloader=dataloader,
        output_path=args.output_path,
        seed=seed,
        results_filename=args.results_filename,
    )
