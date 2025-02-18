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
from typing import List, Optional, Tuple

import yaml

from genepriority.preprocessing.dataloader import DataLoader
from genepriority.preprocessing.side_information_loader import SideInformationLoader
from genepriority.trainer.macau_trainer import MACAUTrainer
from genepriority.scripts.utils import load_omim_meta
from genepriority.utils import serialize


def pre_processing(
    input_path: Path,
    seed: int,
    omim_meta_path: Path,
    side_info: bool,
    num_splits: Optional[int],
    zero_sampling_factor: int,
    num_folds: Optional[int],
    train_size: float,
    validation_size: Optional[float],
) -> Tuple[DataLoader, Optional[SideInformationLoader]]:
    """
    Loads configuration parameters, gene–disease association data, and side information.

    This function performs the following:
      - Loads OMIM metadata (number of genes, diseases, and minimum associations).
      - Initializes a DataLoader to load gene–disease data.
      - If side_info is True, initializes a SideInformationLoader to process side information files.

    Args:
        input_path (Path): Directory containing input CSV files.
        seed (int): Seed for reproducibility.
        omim_meta_path (Path): Path to the OMIM metadata file.
        side_info (bool): Whether to load side information.
        num_splits (Optional[int]): Number of splits for OMIM1; use None if not applicable.
        zero_sampling_factor (int): Factor for zero sampling.
        num_folds (Optional[int]): Number of folds for OMIM2; use None if not applicable.
        train_size (float): Fraction of data to use for training.
        validation_size (Optional[float]): Fraction of data for validation (unused data for comparison with NEGA).

    Returns:
        Tuple[DataLoader, Optional[SideInformationLoader]]: The data loader and, if applicable, the side information loader.
    """
    logger = logging.getLogger("pre_processing")
    logger.debug("Loading OMIM metadata.")
    nb_genes, nb_diseases, min_associations = load_omim_meta(omim_meta_path)

    # Load gene–disease association data.
    dataloader = DataLoader(
        nb_genes=nb_genes,
        nb_diseases=nb_diseases,
        path=input_path / "gene-disease.csv",
        seed=seed,
        num_splits=num_splits,
        zero_sampling_factor=zero_sampling_factor,
        num_folds=num_folds,
        train_size=train_size,
        min_associations=min_associations,
        validation_size=validation_size,
    )
    # dataloader(filter_column="Disease ID")
    dataloader.load_omim1()
    
    # Load side information if requested.
    side_info_loader: SideInformationLoader = None
    if side_info:
        side_info_loader = SideInformationLoader(nb_genes=nb_genes, nb_diseases=nb_diseases)
        side_info_loader.process_side_info(
            gene_side_info_paths=[
                input_path / "interpro.csv",
                input_path / "uniprot.csv",
                input_path / "go.csv",
            ],
            disease_side_info_paths=[input_path / "phenotype.csv"],
            names=["interpro", "uniprot", "GO", "phenotype"],
        )
    return dataloader, side_info_loader


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
        side_info_loader (Optional[SideInformationLoader]): The loader for side information, if available.
        output_path (Path): Directory where training outputs will be saved.
        tensorboard_dir (Path): Directory for TensorBoard logs.
        seed (int): Seed for reproducibility.
        latent_dimensions (List[int]): List of latent dimensions for model training.
        results_filename (str): Filename to use for saving results.
        config_path (Path): Path to the YAML configuration file.
        is_omim1 (bool): True if using OMIM1 (multiple splits), False if using OMIM2 (cross-validation).
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
    train_method = trainer.train_test_splits if is_omim1 else trainer.train_test_cross_validation

    for latent in latent_dimensions:
        results_path = output_path / str(latent)
        results_path.mkdir(parents=True, exist_ok=True)
        trainer.path = results_path
        result = train_method(
            num_latent=latent,
            save_name=f"latent={latent}:model-omim{int(not is_omim1) + 1}.hdf5",
        )
        serialize(result, results_path / results_filename)
        logger.debug("Serialized results for latent dimension %s saved successfully.", latent)


def parse_genehound(subparsers: argparse._SubParsersAction) -> None:
    """
    Parses command-line arguments for the GeneHound reproduction pipeline.

    This function adds two subparsers:
      - "genehound-omim1": Run GeneHound using the OMIM1 dataset with multiple splits.
      - "genehound-omim2": Run GeneHound using the filtered OMIM2 dataset with cross-validation.

    Args:
        subparsers (argparse._SubParsersAction): The subparsers object to which the genehound commands will be added.
    """
    omim1_parser = subparsers.add_parser(
        "genehound-omim1",
        help="Run GeneHound with the OMIM1 dataset (multiple splits).",
    )
    omim2_parser = subparsers.add_parser(
        "genehound-omim2",
        help="Run GeneHound with the filtered OMIM2 dataset (cross-validation).",
    )

    omim1_parser.add_argument(
        "--num-splits",
        type=int,
        default=6,
        help="Number of data splits to use (default: %(default)s).",
    )
    omim2_parser.add_argument(
        "--num-folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation (default: %(default)s).",
    )

    for subparser in [omim1_parser, omim2_parser]:
        subparser.add_argument(
            "--output-path",
            type=str,
            required=True,
            help="Directory to save output results.",
        )
        subparser.add_argument(
            "--zero-sampling-factor",
            type=int,
            required=True,
            help="Zero sampling factor for the gene–disease matrix.",
        )
        subparser.add_argument(
            "--side-info",
            action="store_true",
            help="Include side information for genes and diseases (default: %(default)s).",
        )
        subparser.add_argument(
            "--input-path",
            type=str,
            default="/home/TheGreatestCoder/code/data/postprocessed/",
            help="Directory containing input data files (default: %(default)s).",
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
            "--config-path",
            type=str,
            default=(
                "/home/TheGreatestCoder/code/genepriority/"
                "configurations/genehound/meta.yaml"
            ),
            help="Path to the YAML configuration file for GeneHound (default: %(default)s).",
        )
        subparser.add_argument(
            "--tensorboard-dir",
            type=str,
            default="/home/TheGreatestCoder/code/logs",
            help="Directory for TensorBoard logs (default: %(default)s).",
        )
        subparser.add_argument(
            "--results-filename",
            type=str,
            default="results.pickle",
            help="Filename for serialized results (default: %(default)s).",
        )
        subparser.add_argument(
            "--latent-dimensions",
            type=int,
            nargs="+",
            default=[25, 30, 40],
            help="List of latent dimensions for MACAU (default: %(default)s).",
        )
        subparser.add_argument(
            "--train-size",
            type=float,
            default=0.8,
            help="Proportion of data to use for training (default: %(default)s).",
        )
        subparser.add_argument(
            "--validation-size",
            type=float,
            default=None,
            help="Proportion of data for validation (unused for comparison with NEGA) (default: %(default)s).",
        )
        subparser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed for reproducibility (default: %(default)s).",
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
        num_splits=args.num_splits if hasattr(args, "num_splits") else None,
        zero_sampling_factor=args.zero_sampling_factor,
        num_folds=args.num_folds if hasattr(args, "num_folds") else None,
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
