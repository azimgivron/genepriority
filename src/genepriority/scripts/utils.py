"""
Utils Module for Scripts
========================

Contains utility functions for scripts.
"""

import argparse

# pylint: disable=R0913
import logging
from pathlib import Path
from typing import List, Tuple

import yaml

from genepriority.preprocessing.dataloader import DataLoader
from genepriority.preprocessing.side_information_loader import SideInformationLoader


def load_omim_meta(omim_meta_path: Path) -> Tuple[int, int]:
    """Load OMIM meta data.

    Args:
        omim_meta_path (Path): The path to the meta data of the
            OMIM dataset.

    Returns:
        Tuple[int, int]: The number of genes and the number of diseases
            in OMIM matrix.
    """
    logger = logging.getLogger("load_omim_meta")
    logger.debug("Loading OMIM meta data: %s", omim_meta_path)
    with omim_meta_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    for key in ["nb_genes", "nb_diseases"]:
        if key not in config:
            raise KeyError(
                f"{key} not found in configuration file. "
                "Make sure it is set before running the script again."
            )

    nb_genes = config["nb_genes"]
    nb_diseases = config["nb_diseases"]
    return nb_genes, nb_diseases


def pre_processing(
    gene_disease_path: Path,
    seed: int,
    omim_meta_path: Path,
    side_info: bool,
    gene_side_info_paths: List[Path],
    disease_side_info_paths: List[Path],
    zero_sampling_factor: int,
    num_folds: int,
    validation_size: float,
    max_dims: int,
    **kwargs,
) -> Tuple[DataLoader, SideInformationLoader]:
    """
    Loads configuration parameters, gene-disease association data, and side information.

    This function performs the following:
      - Loads OMIM metadata (number of genes, diseases, and minimum associations).
      - Initializes a DataLoader to load gene-disease association data.
      - If side_info is True, initializes a SideInformationLoader to process side information
        files.

    Args:
        gene_disease_path (Path):
            Path to the CSV file containing gene-disease associations.
        seed (int):
            Random seed for reproducibility.
        omim_meta_path (Path):
            Path to the OMIM metadata file (e.g., containing counts and thresholds).
        side_info (bool):
            Whether to load and process side information files.
        gene_side_info_paths (List[Path]):
            List of Paths to gene-side information CSV files (only used if side_info=True).
        disease_side_info_paths (List[Path]):
            List of Paths to disease-side information CSV files (only used if side_info=True).
        zero_sampling_factor (int):
            Factor by which to oversample negative (zero) examples.
        num_folds (int):
            Number of folds for OMIM2 cross-validation; use None if not applicable.
        validation_size (float):
            Fraction of data to hold out for validation
            (unused data for comparison with NEGA).
        max_dims (int): Maximum number of dimensions of the side
                information.

    Returns:
        Tuple[DataLoader, SideInformationLoader]:
            - dataloader: DataLoader instance for gene-disease association loading.
            - side_info_loader: SideInformationLoader instance if side_info=True; otherwise None.
    """
    logger = logging.getLogger("pre_processing")
    logger.debug("Loading OMIM metadata.")
    nb_genes, nb_diseases = load_omim_meta(omim_meta_path)

    # Load gene-disease association data.
    dataloader = DataLoader(
        nb_genes=nb_genes,
        nb_diseases=nb_diseases,
        path=gene_disease_path,
        seed=seed,
        zero_sampling_factor=zero_sampling_factor,
        num_folds=num_folds,
        validation_size=validation_size,
    )

    # Load side information if requested.
    side_info_loader = None
    if side_info:
        side_info_loader = SideInformationLoader(
            nb_genes=nb_genes, nb_diseases=nb_diseases, max_dims=max_dims
        )
        side_info_loader.process_side_info(
            gene_side_info_paths=gene_side_info_paths,
            disease_side_info_paths=disease_side_info_paths,
            **kwargs,
        )

    return dataloader, side_info_loader


def csv_file(path: str) -> Path:
    """Validate that the given path exists and points to a CSV file.

    Args:
        path (str): The file-system path to check.

    Returns:
        Path: A pathlib.Path instance for the validated CSV file.

    Raises:
        argparse.ArgumentTypeError: If the path does not exist or does not end with “.csv”.
    """
    file_path = Path(path)
    if not file_path.exists():
        return "DEFAULT-PATH-NOT-FOUND"
    if file_path.suffix.lower() != ".csv":
        raise argparse.ArgumentTypeError(f"'{path}' is not a CSV file.")
    return file_path


def yaml_file(path: str) -> Path:
    """Validate that the given path exists and points to a YAML file.

    Args:
        path (str): The file-system path to check.

    Returns:
        Path: A pathlib.Path instance for the validated YAML file.

    Raises:
        argparse.ArgumentTypeError: If the path does not exist or does not end with
            “.yaml” or “.yml”.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise argparse.ArgumentTypeError(f"'{path}' does not exist.")
    if file_path.suffix.lower() not in (".yaml", ".yml"):
        raise argparse.ArgumentTypeError(f"'{path}' is not a YAML file.")
    return file_path


def output_dir(path: str) -> Path:
    """
    Ensure that the given path exists as a directory, creating it (and any
    missing parent directories) if necessary.

    Args:
        path (str): Filesystem path to the desired output directory.

    Returns:
        Path: A pathlib.Path object for the created or existing directory.

    Raises:
        OSError: If the directory cannot be created due to filesystem errors
                 (e.g. insufficient permissions).
    """
    file_path = Path(path)
    file_path.mkdir(parents=True, exist_ok=True)
    return file_path
