"""
Utils Module for Scripts
========================

Contains utility functions for scripts.
"""
# pylint: disable=R0913
import logging
from pathlib import Path
from typing import Tuple

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
    input_path: Path,
    seed: int,
    omim_meta_path: Path,
    side_info: bool,
    zero_sampling_factor: int,
    num_folds: int,
    validation_size: float,
) -> Tuple[DataLoader, SideInformationLoader]:
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
        zero_sampling_factor (int): Factor for zero sampling.
        num_folds (int): Number of folds for OMIM2; use None if not applicable.
        validation_size (float): Fraction of data for validation
            (unused data for comparison with NEGA).

    Returns:
        Tuple[DataLoader, SideInformationLoader]: The data loader and,
            if applicable, the side information loader.
    """
    logger = logging.getLogger("pre_processing")
    logger.debug("Loading OMIM metadata.")
    nb_genes, nb_diseases = load_omim_meta(omim_meta_path)

    # Load gene–disease association data.
    dataloader = DataLoader(
        nb_genes=nb_genes,
        nb_diseases=nb_diseases,
        path=input_path / "gene-disease.csv",
        seed=seed,
        zero_sampling_factor=zero_sampling_factor,
        num_folds=num_folds,
        validation_size=validation_size,
    )

    # Load side information if requested.
    side_info_loader = None
    if side_info:
        side_info_loader = SideInformationLoader(
            nb_genes=nb_genes, nb_diseases=nb_diseases
        )
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
