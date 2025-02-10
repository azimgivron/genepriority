"""
Utils Module for Scripts
========================

Contains utility functions for scripts.
"""
import logging
from pathlib import Path
from typing import Tuple

import yaml


def load_omim_meta(omim_meta_path: Path) -> Tuple[int, int, int]:
    """Load OMIM meta data.

    Args:
        omim_meta_path (Path): _description_

    Returns:
        _type_: _description_
    """
    logger = logging.getLogger("load_omim_meta")
    logger.debug("Loading OMIM meta data: %s", omim_meta_path)
    with omim_meta_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    for key in ["nb_genes", "nb_diseases", "min_associations"]:
        if key not in config:
            raise KeyError(
                f"{key} not found in configuration file. "
                "Make sure it is set before running the script again."
            )

    nb_genes = config["nb_genes"]
    nb_diseases = config["nb_diseases"]
    min_associations = config["min_associations"]
    return nb_genes, nb_diseases, min_associations
