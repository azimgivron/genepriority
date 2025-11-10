"""
Main module for the Gene Prioritization Tool.

This tool implements matrix completion algorithms for gene prioritization on the OMIM
gene-disease matrix. It also supports integrating side information (such as gene features
and disease phenotypes) to enhance prediction accuracy.

Implemented subcommands:
  1. GeneHound (based on MACAU)
  2. Non-Euclidean Gradient Algorithm (NEGA)
  3. Post-processing of evaluation results
"""

import argparse
import datetime
import logging
import traceback
from pathlib import Path
from typing import Any

import pytz

from genepriority.scripts.genehound import genehound
from genepriority.scripts.ncf import ncf
from genepriority.scripts.nega import nega
from genepriority.scripts.parsers import (
    parse_genehound,
    parse_nega,
    parse_ncf,
    parse_post,
)
from genepriority.scripts.post import post


def setup_logger(args: Any):
    """
    Configures the root logger to write logs to a file and the console
    with timestamps in the Brussels time zone.

    Args:
        args (Any): Arguments from the argument parser. Expected to have attributes:
            - output_path: The directory path where log files will be stored.
            - log_filename: The name of the log file.
    """
    output_path = Path(args.output_path).absolute()
    output_path.mkdir(parents=True, exist_ok=True)

    log_file = output_path / args.log_filename

    def brussels_time(*_) -> Any:
        """Return the current time as a time tuple in Brussels time zone."""
        return (
            datetime.datetime.now(datetime.timezone.utc)
            .astimezone(pytz.timezone("Europe/Brussels"))
            .timetuple()
        )

    file_handler = logging.FileHandler(log_file, mode="w")
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - [%(funcName)s] - %(levelname)s - %(message)s"
    )
    formatter.converter = brussels_time
    file_handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[file_handler],
    )


def main():
    """
    Main entry point for the Gene Prioritization Tool.

    Sets up the command-line argument parser with subcommands and dispatches the
    execution to the corresponding function based on the chosen algorithm.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Gene Prioritization Tool\n\n"
            "This tool implements matrix completion algorithms for gene prioritization on the "
            "OMIM gene-disease matrix. It supports integration of side information, such as gene "
            "features and disease phenotypes, to improve prediction accuracy.\n\n"
            "Implemented subcommands:\n"
            "  1. GeneHound (based on MACAU)\n"
            "  2. Non-Euclidean Gradient Algorithm (NEGA)\n"
            "  3. Post-processing of evaluation results"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="algorithm_command", required=True)

    # Set up subparsers for each subcommand
    parse_genehound(subparsers)
    parse_nega(subparsers)
    parse_post(subparsers)
    parse_ncf(subparsers)

    args: Any = parser.parse_args()
    try:
        if "genehound" in args.algorithm_command:
            setup_logger(args)
            genehound(args)
        elif "nega" in args.algorithm_command:
            setup_logger(args)
            nega(args)
        elif "post" in args.algorithm_command:
            post(args)
        elif "ncf" in args.algorithm_command:
            setup_logger(args)
            ncf(args)
        else:
            raise ValueError(f"No such command: {args.algorithm_command}")
    except Exception as exception:
        logger = logging.getLogger("genepriority")
        logger.error("An error occurred during processing: %s", exception)
        logger.error("%s", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
