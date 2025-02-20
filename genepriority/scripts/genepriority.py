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
import logging
import traceback
from pathlib import Path
from typing import Any

from genepriority.scripts.genehound import genehound
from genepriority.scripts.nega import nega
from genepriority.scripts.post import post
from genepriority.scripts.parsers import parse_genehound, parse_nega, parse_post


def setup_logger(args: Any) -> None:
    """
    Configures the root logger to write logs to a file and the console.

    Args:
        args (Any): Arguments from argparser.
    """
    output_path: Path = Path(args.output_path).absolute()
    output_path.mkdir(exist_ok=True)

    log_file: Path = output_path / args.log_filename

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - [%(funcName)s] - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w")],
    )


def main() -> None:
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

    args: Any = parser.parse_args()
    try:
        if "genehound" in args.algorithm_command:
            setup_logger(args)
            genehound(args)
        elif "nega" in args.algorithm_command:
            setup_logger(args)
            nega(args)
        elif "post" in args.algorithm_command:
            args = parser.parse_args()
            if len(args.evaluation_paths) != len(args.model_names):
                parser.error(
                    "The number of evaluation paths must match the number of model names."
                )
            post(args)
        else:
            raise ValueError(f"No such command: {args.algorithm_command}")
    except Exception as exception:
        logger = logging.getLogger("genepriority")
        logger.error("An error occurred during processing: %s", exception)
        logger.error("%s", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
