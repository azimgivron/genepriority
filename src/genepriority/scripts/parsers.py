"""
Parser Module
=============
"""

import argparse
from pathlib import Path

import numpy as np

from genepriority.scripts.utils import csv_file, output_dir, yaml_file

DEFAULT_PATHS = {
    "GENE": [
        "/home/TheGreatestCoder/code/input/gene_rifs.csv",
        "/home/TheGreatestCoder/code/input/go.csv",
        "/home/TheGreatestCoder/code/input/reactome.csv",
        "/home/TheGreatestCoder/code/input/swissprot.csv",
    ],
    "DISEASE": [
        "/home/TheGreatestCoder/code/input/hpo.csv",
        "/home/TheGreatestCoder/code/input/medgen.csv",
        "/home/TheGreatestCoder/code/input/mondo.csv",
    ],
    "OMIM": "/home/TheGreatestCoder/code/input/gene_disease.csv",
}


def data_info(parser: argparse.ArgumentParser):
    """Parse data related information

    Args:
        parser (argparse.ArgumentParser): The parser.
    """
    parser.add_argument(
        "--num-folds",
        type=int,
        required=True,
        help="Number of folds for cross-validation.",
    )
    parser.add_argument(
        "--zero-sampling-factor",
        type=int,
        required=True,
        help=(
            "Factor for zero sampling (number of zeros = factor * " "number of ones). "
        ),
    )
    parser.add_argument(
        "--max_dims",
        type=int,
        default=None,
        help=(
            "The maximum number of dimension to use in the factorization of the side features. "
            "Default is None, meaning no factorization is made. "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--side-info",
        action="store_true",
        help="Include side information for genes and diseases (default: %(default)s).",
    )
    parser.add_argument(
        "--gene-side-info-paths",
        metavar="FILE",
        nargs="+",
        type=csv_file,
        default=[csv_file(file) for file in DEFAULT_PATHS["GENE"]],
        help="Paths to one or more gene-side information CSV files (default: %(default)s).",
    )
    parser.add_argument(
        "--disease-side-info-paths",
        metavar="FILE",
        nargs="+",
        type=csv_file,
        default=[csv_file(file) for file in DEFAULT_PATHS["DISEASE"]],
        help="Paths to one or more disease-side information CSV files (default: %(default)s).",
    )
    parser.add_argument(
        "--gene-disease-path",
        metavar="FILE",
        type=csv_file,
        default=csv_file(DEFAULT_PATHS["OMIM"]),
        help="Path of the gene disease matrix (default: %(default)s).",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.1,
        help=(
            "Proportion of data for validation (unused for comparison with NEGA)"
            " (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: %(default)s).",
    )
    parser.add_argument(
        "--omim-meta-path",
        metavar="FILE",
        type=yaml_file,
        default=yaml_file(
            "/home/TheGreatestCoder/code/genepriority/configurations/omim.yaml"
        ),
        help="Path to the OMIM metadata file (default: %(default)s).",
    )


def parse_genehound(subparsers: argparse._SubParsersAction):
    """
    Parses command-line arguments for the GeneHound reproduction pipeline.

    Args:
        parser (argparse._SubParsersAction): The parser for genehound.
    """
    parser = subparsers.add_parser(
        "genehound",
        help="Run GeneHound on OMIM dataset in the cross-validation setting.",
    )
    parser.add_argument(
        "--output-path",
        metavar="FILE",
        type=output_dir,
        required=True,
        help="Directory to save output results.",
    )
    data_info(parser)
    parser.add_argument(
        "--log-filename",
        type=str,
        default="pipeline.log",
        help="Filename for log output (default: %(default)s).",
    )
    parser.add_argument(
        "--config-path",
        metavar="FILE",
        type=yaml_file,
        default=yaml_file(
            "/home/TheGreatestCoder/code/genepriority/"
            "configurations/genehound/meta.yaml"
        ),
        help="Path to the YAML configuration file for GeneHound (default: %(default)s).",
    )
    parser.add_argument(
        "--tensorboard-dir",
        metavar="FILE",
        type=output_dir,
        default=output_dir("/home/TheGreatestCoder/code/logs"),
        help="Directory for TensorBoard logs (default: %(default)s).",
    )
    parser.add_argument(
        "--results-filename",
        type=str,
        default="results.pickle",
        help="Filename for serialized results (default: %(default)s).",
    )
    parser.add_argument(
        "--latent-dimension",
        type=int,
        default=40,
        help="Size of the latent dimension (default: %(default)s).",
    )


def parse_nega(subparsers: argparse._SubParsersAction):
    """
    Adds the NEGA subcommand with two subsubcommands: "cv" and "fine-tune".

    Subsubcommands:
      - "cv": Train and evaluate the NEGA model using a cross-validation setting.
      - "fine-tune": Run hyperparameter tuning for NEGA.

    Args:
        subparsers (argparse._SubParsersAction): The subparsers object to which
            the NEGA command will be added.
    """
    # Add the main "nega" subcommand
    nega_parser = subparsers.add_parser(
        "nega",
        help="Run NEGA on OMIM dataset.",
    )
    nega_subparsers = nega_parser.add_subparsers(dest="nega_command", required=True)

    # Subsubcommand for hyperparameter tuning ("fine-tune")
    fine_tune_parser = nega_subparsers.add_parser(
        "fine-tune", help="Perform search for hyperparameter tuning of NEGA."
    )

    # Subsubcommand for cross-validation ("cv")
    cv_parser = nega_subparsers.add_parser(
        "cv", help="Train and evaluate the NEGA model on a cross-validation setting."
    )
    # Add common arguments to both "cv" and "fine-tune"
    for parser in [fine_tune_parser, cv_parser]:
        parser.add_argument(
            "--output-path",
            metavar="FILE",
            type=output_dir,
            required=True,
            help="Directory to save output results.",
        )
        data_info(parser)
        parser.add_argument(
            "--log-filename",
            type=str,
            default="pipeline.log",
            help="Filename for log output (default: %(default)s).",
        )
        parser.add_argument(
            "--rank",
            type=int,
            default=40,
            help="Rank (number of latent factors) for the model (default: %(default)s).",
        )
        parser.add_argument(
            "--iterations",
            type=int,
            default=200,
            help="Number of training iterations (default: %(default)s).",
        )
        parser.add_argument(
            "--threshold",
            type=int,
            default=10,
            help="Threshold parameter for the model (default: %(default)s).",
        )
        parser.add_argument(
            "--flip_fraction",
            type=float,
            default=None,
            help=(
                "Fraction of observed positive training entries to flip to negatives "
                "(zeros) to simulate label noise. Must be between 0 and 1. (default: %(default)s)."
            ),
        )
        parser.add_argument(
            "--flip_frequency",
            type=int,
            default=None,
            help=(
                "The frequency at which to resample the observed positive entries in the training "
                "mask to be flipped to negatives. (default: %(default)s)."
            ),
        )
        parser.add_argument(
            "--patience",
            type=int,
            default=None,
            help=(
                "The number of recent epochs/iterations to consider when evaluating the stopping "
                "condition. Default is None, meaning no early stopping is used. "
                "(default: %(default)s)."
            ),
        )
        parser.add_argument(
            "--svd-init",
            action="store_true",
            help=(
                "Instanciate the matrix factorization using SVD decomposition "
                "rather than an random initialization (default: %(default)s)."
            ),
        )
        parser.add_argument(
            "--formulation",
            choices=["imc", "genehound"],
            default="imc",
            help=(
                "Whether to use the Genehound-like or IMC-like objective function"
                " formulation of NEGA (default: %(default)s)."
            ),
        )

    # Additional arguments specific to the "cv" subcommand
    cv_parser.add_argument(
        "--tensorboard-dir",
        metavar="FILE",
        type=output_dir,
        default=output_dir("/home/TheGreatestCoder/code/logs"),
        help="Directory for TensorBoard logs (default: %(default)s).",
    )
    cv_parser.add_argument(
        "--config-path",
        metavar="FILE",
        type=yaml_file,
        default=yaml_file(
            "/home/TheGreatestCoder/code/genepriority/configurations/nega/meta.yaml"
        ),
        help=(
            "Path to the YAML configuration file containing simulation parameters "
            "(default: %(default)s)."
        ),
    )
    cv_parser.add_argument(
        "--results-filename",
        type=str,
        default="results.pickle",
        help="Filename for serialized results (default: %(default)s).",
    )

    # Additional arguments specific to the "fine-tune" subcommand
    fine_tune_parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of trials for hyperparameter tuning (default: %(default)s).",
    )
    fine_tune_parser.add_argument(
        "--timeout",
        type=int,
        default=12,
        help="Number of hours after which to stop the search (default: %(default)s).",
    )


def parse_post(subparsers: argparse._SubParsersAction):
    """
    This command loads serialized Evaluation objects, a YAML configuration file
    containing alpha values, and then produces metric
    plots and tables. It ensures that the number of evaluation paths matches
    the number of provided model names.

    Args:
        subparsers (argparse._SubParsersAction): The subparsers object to which
            the NEGA command will be added.
    """
    parser = subparsers.add_parser(
        "post",
        help="Run Post Processing.",
    )
    parser.add_argument(
        "--output-path",
        metavar="FILE",
        type=output_dir,
        required=True,
        help="Directory to save output results.",
    )
    parser.add_argument(
        "--evaluation-paths",
        type=str,
        nargs="+",
        required=True,
        help="One or more paths to serialized `Evaluation` objects.",
    )
    parser.add_argument(
        "--model-names",
        type=str,
        nargs="+",
        required=True,
        help="One or more model names corresponding to the evaluation paths (in the same order).",
    )
    parser.add_argument(
        "--post-config-path",
        metavar="FILE",
        type=yaml_file,
        default=yaml_file(
            "/home/TheGreatestCoder/code/genepriority/configurations/post.yaml"
        ),
        help=(
            "Path to the post-processing configuration file containing alpha values."
            " (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--no-sharey",
        action="store_false",
        help="Whether to share the y axis for BEDROC boxplots (default: %(default)s).",
    )
    parser.add_argument(
        "--full",
        action="store_false",
        help=(
            "If flagged, assessment is made on whole completed matrix instead of the"
            " test set only (default: %(default)s)."
        ),
    )


def parse_nn(subparsers: argparse._SubParsersAction):
    """
    This command run the Deep matrix completion algorithm: Neural Collaborative Filtering.

    Args:
        subparsers (argparse._SubParsersAction): The subparsers object to which
            the NEGA command will be added.
    """
    parser = subparsers.add_parser(
        "ncf",
        help="Run NNeural Collaborative FilteringEGA on OMIM dataset in the cross-validation setting.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Base output directory for results.",
    )
    parser.add_argument(
        "--log-filename",
        type=str,
        default="pipeline.log",
        help="Filename for log output (default: %(default)s).",
    )
    data_info(parser)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="The batch size (default: %(default)s).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="The learning rate (default: %(default)s).",
    )
    parser.add_argument(
        "--lr_patience",
        type=int,
        default=50,
        help="The learning rate patience before reducing it (default: %(default)s).",
    )
    parser.add_argument(
        "--lr_factor",
        type=float,
        default=1 / np.sqrt(10),
        help="The learning rate reduing factor (default: %(default)s).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="The number of epochs (default: %(default)s).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="The patience for early stopping (default: %(default)s).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="The dropout probability (default: %(default)s).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("/home/TheGreatestCoder/code/logs"),
        help="Base TensorBoard log directory (default: %(default)s).",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
        help="Embedding dimension (default: %(default)s).",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[128, 64, 32],
        help="List of MLP hidden layer sizes (default: %(default)s).",
    )
    parser.add_argument(
        "--load-model",
        type=Path,
        default=None,
        help="Path to a .pt/.pth checkpoint to load before training.",
    )
    parser.add_argument(
        "--save-model",
        type=Path,
        default=None,
        help="Path where to save the trained model state_dict.",
    )
