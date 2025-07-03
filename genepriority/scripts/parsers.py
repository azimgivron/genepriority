"""
Parser Module
=============
"""
import argparse

from genepriority.scripts.utils import csv_file, output_dir, yaml_file


def parse_genehound(subparsers: argparse.ArgumentParser):
    """
    Parses command-line arguments for the GeneHound reproduction pipeline.

    Args:
        parser (argparse.ArgumentParser): The parser for genehound.
    """
    parser = subparsers.add_parser(
        "genehound",
        help="Run GeneHound on OMIM dataset in the cross-validation setting.",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        required=True,
        help="Number of folds for cross-validation.",
    )
    parser.add_argument(
        "--output-path",
        metavar="FILE",
        type=output_dir,
        required=True,
        help="Directory to save output results.",
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
        "--side-info",
        action="store_true",
        help="Include side information for genes and diseases (default: %(default)s).",
    )
    parser.add_argument(
        "--gene-side-info-paths",
        metavar="FILE",
        nargs="+",
        type=csv_file,
        default=[
            csv_file(file)
            for file in [
                "/home/TheGreatestCoder/code/data/postprocessed/interpro.csv",
                "/home/TheGreatestCoder/code/data/postprocessed/uniprot.csv",
                "/home/TheGreatestCoder/code/data/postprocessed/go.csv",
            ]
        ],
        help="Paths to one or more gene-side information CSV files (default: %(default)s).",
    )
    parser.add_argument(
        "--disease-side-info-paths",
        metavar="FILE",
        nargs="+",
        type=csv_file,
        default=[
            csv_file(file)
            for file in ["/home/TheGreatestCoder/code/data/postprocessed/phenotype.csv"]
        ],
        help="Paths to one or more disease-side information CSV files (default: %(default)s).",
    )
    parser.add_argument(
        "--gene-disease-path",
        metavar="FILE",
        type=csv_file,
        default=csv_file(
            "/home/TheGreatestCoder/code/data/postprocessed/gene-disease.csv"
        ),
        help="Path of the gene disease matrix (default: %(default)s).",
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
            "--num-folds",
            type=int,
            required=True,
            help="Number of folds.",
        )
        parser.add_argument(
            "--output-path",
            metavar="FILE",
            type=output_dir,
            required=True,
            help="Directory to save output results.",
        )
        parser.add_argument(
            "--zero-sampling-factor",
            type=int,
            required=True,
            help=(
                "Factor for zero sampling (number of zeros = factor * number of ones)."
            ),
        )
        parser.add_argument(
            "--gene-disease-path",
            metavar="FILE",
            type=csv_file,
            default=csv_file(
                "/home/TheGreatestCoder/code/data/postprocessed/gene-disease.csv"
            ),
            help="Path of the gene disease matrix (default: %(default)s).",
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
            "--validation-size",
            type=float,
            default=0.1,
            help="Proportion of data to use for validation (default: %(default)s).",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed for reproducibility (default: %(default)s).",
        )
        parser.add_argument(
            "--side-info",
            action="store_true",
            help="Include side information for genes and diseases.",
        )
        parser.add_argument(
            "--gene-side-info-paths",
            metavar="FILE",
            nargs="+",
            type=csv_file,
            default=[
                csv_file(file)
                for file in [
                    "/home/TheGreatestCoder/code/data/postprocessed/interpro.csv",
                    "/home/TheGreatestCoder/code/data/postprocessed/uniprot.csv",
                    "/home/TheGreatestCoder/code/data/postprocessed/go.csv",
                    "/home/TheGreatestCoder/code/data/postprocessed/gene-literature.csv",
                ]
            ],
            help="Paths to one or more gene-side information CSV files (default: %(default)s).",
        )
        parser.add_argument(
            "--disease-side-info-paths",
            metavar="FILE",
            nargs="+",
            type=csv_file,
            default=[
                csv_file(file)
                for file in [
                    "/home/TheGreatestCoder/code/data/postprocessed/phenotype.csv"
                ]
            ],
            help="Paths to one or more disease-side information CSV files (default: %(default)s).",
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


def parse_baseline(subparsers: argparse.ArgumentParser):
    """
    Parses command-line arguments for the Baseline model.

    Args:
        parser (argparse.ArgumentParser): The parser for genehound.
    """
    parser = subparsers.add_parser(
        "baseline",
        help="Run Baseline on OMIM dataset in the cross-validation setting.",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        required=True,
        help="Number of folds for cross-validation.",
    )
    parser.add_argument(
        "--output-path",
        metavar="FILE",
        type=output_dir,
        required=True,
        help="Directory to save output results.",
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
        "--gene-disease-path",
        metavar="FILE",
        type=csv_file,
        default=csv_file(
            "/home/TheGreatestCoder/code/data/postprocessed/gene-disease.csv"
        ),
        help="Path of the gene disease matrix (default: %(default)s).",
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
    parser.add_argument(
        "--log-filename",
        type=str,
        default="pipeline.log",
        help="Filename for log output (default: %(default)s).",
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


def parse_post(subparsers: argparse._SubParsersAction):
    """
    Adds the 'post' subcommand to the parser for post-processing evaluation results.

    This subcommand loads serialized Evaluation objects, a YAML configuration file
    containing alpha values, and then produces ROC curves, AUC loss tables, and
    BEDROC plots and tables. It ensures that the number of evaluation paths matches
    the number of provided model names.

    Args:
        subparsers (argparse._SubParsersAction): The argparse subparsers object to
            which the 'post' command will be added.
    """
    parser = subparsers.add_parser(
        "post",
        help="Perform post-processing of evaluation results.",
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
