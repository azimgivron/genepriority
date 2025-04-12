"""
Parser Module
=============
"""
import argparse


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
        type=str,
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
        "--input-path",
        type=str,
        default="/home/TheGreatestCoder/code/data/postprocessed/",
        help="Directory containing input data files (default: %(default)s).",
    )
    parser.add_argument(
        "--omim-meta-path",
        type=str,
        default="/home/TheGreatestCoder/code/genepriority/configurations/omim.yaml",
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
        type=str,
        default=(
            "/home/TheGreatestCoder/code/genepriority/"
            "configurations/genehound/meta.yaml"
        ),
        help="Path to the YAML configuration file for GeneHound (default: %(default)s).",
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=str,
        default="/home/TheGreatestCoder/code/logs",
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
    cv_parser.add_argument(
        "--num-folds",
        type=int,
        required=True,
        help="Number of folds.",
    )

    # Add common arguments to both "cv" and "fine-tune"
    for parser in [fine_tune_parser, cv_parser]:
        parser.add_argument(
            "--output-path",
            type=str,
            required=True,
            help="Directory to save output result.",
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
            "--input-path",
            type=str,
            default="/home/TheGreatestCoder/code/data/postprocessed/",
            help=(
                "Path to the input data directory containing 'gene-disease.csv'"
                " (default: %(default)s)."
            ),
        )
        parser.add_argument(
            "--omim-meta-path",
            type=str,
            default="/home/TheGreatestCoder/code/genepriority/configurations/omim.yaml",
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

    # Additional arguments specific to the "cv" subcommand
    cv_parser.add_argument(
        "--side-info",
        action="store_true",
        help="Include side information for genes and diseases.",
    )
    cv_parser.add_argument(
        "--tensorboard-dir",
        type=str,
        default="/home/TheGreatestCoder/code/logs",
        help="Directory for TensorBoard logs (default: %(default)s).",
    )
    cv_parser.add_argument(
        "--config-path",
        type=str,
        default="/home/TheGreatestCoder/code/genepriority/configurations/nega/meta.yaml",
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
        type=str,
        required=True,
        help="Directory where output results will be saved.",
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
        type=str,
        default="/home/TheGreatestCoder/code/genepriority/configurations/post.yaml",
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
        "--apply-mask",
        action="store_false",
        help=(
            "If flagged, assessment is made on whole completed matrix instead of the"
            " test set only (default: %(default)s)."
        ),
    )
