"""
Parser Module
=============
"""
import argparse


def parse_genehound(subparsers: argparse._SubParsersAction) -> None:
    """
    Parses command-line arguments for the GeneHound reproduction pipeline.

    This function adds two subparsers:
      - "genehound-omim1": Run GeneHound using the OMIM1 dataset with multiple splits.
      - "genehound-omim2": Run GeneHound using the filtered OMIM2 dataset with cross-validation.

    Args:
        subparsers (argparse._SubParsersAction): The subparsers object to which the genehound
            commands will be added.
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
            help="Directory to save output results (default: %(default)s).",
        )
        subparser.add_argument(
            "--zero-sampling-factor",
            type=int,
            required=True,
            help=(
                "Factor for zero sampling (number of zeros = factor * "
                "number of ones) (default: %(default)s). "
            ),
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
            help=(
                "Proportion of data for validation (unused for comparison with NEGA)"
                " (default: %(default)s)."
            ),
        )
        subparser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed for reproducibility (default: %(default)s).",
        )


def parse_nega(subparsers: argparse._SubParsersAction) -> None:
    """
    Adds subcommands for NEGA to the argument parser.

    Two subcommands are provided:
      - "nega-cv": Run cross-validation (for hyperparameter tuning).
      - "nega": Run a single train-evaluation cycle.

    Args:
        subparsers (argparse._SubParsersAction): The subparsers object to which
            the NEGA commands will be added.
    """
    cv_parser = subparsers.add_parser(
        "nega-cv", help="Perform cross-validation for hyperparameter tuning of NEGA."
    )
    eval_parser = subparsers.add_parser(
        "nega", help="Train and evaluate the NEGA model."
    )

    for subparser in [cv_parser, eval_parser]:
        subparser.add_argument(
            "--output-path",
            type=str,
            required=True,
            help="Directory to save output result (default: %(default)s).",
        )
        subparser.add_argument(
            "--num-splits",
            type=int,
            required=True,
            help="Number of data splits (default: %(default)s).",
        )
        subparser.add_argument(
            "--zero-sampling-factor",
            type=int,
            required=True,
            help=(
                "Factor for zero sampling (number of zeros = factor * "
                "number of ones) (default: %(default)s). "
            ),
        )
        subparser.add_argument(
            "--input-path",
            type=str,
            default="/home/TheGreatestCoder/code/data/postprocessed/",
            help=(
                "Path to the input data directory containing 'gene-disease.csv'"
                " (default: %(default)s)."
            ),
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
            "--rank",
            type=int,
            default=40,
            help="Rank (number of latent factors) for the model (default: %(default)s).",
        )
        subparser.add_argument(
            "--iterations",
            type=int,
            default=200,
            help="Number of training iterations (default: %(default)s).",
        )
        subparser.add_argument(
            "--threshold",
            type=int,
            default=10,
            help="Threshold parameter for the model (default: %(default)s).",
        )
        subparser.add_argument(
            "--validation-size",
            type=float,
            default=0.1,
            help="Proportion of data to use for validation (default: %(default)s).",
        )
        subparser.add_argument(
            "--train-size",
            type=float,
            default=0.8,
            help="Proportion of data to use for training (default: %(default)s).",
        )
        subparser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed for reproducibility (default: %(default)s).",
        )

    eval_parser.add_argument(
        "--tensorboard-dir",
        type=str,
        default="/home/TheGreatestCoder/code/logs",
        help="Directory for TensorBoard logs (default: %(default)s).",
    )
    eval_parser.add_argument(
        "--config-path",
        type=str,
        default="/home/TheGreatestCoder/code/genepriority/configurations/nega/meta.yaml",
        help=(
            "Path to the YAML configuration file containing simulation parameters "
            "(default: %(default)s)."
        ),
    )
    eval_parser.add_argument(
        "--results-filename",
        type=str,
        default="results.pickle",
        help="Filename for serialized results (default: %(default)s).",
    )
    cv_parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of trials for hyperparameter tuning (default: %(default)s).",
    )


def parse_post(subparsers: argparse._SubParsersAction) -> None:
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
        help="One or more paths to serialized `Evaluation` objects.",
    )
    parser.add_argument(
        "--model-names",
        type=str,
        nargs="+",
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
        "--shared-y",
        action="store_false",
        help="Whether to share the y axis for BEDROC boxplots (default: %(default)s).",
    )
