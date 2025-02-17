"""
This script loads and processes evaluation results for multiple models, then
generates and saves various performance metrics (AUC loss, BEDROC) and plots
(ROC curves, BEDROC boxplots).

It relies on:
    - A post-processing configuration YAML (post.yaml) that contains alpha values.
    - Serialized (pickle) evaluation results for different evaluation scenarios.
"""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Any, Dict

import yaml

from genepriority import (
    Evaluation,
    ModelEvaluationCollection,
    generate_auc_loss_table,
    generate_bedroc_table,
    plot_bedroc_boxplots,
    plot_roc_curves,
)


def parse_post(subparsers: Any) -> None:
    """
    Adds the 'post' subcommand to the parser for post-processing evaluation results.
    
    This subcommand loads serialized Evaluation objects, a YAML configuration file
    containing alpha values, and then produces ROC curves, AUC loss tables, and
    BEDROC plots and tables. It ensures that the number of evaluation paths matches
    the number of provided model names.

    Args:
        subparsers: The argparse subparsers object to which the 'post' command will be added.
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
        nargs='+',
        help="One or more paths to serialized `Evaluation` objects."
    )
    parser.add_argument(
        "--model-names",
        type=str,
        nargs='+',
        help="One or more model names corresponding to the evaluation paths (in the same order)."
    )
    parser.add_argument(
        "--post-config-path",
        type=str,
        default="/home/TheGreatestCoder/code/genepriority/configurations/post.yaml",
        help="Path to the post-processing configuration file containing alpha values. (default: %(default)s)",
    )


def post(args: argparse.Namespace) -> None:
    """
    Processes evaluation results and generates performance metrics and plots.

    This function performs the following steps:
      1. Loads a YAML configuration file that provides post-processing parameters (e.g., alpha values).
      2. Loads multiple serialized Evaluation objects from the specified file paths.
      3. Constructs a ModelEvaluationCollection with the loaded data.
      4. Generates and saves ROC curves, an AUC loss table, and BEDROC plots and tables
         in the specified output directory.

    Args:
        args: An argparse.Namespace object containing:
            - evaluation_paths (List[str]): Paths to serialized Evaluation objects.
            - model_names (List[str]): Model names corresponding to the evaluation paths.
            - post_config_path (str): Path to the YAML configuration file.
            - output_path (str): Directory where output files will be saved.
    """
    post_config_path: Path = Path(args.post_config_path)
    output_path: Path = Path(args.output_path)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("post_processing")
    logger.debug("Loading configuration file: %s", post_config_path)

    # Load the post-processing configuration (especially alpha values)
    with post_config_path.open("r", encoding="utf-8") as stream:
        config: Dict[str, Any] = yaml.safe_load(stream)

    if "alphas" not in config:
        raise KeyError("Alpha values not found in post-processing configuration.")

    alpha_map: Dict[str, Any] = config["alphas"]
    Evaluation.alphas = list(alpha_map.keys())
    Evaluation.alpha_map = alpha_map

    # Load evaluation results from pickled objects
    results_data: Dict[str, Any] = {}
    for name, path_str in zip(args.model_names, args.evaluation_paths):
        with Path(path_str).open("rb") as stream:
            results_data[name] = pickle.load(stream)

    # Create a ModelEvaluationCollection to hold all results
    results = ModelEvaluationCollection(results_data)

    # Plot and save the ROC curves
    plot_roc_curves(
        evaluation_collection=results,
        output_file=output_path / "roc_curve",
        figsize=(10, 8),
    )

    auc_loss_csv_path: Path = output_path / "auc_loss.csv"
    generate_auc_loss_table(
        results.compute_auc_losses(),
        model_names=args.model_names,
    ).to_csv(auc_loss_csv_path)
    logger.info("AUC/Loss table saved: %s", auc_loss_csv_path)

    # Generate and save the BEDROC plots and tables
    bedroc_plot_path: Path = output_path / "bedroc.png"
    plot_bedroc_boxplots(
        results.compute_bedroc_scores(),
        model_names=args.model_names,
        output_file=bedroc_plot_path,
        figsize=(24, 10),
    )
    logger.info("BEDROC boxplots saved: %s", bedroc_plot_path)

    bedroc_csv_path: Path = output_path / "bedroc.csv"
    generate_bedroc_table(
        results.compute_bedroc_scores(),
        model_names=args.model_names,
        alpha_map=Evaluation.alpha_map,
    ).to_csv(bedroc_csv_path)
    logger.info("BEDROC table saved: %s", bedroc_csv_path)

    logger.debug("Figures and tables creation completed successfully")
