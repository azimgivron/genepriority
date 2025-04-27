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

import yaml

from genepriority.evaluation.evaluation import Evaluation
from genepriority.postprocessing.dataframes import (
    generate_auc_loss_table,
    generate_bedroc_table,
)
from genepriority.postprocessing.figures import (
    plot_auc_boxplots,
    plot_bedroc_boxplots,
)
from genepriority.postprocessing.model_evaluation_collection import (
    ModelEvaluationCollection,
)


def post(args: argparse.Namespace):
    """
    Processes evaluation results and generates performance metrics and plots.

    This function performs the following steps:
      1. Loads a YAML configuration file that provides post-processing parameters
      (e.g., alpha values).
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
    # pylint: disable=R0914
    post_config_path = Path(args.post_config_path)
    output_path = Path(args.output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("post_processing")
    logger.debug("Loading configuration file: %s", post_config_path)

    # Load the post-processing configuration (especially alpha values)
    with post_config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if "alphas" not in config:
        raise KeyError("Alpha values not found in post-processing configuration.")

    alpha_map = config["alphas"]
    Evaluation.alphas = list(alpha_map.keys())
    Evaluation.alpha_map = alpha_map

    results_data = {}
    for name, path_str in zip(args.model_names, args.evaluation_paths):
        with Path(path_str).open("rb") as stream:
            results_data[name] = pickle.load(stream)

    results = ModelEvaluationCollection(results_data)

    auc_loss_csv_path = output_path / "auc_loss.csv"
    auc_losses = results.compute_auc_losses()
    generate_auc_loss_table(
        auc_losses,
        model_names=results.model_names,
    ).to_csv(auc_loss_csv_path)
    logger.info("AUC Loss table saved: %s", auc_loss_csv_path)

    auc_loss_plot_path = output_path / "auc.png"
    plot_auc_boxplots(
        -auc_losses + 1,
        model_names=results.model_names,
        output_file=auc_loss_plot_path,
        figsize=(12, 10),
    )
    logger.info("AUC Loss boxplots saved: %s", auc_loss_plot_path)

    bedroc_plot_path = output_path / "bedroc.png"
    plot_bedroc_boxplots(
        results.compute_bedroc_scores(),
        model_names=results.model_names,
        output_file=bedroc_plot_path,
        figsize=(24, 10),
        sharey=args.no_sharey,
    )
    logger.info("BEDROC boxplots saved: %s", bedroc_plot_path)

    bedroc_csv_path = output_path / "bedroc.csv"
    generate_bedroc_table(
        results.compute_bedroc_scores(),
        model_names=results.model_names,
        alpha_map=Evaluation.alpha_map,
    ).to_csv(bedroc_csv_path)
    logger.info("BEDROC table saved: %s", bedroc_csv_path)

    logger.debug("Figures and tables creation completed successfully")
