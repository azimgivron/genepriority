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
import numpy as np
import yaml
import copy

from genepriority.evaluation import Evaluation
from genepriority.postprocessing.dataframes import (generate_table,
                                                    generate_bedroc_table)
from genepriority.postprocessing.figures import (plot_auc_boxplots,
                                                 plot_avg_precision_boxplots,
                                                 plot_bedroc_boxplots,
                                                 plot_cdf_curves,
                                                 plot_pr_curves,
                                                 plot_roc_curves,
                                                 plot_bedroc_curves)
from genepriority.postprocessing.model_evaluation_collection import \
    ModelEvaluationCollection


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
            res = pickle.load(stream).results
            results_data[name] = Evaluation(res)
    base_res = []
    for r in res:
        baseline = copy.deepcopy(r)
        baseline._y_pred = np.zeros_like(baseline._y_pred)
        base_res.append(baseline)
    results_data["Baseline"] = Evaluation(base_res)

    results = ModelEvaluationCollection(results_data, over=args.over)

    auc_csv_path = output_path / "auroc.csv"
    auc = results.compute_avg_auc()
    generate_table(
        auc,
        model_names=results.model_names,
    ).to_csv(auc_csv_path)
    logger.info("AUC table saved: %s", auc_csv_path)

    auc_plot_path = output_path / "auroc.png"
    plot_auc_boxplots(
        auc,
        model_names=results.model_names,
        output_file=auc_plot_path,
        figsize=(12, 10),
    )
    logger.info("AUC boxplots saved: %s", auc_plot_path)

    average_pr = results.compute_avg_precision()
    avg_pr_plot_path = output_path / "auprc.png"
    plot_avg_precision_boxplots(
        average_pr,
        model_names=results.model_names,
        output_file=avg_pr_plot_path,
        figsize=(12, 10),
    )
    logger.info("Average PR boxplots saved: %s", auc_plot_path)

    average_pr_csv_path = output_path / "average_pr.csv"
    generate_table(
        average_pr,
        model_names=results.model_names,
    ).to_csv(average_pr_csv_path)
    logger.info("AUPRC table saved: %s", average_pr_csv_path)

    roc = results.compute_avg_roc()
    roc_plot_path = output_path / "roc.png"
    plot_roc_curves(
        roc,
        model_names=results.model_names,
        output_file=roc_plot_path,
        figsize=(12, 10),
    )
    logger.info("ROC curves saved: %s", roc_plot_path)

    pr = results.compute_avg_pr()
    pr_plot_path = output_path / "pr.png"
    plot_pr_curves(
        pr,
        model_names=results.model_names,
        output_file=pr_plot_path,
        figsize=(12, 10),
    )
    logger.info("PR curves saved: %s", pr_plot_path)

    cdf = results.compute_avg_cdf()
    cdf_plot_path = output_path / "cdf.png"
    plot_cdf_curves(
        cdf,
        model_names=results.model_names,
        output_file=cdf_plot_path,
        figsize=(12, 10),
    )
    logger.info("CDF curves saved: %s", cdf_plot_path)

    bedroc_plot_path = output_path / "bedroc.png"
    bed = results.compute_avg_bedroc_scores()
    plot_bedroc_boxplots(
        bed,
        model_names=results.model_names,
        output_file=bedroc_plot_path,
        figsize=(12, 10),
        sharey=args.no_sharey,
    )
    logger.info("BEDROC boxplots saved: %s", bedroc_plot_path)

    bedroc_csv_path = output_path / "bedroc.csv"
    generate_bedroc_table(
        bed,
        model_names=results.model_names,
        alpha_map=Evaluation.alpha_map,
    ).to_csv(bedroc_csv_path)
    logger.info("BEDROC table saved: %s", bedroc_csv_path)

    nb_genes = results.evaluations[0].results[0]._y_true.shape[0]
    mapping = lambda alpha: -1/alpha * np.log(1-.99+.99*np.exp(-alpha)) * nb_genes
    Evaluation.alphas = np.linspace(90, 400, 50)
    Evaluation.alpha_map = [mapping(alpha) for alpha in Evaluation.alphas]
    bed = results.compute_avg_bedroc_scores()
    bed = bed.mean(axis=1).T

    bed_plot_path = output_path / "bedroc-curve.png"
    plot_bedroc_curves(
        bed=bed,
        alpha_map=Evaluation.alpha_map,
        model_names=results.model_names,
        output_file=bed_plot_path,
        figsize=(12, 10),
    )
    logger.info("BEDROC curves saved: %s", bed_plot_path)
    logger.debug("Figures and tables creation completed successfully")
