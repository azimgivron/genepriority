# pylint: disable=R0914, R0915, R0801, R0913
"""
Reproduce GeneHound results using a MACAU-based approach.

This script performs the following steps:
1. Loads gene–disease association data.
2. Loads side information for genes and diseases.
3. Trains multiple MACAU models using different latent dimensions.
4. Evaluates the models and produces:
   - ROC curves
   - AUC/loss tables
   - BEDROC scores and boxplots
"""

import argparse
import logging
import pickle
import traceback
from pathlib import Path
from typing import List, Tuple

import yaml

from NEGradient_GenePriority import (
    DataLoader,
    Evaluation,
    MACAUTrainer,
    ModelEvaluationCollection,
    SideInformationLoader,
    generate_auc_loss_table,
    generate_bedroc_table,
    plot_bedroc_boxplots,
    plot_roc_curves,
)
from NEGradient_GenePriority.scripts.utils import load_omim_meta

CONFIG_KEYS = [
    "num_splits",
    "num_folds",
    "zero_sampling_factor",
    "train_size",
]


def post_processing(
    omim1_results: Path,
    omim2_results: Path,
    output_path: Path,
    latent_dimensions: List[int],
    post_config_path: Path,
) -> None:
    """
    Performs post-processing by generating ROC curves, AUC/Loss tables, and BEDROC scores.

    This function:
      - Loads the post-processing configuration.
      - Sets alpha values in the Evaluation class.
      - Creates ROC curve plots for two sets of evaluations.
      - Generates an AUC/Loss CSV table for the first evaluation collection.
      - Creates BEDROC boxplots and a CSV table for the second evaluation collection.

    Args:
        omim1_results (Path): Path to the results for OMIM1.
        omim2_results (Path): Path to the results for OMIM2.
        output_path (Path): Directory where output files will be saved.
        latent_dimensions (List[int]): List of latent dimensions used (for labeling purposes).
        post_config_path (Path): Path to the YAML configuration file for post-processing.
    """
    logger = logging.getLogger("post_processing")
    logger.debug("Loading configuration file: %s", post_config_path)
    with post_config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if "alphas" not in config:
        raise KeyError("Alpha values not found in post-processing configuration.")

    alpha_map = config["alpha"]
    logger.debug("Starting figures and tables creation.")
    Evaluation.alphas = list(alpha_map.keys())
    Evaluation.alpha_map = alpha_map

    logger.debug("Loading OMIM1 results.")
    with omim1_results.open("rb") as stream:
        omim1_results = pickle.load(stream)

    logger.debug("Loading OMIM2 results.")
    with omim2_results.open("rb") as stream:
        omim2_results = pickle.load(stream)

    collections = [
        ModelEvaluationCollection(omim1_results),
        ModelEvaluationCollection(omim2_results),
    ]

    # Plot ROC curves for both evaluation collections.
    for i, collection in enumerate(collections, start=1):
        plot_roc_curves(
            evaluation_collection=collection,
            output_file=output_path / f"roc_curve_omim{i}",
        )

    # Generate and save AUC/Loss table for the first collection.
    auc_loss_csv_path = output_path / "auc_loss_omim1.csv"
    generate_auc_loss_table(
        collections[0].compute_auc_losses(),
        model_names=collections[0].model_names,
    ).to_csv(auc_loss_csv_path)
    logger.info("AUC/Loss table saved: %s", auc_loss_csv_path)

    # Plot BEDROC boxplots for the second collection.
    bedroc_plot_path = output_path / "bedroc_omim2.png"
    plot_bedroc_boxplots(
        collections[1].compute_bedroc_scores(),
        model_names=latent_dimensions,
        output_file=bedroc_plot_path,
    )
    logger.info("BEDROC boxplots saved: %s", bedroc_plot_path)

    # Generate and save BEDROC table for the second collection.
    bedroc_csv_path = output_path / "bedroc_omim2.csv"
    generate_bedroc_table(
        collections[1].compute_bedroc_scores(),
        model_names=collections[1].model_names,
        alpha_map=Evaluation.alpha_map,
    ).to_csv(bedroc_csv_path)
    logger.info("BEDROC table saved: %s", bedroc_csv_path)

    logger.debug("Figures and tables creation completed successfully")


def pre_processing(
    input_path: Path,
    config_path: Path,
    seed: int,
    omim_meta_path: Path,
    side_info: bool,
) -> Tuple[DataLoader, SideInformationLoader]:
    """
    Loads configuration parameters, gene–disease association data, and side information.

    This function:
      - Loads a YAML configuration file and verifies required keys.
      - Initializes a DataLoader to load gene–disease data.
      - Initializes a SideInformationLoader to process side information files.

    Args:
        input_path (Path): Directory containing the input CSV files.
        config_path (Path): Path to the YAML configuration file.
        seed (int): Seed for reproducibility in sampling.
        omim_meta_path (Path): Path to OMIM meta data.
        side_info (bool): Whether to load side information.

    Returns:
        Tuple[DataLoader, SideInformationLoader]: The data loader and side information
            loader objects.
    """
    logger = logging.getLogger("pre_processing")
    logger.debug("Loading configuration file: %s", config_path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    for key in CONFIG_KEYS:
        if key not in config:
            raise KeyError(
                f"{key} not found in configuration file. "
                "Make sure it is set before running the script again."
            )

    nb_genes, nb_diseases, min_associations = load_omim_meta(omim_meta_path)
    num_splits = config["num_splits"]
    zero_sampling_factor = config["zero_sampling_factor"]
    num_folds = config["num_folds"]
    train_size = config["train_size"]

    # Load gene–disease data.
    dataloader = DataLoader(
        nb_genes=nb_genes,
        nb_diseases=nb_diseases,
        path=input_path / "gene-disease.csv",
        seed=seed,
        num_splits=num_splits,
        zero_sampling_factor=zero_sampling_factor,
        num_folds=num_folds,
        train_size=train_size,
        min_associations=min_associations,
    )
    dataloader(filter_column="Disease ID")

    # Load side information.
    if side_info:
        side_info_loader = SideInformationLoader(
            nb_genes=nb_genes, nb_diseases=nb_diseases
        )
        side_info_loader.process_side_info(
            gene_side_info_paths=[
                input_path / "interpro.csv",
                input_path / "uniprot.csv",
                input_path / "go.csv",
            ],
            disease_side_info_paths=[input_path / "phenotype.csv"],
            names=["interpro", "uniprot", "GO", "phenotype"],
        )
    else:
        side_info_loader = None
    return dataloader, side_info_loader


def run(
    dataloader: DataLoader,
    side_info_loader: SideInformationLoader,
    output_path: Path,
    tensorboard_dir: Path,
    seed: int,
    latent_dimensions: List[int],
    omim1_filename: str,
    omim2_filename: str,
) -> None:
    """
    Configures and runs the MACAU training session.

    This function initializes a MACAUTrainer with the provided parameters and runs
    the training process. It saves the results using the provided filenames.

    Args:
        dataloader (DataLoader): The data loader object containing gene–disease data.
        side_info_loader (SideInformationLoader): The side information loader object.
        output_path (Path): Directory to save training outputs.
        tensorboard_dir (Path): Directory for TensorBoard logs.
        seed (int): Seed for reproducibility.
        latent_dimensions (List[int]): List of latent dimensions for model training.
        omim1_filename (str): Filename for saving OMIM1 results.
        omim2_filename (str): Filename for saving OMIM2 results.
    """
    logger = logging.getLogger("run")
    logger.debug("Configuring MACAU session")
    trainer = MACAUTrainer(
        dataloader=dataloader,
        side_info_loader=side_info_loader,
        path=output_path,
        num_samples=3_500,
        burnin_period=500,
        direct=False,
        univariate=True,
        seed=seed,
        save_freq=100,
        verbose=0,
        tensorboard_dir=tensorboard_dir,
    )
    _ = trainer(
        latent_dimensions=latent_dimensions,
        save_results=True,
        omim1_filename=omim1_filename,
        omim2_filename=omim2_filename,
    )


def setup_logger(log_file: Path) -> None:
    """
    Configures the root logger to write to a log file and the console.

    The logger is set to DEBUG level and uses a specific format.

    Args:
        log_file (Path): Path to the log file where logs will be written.
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - [%(funcName)s] - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
        ],
    )


def parse() -> argparse.Namespace:
    """
    Parses and retrieves command-line arguments for the GeneHound reproduction pipeline.

    The supported arguments include paths for input data, configuration files,
    output directories, TensorBoard logs, a random seed for reproducibility,
    flags to control the execution of training and post-processing, filenames for
    result storage, and latent dimensions for the MACAU models.

    Returns:
        argparse.Namespace: Namespace containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Reproduce GeneHound results using a MACAU-based approach."
    )
    parser.add_argument(
        "--run",
        type=bool,
        required=True,
        action=argparse.BooleanOptionalAction,
        help=(
            "Flag indicating whether to execute the training simulation. "
            "If set, the script will run the MACAU model training using the "
            "provided data and configuration. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--post",
        type=bool,
        required=True,
        action=argparse.BooleanOptionalAction,
        help=(
            "Flag indicating whether to perform post-processing on the simulation results. "
            "If enabled, the script will generate evaluation plots and tables such as ROC "
            "curves, AUC/loss tables, and BEDROC scores. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--side-info",
        type=bool,
        required=True,
        action=argparse.BooleanOptionalAction,
        help=(
            "Flag indicating whether to add side information for the simulation. "
            "If enabled, the script will add both row and column side information "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="/home/TheGreatestCoder/code/data/postprocessed/",
        help=(
            "Path to the directory containing input data files required for the simulation. "
            "This directory must include the 'gene-disease.csv' file with gene–disease "
            "associations and may include additional files for side information. "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--omim-meta-path",
        type=str,
        default=(
            "/home/TheGreatestCoder/code/NEGradient-GenePriority"
            "/configurations/omim.yaml"
        ),
        help=(
            "Path to the OMIM file which contains the meta data about "
            "the OMIM association matrix. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=(
            "/home/TheGreatestCoder/code/NEGradient-GenePriority"
            "/configurations/genehound/meta.yaml"
        ),
        help=(
            "Path to the YAML configuration file that contains parameters for data processing and "
            "simulation. The file should define keys such as 'num_splits', 'num_folds', "
            "'nb_genes', etc. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--post-config-path",
        type=str,
        default=(
            "/home/TheGreatestCoder/code/NEGradient-GenePriority"
            "/configurations/genehound/post.yaml"
        ),
        help=(
            "Path to the YAML configuration file for post-processing. "
            "This file should contain settings like the alpha values used to compute "
            "evaluation metrics (e.g., BEDROC scores) during post-processing. "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/home/TheGreatestCoder/code/genehounds/",
        help=(
            "Path to the directory where output results will be saved. "
            "This includes training logs, plots (e.g., ROC curves and BEDROC boxplots), "
            "and CSV tables with evaluation metrics. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--tensorboard-base-log-dir",
        type=str,
        default="/home/TheGreatestCoder/code/logs",
        help=(
            "Path to the base directory for TensorBoard logs. Training progress and other "
            "metrics will be logged here for visualization using TensorBoard. "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed used for reproducibility of data splits and sampling. "
            "Setting this seed ensures that the simulation results remain consistent "
            "between runs. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--omim1-filename",
        type=str,
        default="omim1_results.pickle",
        help=(
            "Filename for saving the results corresponding to the first dataset (OMIM1). "
            "The file will be stored in the specified output directory. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--omim2-filename",
        type=str,
        default="omim2_results.pickle",
        help=(
            "Filename for saving the results corresponding to the second dataset (OMIM2). "
            "The file will be stored in the specified output directory. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--latent-dimensions",
        type=int,
        nargs="+",
        default=[25, 30, 40],
        help=(
            "Space-separated list of latent dimensions to be used for training the MACAU models. "
            "For example, '--latent_dimensions 25 30 40' will run three models with latent "
            "dimensions 25, 30, and 40, respectively. "
            "(default: %(default)s)"
        ),
    )

    args = parser.parse_args()
    return args


def main() -> None:
    """
    Main entry point for the GeneHound reproduction pipeline.

    This function performs the following steps:
      1. Parses command-line arguments.
      2. Sets up output directories and logging.
      3. Executes pre-processing to load data and side information.
      4. If requested, runs the MACAU training simulation.
      5. If requested, performs post-processing to generate evaluation figures and tables.

    Exceptions are caught and logged before being re-raised.
    """
    args = parse()

    output_path = Path(args.output_path).absolute()
    output_path.mkdir(parents=True, exist_ok=True)

    latent_dimensions = args.latent_dimensions

    # Create a logger for the pipeline (used in both run and post-processing).
    logger = logging.getLogger("GeneHound")

    try:
        if args.run:
            tensorboard_dir: Path = Path(args.tensorboard_base_log_dir).absolute()
            tensorboard_dir.mkdir(parents=True, exist_ok=True)

            # Setup logger
            log_file: Path = output_path / "pipeline.log"
            setup_logger(log_file)

            seed = args.seed

            input_path = Path(args.input_path).absolute()
            if not input_path.exists():
                raise FileNotFoundError(f"The input path does not exist: {input_path}")

            config_path = Path(args.config_path).absolute()
            if not config_path.exists():
                raise FileNotFoundError(
                    f"The configuration path does not exist: {config_path}"
                )

            omim_meta_path = Path(args.omim_meta_path).absolute()
            if not omim_meta_path.exists():
                logger.error("OMIM meta data path does not exist: %s", omim_meta_path)
                raise FileNotFoundError(
                    f"OMIM meta data path does not exist: {omim_meta_path}"
                )

            dataloader, side_info_loader = pre_processing(
                input_path=input_path,
                config_path=config_path,
                seed=seed,
                omim_meta_path=omim_meta_path,
                side_info=args.side_info,
            )
            run(
                dataloader=dataloader,
                side_info_loader=side_info_loader,
                output_path=output_path,
                tensorboard_dir=tensorboard_dir,
                seed=seed,
                latent_dimensions=latent_dimensions,
                omim1_filename=args.omim1_filename,
                omim2_filename=args.omim2_filename,
            )

        if args.post:
            post_config_path = Path(args.post_config_path).absolute()
            if not post_config_path.exists():
                raise FileNotFoundError(
                    f"The post-processing configuration path does not exist: {post_config_path}"
                )

            post_processing(
                omim1_results=output_path / args.omim1_filename,
                omim2_results=output_path / args.omim2_filename,
                output_path=output_path,
                latent_dimensions=latent_dimensions,
                post_config_path=post_config_path,
            )
    except Exception as exception:
        logger.error("An error occurred during processing: %s", exception)
        logger.error("%s", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
