"""Profiling for smc"""
import cProfile, pstats
import yaml
from pathlib import Path
import logging
from genepriority.scripts.utils import pre_processing
from genepriority.scripts.nega import train_eval


def main():
    input_path = Path(input_path).absolute()
    omim_meta_path = Path(omim_meta_path).absolute()
    output_path = Path(output_path).absolute()
    logger = logging.getLogger("NEGA")

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if not omim_meta_path.exists():
        raise FileNotFoundError(f"OMIM metadata path does not exist: {omim_meta_path}")

    dataloader, _ = pre_processing(
        input_path=input_path,
        seed=42,
        omim_meta_path=omim_meta_path,
        side_info=False,
        num_splits=1,
        zero_sampling_factor=5,
        num_folds=None,
        train_size=.8,
        validation_size=.1,
    )

    config_path = Path(config_path).absolute()
    if not config_path.exists():
        raise FileNotFoundError(
            f"The configuration path does not exist: {config_path}"
        )

    logger.debug("Loading configuration file: %s", config_path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    regularization_parameter = config.get("regularization_parameter")
    symmetry_parameter = config.get("symmetry_parameter")
    smoothness_parameter = config.get("smoothness_parameter")
    rho_increase = config.get("rho_increase")
    rho_decrease = config.get("rho_decrease")
    tensorboard_dir = Path(tensorboard_dir).absolute()

    train_eval(
        logger=logger,
        output_path=output_path,
        dataloader=dataloader,
        rank=40,
        iterations=200,
        threshold=10,
        seed=42,
        regularization_parameter=regularization_parameter,
        symmetry_parameter=symmetry_parameter,
        smoothness_parameter=smoothness_parameter,
        rho_increase=rho_increase,
        rho_decrease=rho_decrease,
        tensorboard_dir=tensorboard_dir,
        results_filename="results.pickle",
    )

cProfile.run('main()', 'profile_stats')
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumtime').print_stats(10)