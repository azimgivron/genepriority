import logging
from pathlib import Path
import pandas as pd
import mypackage.main as main
from preprocessing import (
    disease_count,
    from_omim1_to_splits,
    from_omim2_to_folds,
    gene_disease_to_omim2_df,
    omim_as_coo,
    sample_zeros,
)


def setup_logger(log_file: str) -> logging.Logger:
    """
    Configures and returns a logger that writes to both the console and a file.

    Args:
        log_file (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger("main_logger")
    logger.setLevel(logging.DEBUG)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def main():
    # Setup paths and logger
    log_file = "pipeline.log"
    logger = setup_logger(log_file)
    input_path = Path("/home/TheGreatestCoder/code/data/postprocessed/")

    if not input_path.exists():
        logger.error("The input path does not exist: %s", input_path)
        raise FileNotFoundError(f"The input path does not exist: {input_path}")

    try:
        # Load data
        logger.info("Loading gene-disease data from %s", input_path)
        gene_disease = pd.read_csv(input_path / "gene-disease.csv")

        # Convert gene-disease DataFrame to sparse matrix
        omim = omim_as_coo(gene_disease)

        # Calculate sparsity
        sparsity = omim.count_nonzero() / (omim.shape[0] * omim.shape[1])
        df_sparsity = pd.DataFrame([[sparsity * 100]], columns=["sparsity [%]"])
        logger.info("Sparsity of the data:\n%s\n", df_sparsity.to_markdown())

        # Set parameters
        alphas = [228.5, 160.9, 32.2, 16.1, 5.3]
        latent_dims = [25, 30, 40]
        nb_split = 6
        factor = 5

        # Generate splits
        logger.info("Sampling zeros with factor %d", factor)
        omim1 = sample_zeros(omim, factor)
        splits = from_omim1_to_splits(omim1, n_splits=nb_split)

        # Disease count statistics
        counts = disease_count(splits)
        logger.info("Disease count statistics:\n%s\n", counts)

        # Filter and sample for OMIM2
        logger.info("Processing OMIM2 data")
        omim2_df = gene_disease_to_omim2_df(gene_disease, threshold=10)
        omim2 = sample_zeros(omim, factor)
        from_omim2_to_folds(omim2, n_folds=5)

        # Configure artificial dataset
        logger.info("Configuring artificial dataset and BPMF session")
        nsamples = 3500
        burnin = 500

        session = main.BPMFSession(
            Ytrain=Ytrain,
            Ytest=Ytest,
            is_scarce=False,
            direct=True,
            univariate=True,
            num_latent=num_latent,
            burnin=burnin,
            nsamples=nsamples,
        )
        logger.info("BPMF session initialized successfully")

    except Exception as e:
        logger.exception("An error occurred during processing")
        raise


if __name__ == "__main__":
    main()
