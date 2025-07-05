"""
Module for preprocessing gene, disease, ontology, and phenotype data.
Strings and file paths are centralized via dataclasses.
Provides functions to load, process, and save various biological datasets,
plus a CLI entry point.

Raw data is coming from:
https://drive.google.com/drive/folders/1_CcANqmLBt0PsrwzoBU2K5fdDzZTVRz0

For more information, reach out:
    * Pooya Zakeri : zakeri@ifi.uio.no
"""

import argparse
import logging
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

NB_GENES: int = 14_195
NB_DISEASES: int = 314


def load_csv(
    path: Path, header=None, names=None, usecols=None, dtype=None
) -> pd.DataFrame:
    """
    Load a CSV file into a Pandas DataFrame.

    Args:
        path (Path): Path to the CSV file.
        header (int or None): Row number to use as the column names.
        names (list[str] or None): Column names to assign.
        usecols (list[int] or None): Columns to read.
        dtype (dict[str, type] or None): Data types for columns.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    df = pd.read_csv(path, header=header, usecols=usecols)
    if names:
        df.columns = names
    if dtype:
        df = df.astype(dtype)
    return df


def load_mat(path: Path, key: str):
    """
    Load a MATLAB .mat file and extract a variable.

    Args:
        path (Path): Path to the .mat file.
        key (str): Variable name to extract.

    Returns:
        Any: The extracted MATLAB variable.
    """
    mat = loadmat(path)
    return mat[key]


@dataclass(frozen=True)
class FileConfig:
    """
    Configuration for raw and output file names.

    Attributes:
        gene_disease (str): Raw gene-disease CSV filename.
        gene_ids (str): Raw gene IDs CSV filename.
        gene_symbols (str): Raw gene symbols CSV filename.
        go (str): Raw GO associations CSV filename.
        interpro (str): Raw InterPro associations CSV filename.
        uniprot (str): Raw UniProt associations CSV filename.
        phen (str): Raw phenotype .mat filename.
        phen_terms (str): Raw phenotype-terms .mat filename.
        text (str): Raw text-feature CSV filename.
        out_gene_disease (str): Output gene-disease CSV filename.
        out_gene_symbols (str): Output gene-symbols CSV filename.
        out_go (str): Output GO CSV filename.
        out_interpro (str): Output interpro CSV filename.
        out_uniprot (str): Output uniprot CSV filename.
        out_phen (str): Output phenotype CSV filename.
        out_phen_terms (str): Output phenotype-terms CSV filename.
        out_gene_literature (str): Output gene–literature CSV filename.
        readme (str): README markdown filename.
    """

    gene_disease: str = "Gene-Disease-Matrix.csv"
    gene_ids: str = "geneIds_(only for text data).csv"
    gene_symbols: str = "geneSymbols.csv"
    go: str = "go.csv"
    interpro: str = "interpro.csv"
    uniprot: str = "uniprot.csv"
    phen: str = "Phen.mat"
    phen_terms: str = "Phen_terms.mat"
    text: str = "text.csv"

    out_gene_disease: str = "gene-disease.csv"
    out_gene_symbols: str = "gene-symbols.csv"
    out_go: str = "go.csv"
    out_interpro: str = "interpro.csv"
    out_uniprot: str = "uniprot.csv"
    out_phen: str = "phenotype.csv"
    out_phen_terms: str = "phenotype-terms.csv"
    out_gene_literature: str = "gene-literature.csv"
    readme: str = "README.md"


@dataclass
class ReadmeEntry:
    """
    Metadata for a single file in the README.

    Attributes:
        file (str): Filename.
        description (str): Description of the file contents.
        rows (int | None): Number of rows, or None if unknown.
        cols (int | None): Number of columns, or None if unknown.
    """

    file: str
    description: str
    rows: int | None
    cols: int | None


@dataclass
class ReadmeConfig:
    """
    Configuration for the README summary.

    Attributes:
        entries (list[ReadmeEntry]): List of file metadata entries.
        terms_description (str): Markdown description of key fields.
    """

    entries: list[ReadmeEntry] = field(
        default_factory=lambda: [
            ReadmeEntry(
                "geneSymbols.csv", "Maps gene symbols to identifiers", NB_GENES, 2
            ),
            ReadmeEntry(
                "geneIds_(only for text data).csv",
                "Gene IDs used in text data",
                NB_GENES,
                1,
            ),
            ReadmeEntry("go.csv", "Gene ontology associations", 1_365_394, 2),
            ReadmeEntry(
                "interpro.csv", "Protein domain associations (InterPro)", 51_500, 2
            ),
            ReadmeEntry("uniprot.csv", "UniProt protein associations", 164_126, 2),
            ReadmeEntry("gene-disease.csv", "Gene–disease associations", 2625, 2),
            ReadmeEntry("phenotype.csv", "Phenotypic association scores", 1_326_832, 3),
            ReadmeEntry("phenotype-terms.csv", "OMIM phenotype terms", NB_DISEASES, 2),
            ReadmeEntry(
                "Gene–literature feature associations",
                "Gene literature data",
                3_198_556,
                3,
            ),
        ]
    )
    terms_description: str = (
        "- **UniProt ID**: identifier for protein entries in UniProt, linking gene to protein information.\n"
        "- **InterPro domain ID**: conserved protein domain signatures from InterPro.\n"
        "- **GO term ID**:\n"
        "  - Biological Process (BP)\n"
        "  - Molecular Function (MF)\n"
        "  - Cellular Component (CC)\n"
        "**Literature data**:  Gene feature vector from literature data from web scrapping."
    )


def process_gene_disease(raw: Path, cfg: FileConfig) -> pd.DataFrame:
    """
    Process the gene-disease association matrix.

    Args:
        raw (Path): Directory containing raw files.
        cfg (FileConfig): File configuration.

    Returns:
        pd.DataFrame: Gene–disease associations with columns ['Gene ID', 'Disease ID'].
    """
    path = raw / cfg.gene_disease
    df = load_csv(
        path,
        header=None,
        names=["Gene ID", "Disease ID"],
        dtype={"Gene ID": int, "Disease ID": int},
    )
    if (
        df["Gene ID"].min() > 0 and df["Gene ID"].max() >= NB_GENES
    ):  # indexes are starting at 1
        df["Gene ID"] -= 1
    if (
        df["Disease ID"].min() > 0 and df["Disease ID"].max() >= NB_DISEASES
    ):  # indexes are starting at 1
        df["Disease ID"] -= 1
    return df


def process_gene_symbols(raw: Path, cfg: FileConfig) -> pd.DataFrame:
    """
    Align gene symbols to their numeric IDs.

    Args:
        raw (Path): Directory containing raw files.
        cfg (FileConfig): File configuration.

    Returns:
        pd.DataFrame: DataFrame with ['Gene ID', 'Gene Symbol'].
    """
    ids_df = load_csv(raw / cfg.gene_ids, header=None)
    sym_df = load_csv(raw / cfg.gene_symbols, header=None)
    aligned = sym_df.iloc[ids_df[0] - 1].reset_index(drop=True)
    return pd.DataFrame({"Gene ID": np.arange(len(aligned)), "Gene Symbol": aligned[0]})


def process_go(raw: Path, cfg: FileConfig) -> pd.DataFrame:
    """
    Load and clean gene ontology associations.

    Args:
        raw (Path): Directory containing raw files.
        cfg (FileConfig): File configuration.

    Returns:
        pd.DataFrame: GO associations with ['Gene ID', 'GO term ID'].
    """
    df = load_csv(raw / cfg.go, header=None, names=["Gene ID", "GO term ID", "_"])
    df = df.drop(columns=["_"])
    if (
        df["Gene ID"].min() > 0 and df["Gene ID"].max() >= NB_GENES
    ):  # indexes are starting at 1
        df["Gene ID"] -= 1
    if df["GO term ID"].min() > 0:  # indexes are starting at 1
        df["GO term ID"] -= 1
    return df


def process_interpro(raw: Path, cfg: FileConfig) -> pd.DataFrame:
    """
    Load and clean InterPro domain associations.

    Args:
        raw (Path): Directory containing raw files.
        cfg (FileConfig): File configuration.

    Returns:
        pd.DataFrame: InterPro associations with ['Gene ID', 'InterPro domain ID'].
    """
    df = load_csv(
        raw / cfg.interpro, header=None, names=["Gene ID", "InterPro domain ID", "_"]
    )
    df = df.drop(columns=["_"])
    if (
        df["Gene ID"].min() > 0 and df["Gene ID"].max() >= NB_GENES
    ):  # indexes are starting at 1
        df["Gene ID"] -= 1
    if df["InterPro domain ID"].min() > 0:  # indexes are starting at 1
        df["InterPro domain ID"] -= 1
    return df


def process_uniprot(raw: Path, cfg: FileConfig) -> pd.DataFrame:
    """
    Load and clean UniProt associations.

    Args:
        raw (Path): Directory containing raw files.
        cfg (FileConfig): File configuration.

    Returns:
        pd.DataFrame: UniProt associations with ['Gene ID', 'UniProt ID'].
    """
    df = load_csv(raw / cfg.uniprot, header=None, names=["Gene ID", "UniProt ID", "_"])
    df = df.drop(columns=["_"])
    if (
        df["Gene ID"].min() > 0 and df["Gene ID"].max() >= NB_GENES
    ):  # indexes are starting at 1
        df["Gene ID"] -= 1
    if df["UniProt ID"].min() > 0:  # indexes are starting at 1
        df["UniProt ID"] -= 1
    return df


def process_phenotypes(raw: Path, cfg: FileConfig) -> pd.DataFrame:
    """
    Extract phenotype association scores from a .mat file.

    Args:
        raw (Path): Directory containing raw files.
        cfg (FileConfig): File configuration.

    Returns:
        pd.DataFrame: DataFrame with ['Disease ID', 'Phenotypic term ID', 'Association score'].
    """
    mat = load_mat(raw / cfg.phen, "omimXterms_tf_Idf")
    coo = mat.tocoo()
    df = pd.DataFrame(
        {
            "Disease ID": coo.row.astype(np.int32),
            "Phenotypic term ID": coo.col.astype(np.int32),
            "Association score": coo.data,
        }
    )
    if (
        df["Disease ID"].min() > 0 and df["Disease ID"].max() > NB_DISEASES
    ):  # indexes are starting at 1
        df["Disease ID"] -= 1
    if df["Phenotypic term ID"].min() > 0:  # indexes are starting at 1
        df["Phenotypic term ID"] -= 1
    return df


def process_phen_terms(raw: Path, cfg: FileConfig) -> pd.DataFrame:
    """
    Load OMIM phenotype term labels from a .mat file.

    Args:
        raw (Path): Directory containing raw files.
        cfg (FileConfig): File configuration.

    Returns:
        pd.DataFrame: DataFrame with ['Disease ID', 'Phenotype term'].
    """
    raw_terms = load_mat(raw / cfg.phen_terms, "omim_terms")
    terms = [t.item() for t in raw_terms.squeeze()]
    df = pd.DataFrame({"Disease ID": np.arange(len(terms)), "Phenotype term": terms})
    if (
        df["Disease ID"].min() > 0 and df["Disease ID"].max() > NB_DISEASES
    ):  # indexes are starting at 1
        df["Disease ID"] -= 1
    return df


def _extract_nonzero(args):
    """
    Helper to parse one CSV row, extracting non-zero values.

    Args:
        args (tuple[int, str]): A tuple where the first element is the row index
            in the text matrix, and the second is the CSV-formatted line string.

    Returns:
        list[tuple[int, int, float]]: List of (line_index, column_index, value)
            for each non-zero entry in the row.
    """
    i, line = args
    out = []
    for j, tok in enumerate(line.rstrip("\n").split(",")):
        v = float(tok)
        if v != 0.0:
            out.append((i, j, v))
    return out


def process_text_data(raw: Path, cfg: FileConfig) -> pd.DataFrame:
    """
    Process the raw text feature matrix in parallel, then remap gene IDs.

    Reads a comma-delimited ‘text.csv’, extracts all non-zero values
    in parallel using multiprocessing, and merges with the gene_id map
    to produce a unified ['Gene ID', 'Feature ID', 'Value'] table.

    Args:
        raw (Path): Directory containing raw files.
        cfg (FileConfig): File configuration containing 'text' and 'gene_ids'.

    Returns:
        pd.DataFrame: DataFrame with columns ['Gene ID', 'Feature ID', 'Value']
            representing all non-zero literature-derived features.
    """
    text_path = raw / cfg.text

    # 1) Read all lines with their indices
    with text_path.open("r", encoding="utf-8") as f:
        indexed_lines = list(enumerate(f))

    # 2) Parallel extraction of non-zero entries
    with Pool(processes=cpu_count()) as pool:
        # chunksize tuned for large files
        chunks = pool.imap(_extract_nonzero, indexed_lines, chunksize=100)
        data = []
        for result in chunks:
            data.extend(result)

    # 3) Build DataFrame
    df = pd.DataFrame(data, columns=["Gene ID", "Feature ID", "Value"])

    # 4) Remap sparse indices -> canonical Gene ID
    map_df = load_csv(raw / cfg.gene_ids, header=None).reset_index()
    map_df.columns = ["Gene ID Map", "Gene ID"]

    merged = df.merge(
        map_df,
        how="right",
        on="Gene ID",
    )
    merged = merged.fillna(0)
    merged = merged.drop(columns=["Gene ID"]).rename(
        columns={"Gene ID Map": "Gene ID"}
    )[["Gene ID", "Feature ID", "Value"]]
    return merged


def write_readme(out: Path, cfg: FileConfig, rd_cfg: ReadmeConfig):
    """
    Generate a README.md summarizing all processed files.

    Args:
        out (Path): Output directory.
        cfg (FileConfig): File configuration.
        rd_cfg (ReadmeConfig): README content configuration.
    """
    df = pd.DataFrame(
        [
            {
                "File": e.file,
                "Description": e.description,
                "Rows": e.rows or "-",
                "Cols": e.cols or "-",
            }
            for e in rd_cfg.entries
        ]
    )
    readme_path = out / cfg.readme
    with readme_path.open("w", encoding="utf-8") as f:
        f.write("# Dataset Overview\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Field Descriptions\n")
        f.write(rd_cfg.terms_description)
        f.write("\n")


def verify_datasets_consistency(datasets: dict[str, pd.DataFrame], cfg: FileConfig):
    """
    Verify that gene and disease IDs across all datasets are consistent with the
        gene-disease map.

    Args:
        datasets (dict[str, pd.DataFrame]): Mapping from output filename to DataFrame.
        cfg (FileConfig): File configuration holding output keys.

    Raises:
        ValueError: If any dataset contains IDs not present in the gene-disease
            associations.
    """
    # Reference gene-disease associations
    valid_genes = set(np.arange(NB_GENES))
    valid_diseases = set(np.arange(NB_DISEASES))

    # Check gene-based datasets
    for key in (cfg.out_go, cfg.out_uniprot, cfg.out_interpro, cfg.out_gene_literature):
        df = datasets[key]
        missing = set(df["Gene ID"]) - valid_genes
        if missing:
            raise ValueError(
                f"Dataset '{key}' contains Gene IDs not "
                f"in '{cfg.out_gene_disease}': {sorted(missing)[:5]}"
            )

    # Check phenotype dataset
    phen_df = datasets[cfg.out_phen]
    missing_ph = set(phen_df["Disease ID"]) - valid_diseases
    if missing_ph:
        raise ValueError(
            f"Phenotype dataset '{cfg.out_phen}' contains Disease IDs not in '{cfg.out_gene_disease}': {sorted(missing_ph)[:5]}..."
        )


def save_dataframe(df: pd.DataFrame, path: Path):
    """
    Save a DataFrame to CSV, without the index.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        path (Path): Destination file path.
    """
    df.to_csv(path, index=False)


def main():
    """
    CLI entry point for dataset preprocessing.

    Parses arguments, processes all data files, and writes outputs.
    """
    parser = argparse.ArgumentParser(description="Preprocess biological datasets")
    parser.add_argument(
        "--raw-dir", type=Path, required=True, help="Directory with raw files"
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raw, out = args.raw_dir, args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    cfg = FileConfig()
    rd_cfg = ReadmeConfig()

    logging.info("Processing datasets...")
    datasets = {
        cfg.out_gene_disease: process_gene_disease(raw, cfg),
        cfg.out_gene_symbols: process_gene_symbols(raw, cfg),
        cfg.out_go: process_go(raw, cfg),
        cfg.out_interpro: process_interpro(raw, cfg),
        cfg.out_uniprot: process_uniprot(raw, cfg),
        cfg.out_phen: process_phenotypes(raw, cfg),
        cfg.out_phen_terms: process_phen_terms(raw, cfg),
        cfg.out_gene_literature: process_text_data(raw, cfg),
    }
    for key, df in datasets.items():
        logging.info(
            "ID in %s ranges from %d to %d.",
            key,
            df.iloc[:, 0].min(),
            df.iloc[:, 0].max(),
        )

    # Verify consistency before saving
    verify_datasets_consistency(datasets, cfg)

    for fname, df in datasets.items():
        out_path = out / fname
        if out_path.exists() and not args.overwrite:
            logging.warning(f"{fname} exists; skipping (use --overwrite to replace).")
            continue
        save_dataframe(df, out_path)
        logging.info(f"Wrote {fname}: {df.shape[0]} rows, {df.shape[1]} cols")

    write_readme(out, cfg, rd_cfg)
    logging.info("README.md generated.")


if __name__ == "__main__":
    main()
