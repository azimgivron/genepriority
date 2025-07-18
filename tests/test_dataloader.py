from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest

from genepriority.preprocessing.dataloader import DataLoader
from genepriority.utils import mask_sparse_containing_0s

# Set global seed for reproducibility
np.random.seed(42)


@pytest.fixture(name="nb_genes", scope="session")
def get_genes() -> int:
    """Return the number of genes for testing."""
    return 10


@pytest.fixture(name="nb_diseases", scope="session")
def get_diseases() -> int:
    """Return the number of diseases for testing."""
    return 10


@pytest.fixture(name="path", scope="session")
def get_path(nb_genes: int, nb_diseases: int) -> Generator[Path, None, None]:
    """Create and yield a CSV file path for gene-disease data; clean up after tests."""
    path = Path("./data")
    path.mkdir(exist_ok=True)
    filename = "gene-disease.csv"
    file_path = path / filename

    created: bool = False
    if not file_path.exists():
        # Generate reproducible random gene-disease associations.
        genes_ID = np.random.randint(0, nb_genes, 10, dtype=np.int64)
        diseases_ID = np.random.randint(0, nb_diseases, 10, dtype=np.int64)
        data = np.vstack((genes_ID, diseases_ID)).T
        pd.DataFrame(data, columns=["Gene ID", "Disease ID"]).to_csv(
            file_path, index=False
        )
        created = True

    yield file_path

    # Teardown: delete the file only if it was created by this fixture.
    if created and file_path.exists():
        file_path.unlink()
        path.rmdir()


def test_dataloader_construction(nb_genes: int, nb_diseases: int, path: Path):
    """Test DataLoader construction and its masking behavior."""
    # DataLoader without zeros.
    num_folds = 5
    validation_size = 0.1
    zero_sampling_factor = 5

    without_0s = DataLoader(
        nb_genes,
        nb_diseases,
        path,
        seed=42,
        num_folds=num_folds,
        validation_size=validation_size,
        zero_sampling_factor=0,
    )
    assert not (without_0s.omim.data == 0).any()
    assert (without_0s.omim.data == 1).sum() == without_0s.omim.nnz

    train_mask, test_mask, val_mask = next(iter(without_0s.omim_masks))
    data = without_0s.omim
    train_data = mask_sparse_containing_0s(data, train_mask)
    test_data = mask_sparse_containing_0s(data, test_mask)
    val_data = mask_sparse_containing_0s(data, val_mask)
    tot = len(without_0s.omim.data)
    train_size = (num_folds - 1) / num_folds * (1 - validation_size)
    assert tot == len(train_data.data) + len(test_data.data) + len(val_data.data)
    assert np.isclose(len(train_data.data) / tot, np.floor(train_size * tot) / tot)
    assert np.isclose(len(val_data.data) / tot, validation_size)
    assert np.isclose(
        len(test_data.data) / tot,
        1 - np.floor(train_size * tot) / tot - validation_size,
    )

    # DataLoader with zero sampling.
    with0s = DataLoader(
        nb_genes,
        nb_diseases,
        path,
        seed=42,
        num_folds=2,
        validation_size=validation_size,
        zero_sampling_factor=zero_sampling_factor,
    )
    assert (with0s.omim.data == 0).any()
    assert (with0s.omim.data == 1).sum() + (
        with0s.omim.data == 0
    ).sum() == with0s.omim.nnz
    assert ((with0s.omim.data == 1).sum() * zero_sampling_factor) == (
        (with0s.omim.data == 0).sum()
    )
