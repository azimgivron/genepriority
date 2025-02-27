from genepriority.preprocessing.dataloader import DataLoader
from genepriority.utils import mask_sparse_containing_0s

import pytest
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Generator

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
        pd.DataFrame(data, columns=["Gene ID", "Disease ID"]).to_csv(file_path, index=False)
        created = True

    yield file_path

    # Teardown: delete the file only if it was created by this fixture.
    if created and file_path.exists():
        file_path.unlink()
        path.rmdir()

def test_dataloader_construction(nb_genes: int, nb_diseases: int, path: Path):
    """Test DataLoader construction and its masking behavior."""
    train_size: float = 0.8

    # DataLoader without zero sampling.
    without_0s = DataLoader(
        nb_genes,
        nb_diseases,
        path,
        seed=42,
        num_splits=1,
        num_folds=2,
        train_size=train_size,
        min_associations=2,
        validation_size=None,
        zero_sampling_factor=0,
    )
    without_0s(filter_column="Disease ID")

    assert not (without_0s.omim1.data == 0).any()
    assert not (without_0s.omim2.data == 0).any()
    assert (without_0s.omim1.data == 1).sum() == without_0s.omim1.nnz

    train_mask, test_mask = next(iter(without_0s.splits))
    data = without_0s.omim1
    train_data = mask_sparse_containing_0s(data, train_mask)
    test_data = mask_sparse_containing_0s(data, test_mask)
    assert np.isclose(
        (train_data.data == 1).sum(), np.ceil(train_size * (data.data == 1).sum())
    )
    assert np.isclose(
        (test_data.data == 1).sum(), np.ceil((1 - train_size) * (data.data == 1).sum())
    )

    # DataLoader with zero_sampling_factor set to None.
    with_none_factor = DataLoader(
        nb_genes,
        nb_diseases,
        path,
        seed=42,
        num_splits=1,
        num_folds=2,
        train_size=train_size,
        min_associations=2,
        validation_size=None,
        zero_sampling_factor=None,
    )
    with_none_factor(filter_column="Disease ID")

    assert not (with_none_factor.omim1.data == 0).any()
    assert not (with_none_factor.omim2.data == 0).any()

    # DataLoader with a validation set.
    validation_size: float = 0.1
    with_validation_set = DataLoader(
        nb_genes,
        nb_diseases,
        path,
        seed=42,
        num_splits=1,
        num_folds=2,
        train_size=train_size,
        min_associations=2,
        validation_size=validation_size,
        zero_sampling_factor=0,
    )
    with_validation_set(filter_column="Disease ID")

    assert not (with_validation_set.omim1.data == 0).any()
    assert not (with_validation_set.omim2.data == 0).any()

    train_mask, test_mask = next(iter(with_validation_set.splits))
    data = with_validation_set.omim1
    train_data = mask_sparse_containing_0s(data, train_mask)
    test_data = mask_sparse_containing_0s(data, test_mask)

    assert np.isclose(
        (train_data.data == 1).sum(),
        np.floor(np.ceil(train_size * (data.data == 1).sum()) * (1 - validation_size)),
    )
    assert np.isclose(
        (test_data.data == 1).sum(),
        np.ceil((1 - validation_size) * (1 - train_size) * (data.data == 1).sum()),
    )

    # DataLoader with zero sampling.
    zero_sampling_factor: int = 5
    with0s = DataLoader(
        nb_genes,
        nb_diseases,
        path,
        seed=42,
        num_splits=1,
        num_folds=2,
        train_size=0.8,
        min_associations=2,
        validation_size=None,
        zero_sampling_factor=zero_sampling_factor,
    )
    with0s(filter_column="Disease ID")

    assert (with0s.omim1.data == 0).any()
    assert (with0s.omim2.data == 0).any()
    assert (with0s.omim1.data == 1).sum() + (with0s.omim1.data == 0).sum() == with0s.omim1.nnz
    assert ((with0s.omim1.data == 1).sum() * zero_sampling_factor) == ((with0s.omim1.data == 0).sum())

    # DataLoader without folds.
    without_folds = DataLoader(
        nb_genes,
        nb_diseases,
        path,
        seed=42,
        num_splits=1,
        num_folds=None,
        train_size=train_size,
        min_associations=2,
        validation_size=None,
        zero_sampling_factor=0,
    )
    without_folds(filter_column="Disease ID")

    assert not (without_folds.omim1.data == 0).any()
    assert without_folds.omim2 is None

    # DataLoader without splits.
    without_splits = DataLoader(
        nb_genes,
        nb_diseases,
        path,
        seed=42,
        num_splits=None,
        num_folds=2,
        train_size=train_size,
        min_associations=2,
        validation_size=None,
        zero_sampling_factor=0,
    )
    without_splits(filter_column="Disease ID")

    assert without_splits.omim1 is None
    assert not (without_splits.omim2.data == 0).any()
