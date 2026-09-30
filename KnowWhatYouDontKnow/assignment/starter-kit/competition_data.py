"""Load the competition data into numpy arrays and pandas tables.

Every image file is a ``.npy`` array of shape ``(n, 28, 28)``, dtype ``uint8``,
and **row k of the matching CSV describes image k**:

    train.csv            <->  train_images.npy
    dev_test_batches.csv <->  dev_images.npy
    test_batches.csv     <->  test_images.npy

    from competition_data import load_training, load_batches

    X, labels, locations = load_training(DATA)
    X_dev, dev_rows = load_batches(DATA, "dev")     # + dev_solution.csv columns
    X_test, test_rows = load_batches(DATA, "test")

``dev_rows`` and ``test_rows`` are sorted by ``id_test`` and then ``slot``, so
the images of one batch are contiguous; ``batch_slices`` yields them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_training(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(images (n, 28, 28) uint8, label (n,), location (n,))``."""
    data_dir = Path(data_dir)
    X = np.load(data_dir / "train_images.npy")
    table = pd.read_csv(data_dir / "train.csv")
    assert len(X) == len(table), "train.csv and train_images.npy disagree"
    return X, table["label"].to_numpy(), table["location"].to_numpy()


def load_batches(data_dir: Path, split: str) -> tuple[np.ndarray, pd.DataFrame]:
    """``(images, rows)`` of the ``"dev"`` or ``"test"`` batches.

    ``rows`` has ``row_id``, ``id_test``, ``slot`` and ``m``. For ``"dev"`` it
    also carries ``label`` and ``pred_ref`` from ``dev_solution.csv``.
    """
    assert split in ("dev", "test"), split
    data_dir = Path(data_dir)
    prefix = "dev_" if split == "dev" else ""
    X = np.load(data_dir / f"{split}_images.npy")
    rows = pd.read_csv(data_dir / f"{prefix}test_batches.csv")
    assert len(X) == len(rows), f"{prefix}test_batches.csv and its images disagree"
    if split == "dev":
        sol = pd.read_csv(data_dir / "dev_solution.csv")[
            ["row_id", "label", "pred_ref"]]
        rows = rows.merge(sol, on="row_id", how="left", validate="one_to_one")
    # Sort images and rows together, so one batch is one contiguous block.
    order = np.lexsort((rows["slot"].to_numpy(), rows["id_test"].to_numpy()))
    return X[order], rows.iloc[order].reset_index(drop=True)


def batch_slices(rows: pd.DataFrame):
    """``(id_test, slice)`` for every batch of a table from ``load_batches``."""
    ids = rows["id_test"].to_numpy()
    starts = np.concatenate([[0], np.flatnonzero(np.diff(ids)) + 1, [len(ids)]])
    for a, b in zip(starts[:-1], starts[1:]):
        yield int(ids[a]), slice(int(a), int(b))
