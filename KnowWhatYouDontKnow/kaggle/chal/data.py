"""Download and load TissueMNIST (MedMNIST v2) -- the only dataset here.

The archive is a single ``.npz`` of uint8 images and int labels, so neither the
``medmnist`` package nor Pillow is needed to read it.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

URL = ("https://zenodo.org/records/10519652/files/tissuemnist.npz?download=1")
FILENAME = "tissuemnist.npz"
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

NUM_CLASSES = 8
IMAGE_SHAPE = (28, 28)
CLASS_NAMES = [
    "Collecting Duct, Connecting Tubule",
    "Distal Convoluted Tubule",
    "Glomerular endothelial cells",
    "Interstitial endothelial cells",
    "Leukocytes",
    "Podocytes",
    "Proximal Tubule Segments",
    "Thick Ascending Limb",
]

_USER_AGENT = "challenge-tissuemnist/1.0 (+urllib)"


@dataclass
class Dataset:
    """``splits[name] = (X uint8 (N, 28, 28), y int64 (N,))``."""

    splits: dict[str, tuple[np.ndarray, np.ndarray]]

    @property
    def num_classes(self) -> int:
        return NUM_CLASSES


def download(data_root: Path = DATA_ROOT) -> Path:
    """Fetch the archive into ``data/tissuemnist/`` unless it is already there."""
    dest = Path(data_root) / "tissuemnist" / FILENAME
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(URL, headers={"User-Agent": _USER_AGENT})
    print(f"downloading {FILENAME} -> {dest}")
    with urllib.request.urlopen(req) as resp:          # noqa: S310 (trusted URL)
        with open(tmp, "wb") as fh:
            for chunk in iter(lambda: resp.read(1 << 16), b""):
                fh.write(chunk)
    tmp.rename(dest)
    return dest


def load(data_root: Path = DATA_ROOT, auto_download: bool = True) -> Dataset:
    path = Path(data_root) / "tissuemnist" / FILENAME
    if auto_download:
        path = download(data_root)
    npz = np.load(path)
    splits = {}
    for name in ("train", "val", "test"):
        splits[name] = (npz[f"{name}_images"].astype(np.uint8),
                        npz[f"{name}_labels"].astype(np.int64).reshape(-1))
    ds = Dataset(splits)
    for name, (X, y) in splits.items():
        assert len(X) == len(y), f"{name}: {len(X)} images vs {len(y)} labels"
        assert y.min() >= 0 and y.max() < NUM_CLASSES, f"{name}: label out of range"
    return ds
