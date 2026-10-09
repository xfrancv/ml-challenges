#!/usr/bin/env python3
"""Download and verify the TissueMNIST archive.

    python download_data.py
    python download_data.py --data-root /scratch/data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from chal import data as chdata


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=chdata.DATA_ROOT)
    args = p.parse_args()

    path = chdata.download(args.data_root)
    print(f"archive: {path}  ({path.stat().st_size / 1e6:.1f} MB)")

    ds = chdata.load(args.data_root, auto_download=False)
    print(f"{'split':<8} {'images':>9}  class counts")
    for name, (X, y) in ds.splits.items():
        counts = np.bincount(y, minlength=chdata.NUM_CLASSES)
        print(f"{name:<8} {len(X):>9,}  {counts.tolist()}")
    n_eval = len(ds.splits["val"][1]) + len(ds.splits["test"][1])
    print(f"\nevaluation pool (val + test merged): {n_eval:,}, "
          f"{n_eval // 3:,} per usage")


if __name__ == "__main__":
    main()
