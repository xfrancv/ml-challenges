#!/usr/bin/env python3
"""Rotate every image once and split the data.

1. **Rotation.** Every image of TissueMNIST -- train, val and test splits -- is
   rotated once by 90, 180 or 270 degrees (``chal.transform``). No noise. The
   rotated image is the one the label model sees and the one released.
2. **Split.** The original training split is divided, stratified by the
   original class, into

   * the **secret set** (``--secret-fraction``, 30 %): fits the label model,
     never released;
   * the **challenge set**, itself split into the **training data**
     (released, labeled, with locations) and the **development pool**
     (``--dev-fraction`` of it, 10 %) the development batches are drawn from.

   The **test pool** is the original validation + test splits, partitioned by
   original class into the three Kaggle ``Usage`` pools.

The original labels are used for this stratification and for fitting the
label model, nothing else. They are kept in ``split.npz`` only so that the
organisers can report how the generated labels relate to them.

Output: ``out_dir/split.npz`` (organiser only), ``out_dir/report.txt``.

    python make_split.py out/final/split
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from chal import data as chdata
from chal.splits import USAGES, assign_pools
from chal.transform import rotate


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--data-root", type=Path, default=chdata.DATA_ROOT)
    p.add_argument("--secret-fraction", type=float, default=0.30,
                   help="share of the original training split that fits the "
                        "label model (default 0.30)")
    p.add_argument("--dev-fraction", type=float, default=0.10,
                   help="share of the challenge set held out as the "
                        "development pool (default 0.10)")
    p.add_argument("--seed", type=int, default=20260920)
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    master = np.random.default_rng(args.seed)
    rng_rot_train, rng_rot_eval = master.spawn(2)
    split_seed = int(master.integers(1 << 31))

    ds = chdata.load(args.data_root)
    X_orig, y_orig = ds.splits["train"]
    X_eval = np.concatenate([ds.splits["val"][0], ds.splits["test"][0]])
    y_eval = np.concatenate([ds.splits["val"][1], ds.splits["test"][1]])

    # --- 1. rotation, once per image ---------------------------------------
    X_orig, rot_orig = rotate(X_orig, rng_rot_train)
    X_eval, rot_eval = rotate(X_eval, rng_rot_eval)

    # --- 2. split ------------------------------------------------------------
    idx = np.arange(len(y_orig))
    idx_challenge, idx_secret = train_test_split(
        idx, test_size=args.secret_fraction, stratify=y_orig,
        random_state=split_seed)
    idx_train, idx_devpool = train_test_split(
        idx_challenge, test_size=args.dev_fraction,
        stratify=y_orig[idx_challenge], random_state=split_seed + 1)
    idx_secret, idx_train, idx_devpool = (np.sort(idx_secret),
                                          np.sort(idx_train),
                                          np.sort(idx_devpool))
    assert len(idx_secret) + len(idx_train) + len(idx_devpool) == len(y_orig)
    usage = assign_pools(y_eval, seed=split_seed + 2)

    run_id = "".join(f"{v:08x}" for v in master.integers(0, 1 << 32, 4))
    np.savez(
        out_dir / "split.npz",
        run_id=run_id, seed=args.seed, usages=np.array(USAGES),
        secret_fraction=args.secret_fraction, dev_fraction=args.dev_fraction,
        # images as rotated; *_source indexes the original split, *_rot is the
        # number of quarter turns applied
        secret_images=X_orig[idx_secret], secret_orig_label=y_orig[idx_secret],
        secret_source=idx_secret, secret_rot=rot_orig[idx_secret],
        train_images=X_orig[idx_train], train_orig_label=y_orig[idx_train],
        train_source=idx_train, train_rot=rot_orig[idx_train],
        devpool_images=X_orig[idx_devpool],
        devpool_orig_label=y_orig[idx_devpool],
        devpool_source=idx_devpool, devpool_rot=rot_orig[idx_devpool],
        # the test pool: original val + test, in that order
        test_images=X_eval, test_orig_label=y_eval,
        test_source=np.arange(len(y_eval)), test_rot=rot_eval,
        test_usage=usage,
    )

    Y = chdata.NUM_CLASSES

    def freq(y):
        return "  ".join(f"{v:.4f}" for v in np.bincount(y, minlength=Y) / len(y))

    lines = [
        "Split: secret set, training data, development pool, test pool",
        "=" * 78,
        f"timestamp : {datetime.now().isoformat(timespec='seconds')}",
        f"command   : {' '.join(sys.argv)}",
        f"run_id    : {run_id}",
        "-" * 78,
        f"{'part':<22} {'images':>8}   original class frequency",
        f"{'secret set':<22} {len(idx_secret):>8,}   {freq(y_orig[idx_secret])}",
        f"{'training data':<22} {len(idx_train):>8,}   {freq(y_orig[idx_train])}",
        f"{'development pool':<22} {len(idx_devpool):>8,}   "
        f"{freq(y_orig[idx_devpool])}",
        f"{'test pool':<22} {len(y_eval):>8,}   {freq(y_eval)}",
    ] + [
        f"  {u:<20} {int((usage == i).sum()):>8,}"
        for i, u in enumerate(USAGES)
    ] + [
        "-" * 78,
        "every image rotated once by 90/180/270 degrees; no noise",
        "",
    ]
    report = "\n".join(lines)
    (out_dir / "report.txt").write_text(report)
    print(report)
    print(f"wrote {out_dir}/split.npz, {out_dir}/report.txt")


if __name__ == "__main__":
    main()
