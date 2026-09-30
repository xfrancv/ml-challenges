#!/usr/bin/env python3
"""Write a valid submission -- the format, and nothing more.

For every image it predicts the most frequent class of the training data, with
the same confidence for all. That is ``sample_submission.csv``: it looks at no
image and scores badly, and it exists to show the three columns

    row_id      copied from the batch listing; one row per image
    pred        the predicted label, an integer 0..7
    confidence  any real number; higher means "keep this prediction"

Only the *ranking* induced by ``confidence`` matters -- the organiser sorts by
it, keeps the most confident 80 %, and scores those. Ties are broken by
ascending ``row_id``. The ranking is pooled **across batches** of the same size,
so confidences have to be comparable between batches, not just within one.

Replace ``predict_batch`` with your own predictor; everything else stays.

    # the offline development benchmark
    python make_sample_submission.py --data-dir ../competition-data --split dev \\
        --out dev_submission.csv
    python evaluate.py dev_submission.csv ../competition-data/dev_solution.csv

    # the real thing, for uploading to Kaggle
    python make_sample_submission.py --data-dir ../competition-data --split test \\
        --out submission.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from competition_data import batch_slices, load_batches, load_training


def predict_batch(images: np.ndarray, majority: int
                  ) -> tuple[np.ndarray, np.ndarray]:
    """``(pred, confidence)`` for the ``m`` images of one batch.

    The whole batch is available here, not just one image at a time.
    """
    m = len(images)
    return np.full(m, majority, dtype=np.int64), np.zeros(m)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--split", choices=("dev", "test"), default="dev")
    p.add_argument("--out", type=Path, default=Path("dev_submission.csv"))
    args = p.parse_args()

    _X, labels, _locations = load_training(args.data_dir)
    majority = int(np.bincount(labels).argmax())

    X, rows = load_batches(args.data_dir, args.split)
    pred = np.empty(len(rows), dtype=np.int64)
    conf = np.empty(len(rows))
    for _id, sl in batch_slices(rows):
        pred[sl], conf[sl] = predict_batch(X[sl], majority)

    out = pd.DataFrame({"row_id": rows["row_id"], "pred": pred,
                        "confidence": conf})
    out.sort_values("row_id", kind="stable").to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(out):,} rows of the {args.split} batches)")


if __name__ == "__main__":
    main()
