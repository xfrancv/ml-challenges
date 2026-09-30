#!/usr/bin/env python3
"""Score a submission offline, exactly the way the leaderboard will.

Takes a submission and the matching solution file and prints
**AvgRegAtCoverage** -- the competition metric -- plus the per-batch-size
breakdown that tells you *where* your predictor is winning or losing.

The score comes from ``score()`` in ``metric.py``, which is the same function
the organisers run on Kaggle. A number you get here is a number you would get on
the leaderboard, for the same rows.

    python evaluate.py dev_sample_submission.csv ../competition-data/dev_solution.csv
    python evaluate.py my_submission.csv ../competition-data/dev_solution.csv --plot

**Lower is better.** The reference predictor makes the best possible prediction
knowing each batch's location, so the expected score is never below zero; a
slightly negative number is chance, not a win.

You can only score yourself on the *development* batches
(``dev_test_batches.csv`` / ``dev_solution.csv``), because those use images
whose labels you have. The real test labels are not in the competition data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from metric import COVERAGE, score


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("submission", type=Path)
    p.add_argument("solution", type=Path,
                   help="dev_solution.csv from the competition data")
    p.add_argument("--coverage", type=float, default=COVERAGE)
    p.add_argument("--plot", action="store_true",
                   help="also save a regret-coverage curve (needs matplotlib)")
    args = p.parse_args()

    sol = pd.read_csv(args.solution)
    sub = pd.read_csv(args.submission)
    if "Usage" in sol.columns:
        sol = sol.drop(columns=["Usage"])

    sizes = tuple(sorted(set(int(v) for v in sol["m"].unique())))
    total = score(sol.copy(), sub.copy(), "row_id", coverage=args.coverage,
                  expected_sizes=sizes)

    merged = sol.merge(sub[["row_id", "pred", "confidence"]], on="row_id",
                       how="inner", validate="one_to_one")
    if len(merged) != len(sol):
        raise SystemExit("your submission does not cover every solution row")
    merged["excess"] = ((merged["pred"] != merged["label"]).astype(float)
                        - (merged["pred_ref"] != merged["label"]).astype(float))

    print(f"\n{args.submission.name}  vs  {args.solution.name}")
    print("=" * 68)
    print(f"{'m':>5} {'rows':>8} {'Reg@%.2f' % args.coverage:>10} "
          f"{'your err':>9} {'ref err':>9}  {'':>12}")
    per_m = []
    for m in sizes:
        g = merged[merged["m"] == m].sort_values("row_id", kind="stable")
        excess = g["excess"].to_numpy()
        conf = g["confidence"].to_numpy(dtype=float)
        k = max(1, int(np.ceil(args.coverage * len(g))))
        order = np.lexsort((np.arange(len(g)), -conf))
        reg = float(excess[order[:k]].mean())
        per_m.append(reg)
        err = float((g["pred"] != g["label"]).mean())
        ref = float((g["pred_ref"] != g["label"]).mean())
        bar = "#" * int(max(0.0, reg) * 200)
        print(f"{m:>5} {len(g):>8,} {reg:>+10.4f} {err:>9.4f} {ref:>9.4f}  {bar}")
    print("-" * 68)
    print(f"AvgRegAtCoverage (c = {args.coverage}) = {total:+.6f}"
          f"      <-- lower is better")
    assert abs(float(np.mean(per_m)) - total) < 1e-9

    print(f"\n'your err' and 'ref err' are error rates at FULL coverage, for "
          f"context only;\nthe score uses only the most confident "
          f"{args.coverage:.0%} of each batch size.")
    if total > 0:
        print("A positive score means the reference predictor -- which knows "
              "each batch's\nlocation -- beat you on the rows you were most "
              "confident about.")

    if args.plot:
        _plot(merged, sizes, args.coverage, args.submission)


def _plot(merged: pd.DataFrame, sizes, coverage: float, submission: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        raise SystemExit("--plot needs matplotlib: pip install matplotlib")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    cmap = plt.get_cmap("viridis")
    for i, m in enumerate(sizes):
        g = merged[merged["m"] == m].sort_values("row_id", kind="stable")
        excess = g["excess"].to_numpy()
        order = np.lexsort((np.arange(len(g)), -g["confidence"].to_numpy(float)))
        k = np.arange(1, len(g) + 1)
        ax.plot(k / len(g), np.cumsum(excess[order]) / k, lw=1.5,
                color=cmap(i / max(1, len(sizes) - 1)), label=f"m = {m}")
    ax.axvline(coverage, color="0.4", ls=":", lw=1.2)
    ax.axhline(0.0, color="0.5", lw=1, ls="--")
    ax.set_xlabel("coverage (fraction of predictions kept)")
    ax.set_ylabel("selective regret")
    ax.set_title(f"{submission.name}\nthe score reads each curve at "
                 f"coverage {coverage:g}", fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = submission.with_suffix("").with_name(submission.stem + "_regret.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
