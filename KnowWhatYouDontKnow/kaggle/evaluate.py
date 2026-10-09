#!/usr/bin/env python3
"""Score a submission and plot how it behaves across batch sizes.

    python evaluate.py out/final/submissions/bayes_epistemic_em.csv \\
        out/final/kaggle/organiser/test/solution.csv
    python evaluate.py sub.csv sol.csv --usage Private --out-dir figures/

The headline number is exactly what Kaggle computes: ``score()`` from
``chal/metric.py``, the same function ``metric-template.ipynb`` contains. Every
other number here is diagnostic.

**Bootstrap intervals resample batches, not rows.** Rows inside one batch share
the location and the same adaptation evidence, so they are strongly dependent;
resampling rows would understate the intervals badly. The resampling is done
without re-sorting: the rows of a size group are sorted by confidence once, and
a replicate only changes each row's *multiplicity*, so the accepted prefix is
found by a cumulative sum. That is what makes 1000 replicates over 42 000 rows
cheap.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from chal.bootstrap import bootstrap_regret, regret_at_coverage
from chal.metric import COVERAGE, score

REPS = 1000
ALPHA = 0.05


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("submission", type=Path)
    p.add_argument("solution", type=Path)
    p.add_argument("--usage", default=None,
                   choices=("Public", "Private", "Ignored"),
                   help="score only this Usage (default: every row, which is "
                        "what the development benchmark has)")
    p.add_argument("--coverage", type=float, default=COVERAGE)
    p.add_argument("--reps", type=int, default=REPS,
                   help=f"bootstrap replicates (default {REPS}; 0 to skip)")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="write figures here (default: alongside the submission)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    sol = pd.read_csv(args.solution)
    sub = pd.read_csv(args.submission)
    rng = np.random.default_rng(args.seed)

    if args.usage is not None:
        if "Usage" not in sol.columns:
            raise SystemExit(
                f"--usage {args.usage} was asked for but {args.solution} has no "
                f"Usage column (the development benchmark has none by design)")
        keep = sol["Usage"] == args.usage
        sol = sol[keep].reset_index(drop=True)
        sub = sub[sub["row_id"].isin(sol["row_id"])].reset_index(drop=True)
        print(f"scoring the {args.usage} rows only: {len(sol):,} rows")

    # --- the headline number, from the metric Kaggle runs -------------------
    sol_for_score = sol.drop(columns=[c for c in ("Usage",) if c in sol.columns])
    sizes = tuple(sorted(set(int(v) for v in sol_for_score["m"].unique())))
    headline = score(sol_for_score.copy(), sub.copy(), "row_id",
                     coverage=args.coverage, expected_sizes=sizes)

    # --- per-size breakdown with batch-level bootstrap ---------------------
    merged = sol.merge(sub[["row_id", "pred", "confidence"]], on="row_id",
                       how="inner", validate="one_to_one")
    assert len(merged) == len(sol), "submission does not cover every solution row"
    merged["excess"] = ((merged["pred"] != merged["label"]).astype(float)
                        - (merged["pred_ref"] != merged["label"]).astype(float))

    rows = []
    for m in sizes:
        g = merged[merged["m"] == m]
        g = g.sort_values("row_id", kind="stable")
        excess = g["excess"].to_numpy()
        conf = g["confidence"].to_numpy(dtype=float)
        rank_id = np.arange(len(g))
        codes, _ = pd.factorize(g["id_test"])
        point = regret_at_coverage(excess, conf, rank_id, args.coverage)
        risk = float((g["pred"] != g["label"]).mean())
        ref_risk = float((g["pred_ref"] != g["label"]).mean())
        lo = hi = float("nan")
        if args.reps > 0:
            draws = bootstrap_regret(excess, conf, rank_id, codes,
                                     args.coverage, args.reps, rng)
            lo, hi = np.quantile(draws, [ALPHA / 2, 1 - ALPHA / 2])
        rows.append({"m": m, "n_batches": g["id_test"].nunique(), "B_m": len(g),
                     "reg_at_c": point, "lo": lo, "hi": hi,
                     "risk_full": risk, "ref_risk_full": ref_risk})
    table = pd.DataFrame(rows)

    # --- report ------------------------------------------------------------
    name = args.submission.stem
    print(f"\n{'=' * 74}")
    print(f"{name}   vs   {args.solution}")
    print(f"{'=' * 74}")
    print(f"{'m':>5} {'batches':>8} {'B_m':>8} {'Reg@%.2f' % args.coverage:>9} "
          f"{'95% CI':>19} {'err':>7} {'ref err':>8}")
    for r in table.itertuples():
        ci = (f"[{r.lo:+.4f}, {r.hi:+.4f}]" if np.isfinite(r.lo) else "")
        print(f"{r.m:>5} {r.n_batches:>8,} {r.B_m:>8,} {r.reg_at_c:>+9.4f} "
              f"{ci:>19} {r.risk_full:>7.4f} {r.ref_risk_full:>8.4f}")
    print("-" * 74)
    print(f"AvgRegAtCoverage (c = {args.coverage}) = {headline:+.6f}"
          f"      lower is better")
    mean_of_table = float(table["reg_at_c"].mean())
    assert abs(mean_of_table - headline) < 1e-9, (
        f"the per-size table ({mean_of_table:+.6f}) disagrees with score() "
        f"({headline:+.6f}) -- they must be the same computation")

    # --- figures -----------------------------------------------------------
    out_dir = Path(args.out_dir or args.submission.parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    _figure(table, merged, name, args.coverage, out_dir, sizes)
    table.to_csv(out_dir / f"{name}_per_size.csv", index=False)
    print(f"\nfigures and per-size table in {out_dir}/")


def _figure(table: pd.DataFrame, merged: pd.DataFrame, name: str,
            coverage: float, out_dir: Path, sizes) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    ax = axes[0]
    ax.axhline(0.0, color="0.5", lw=1, ls="--",
               label="reference (Bayes given the location)")
    if np.isfinite(table["lo"]).all():
        ax.fill_between(table["m"], table["lo"], table["hi"], alpha=0.2, color="C0")
    ax.plot(table["m"], table["reg_at_c"], "o-", color="C0", lw=1.8, ms=4,
            label=name)
    ax.set_xscale("log")
    ax.set_xlabel("batch size $m$")
    ax.set_ylabel(f"Reg@{coverage:g}")
    ax.set_title(f"selective regret at {coverage:.0%} coverage\n"
                 f"(95% bootstrap band, batch as the resampling unit)",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    cmap = plt.get_cmap("viridis")
    for i, m in enumerate(sizes):
        g = merged[merged["m"] == m].sort_values("row_id", kind="stable")
        excess = g["excess"].to_numpy()
        order = np.lexsort((np.arange(len(g)), -g["confidence"].to_numpy(float)))
        k = np.arange(1, len(g) + 1)
        curve = np.cumsum(excess[order]) / k
        ax.plot(k / len(g), curve, lw=1.4, color=cmap(i / max(1, len(sizes) - 1)),
                label=f"m = {m}")
    ax.axvline(coverage, color="0.4", ls=":", lw=1.2)
    ax.axhline(0.0, color="0.5", lw=1, ls="--")
    ax.set_xlabel("coverage")
    ax.set_ylabel("selective regret")
    ax.set_title("regret-coverage curve, one per batch size", fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.25)

    fig.suptitle(f"{name}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_regret.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
