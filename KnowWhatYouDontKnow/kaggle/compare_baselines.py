#!/usr/bin/env python3
"""Compare submissions on **shared** bootstrap resamples.

The pre-launch checks ask whether `bayes_total` and `bayes_epistemic` are
"separated by more than their bootstrap intervals". Comparing two independently
bootstrapped *marginal* intervals is the wrong test, and it is a conservative
one: the two rejectors are built on the **same base predictor** and differ only
in how they rank, so most of the variance in each is shared and cancels in the
difference. Two heavily overlapping marginal bands are perfectly compatible with
a difference that is significant on every single replicate.

This script resamples the batches once per replicate and scores every
submission on that same resample, then reports

* each submission's ``AvgRegAtCoverage`` with a marginal interval, and
* for each submission, the **paired** difference against the chosen reference:
  its interval, and the fraction of replicates on which it wins.

    python compare_baselines.py out/final/kaggle/organiser/test/solution.csv \\
        out/final/submissions/*.csv \\
        --usage Private --vs bayes_total
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from chal.bootstrap import paired_draws, regret_at_coverage
from chal.metric import COVERAGE

ALPHA = 0.05


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("solution", type=Path)
    p.add_argument("submissions", type=Path, nargs="+")
    p.add_argument("--usage", default=None,
                   choices=("Public", "Private", "Ignored"))
    p.add_argument("--vs", default=None,
                   help="reference submission name for the paired difference "
                        "(default: the best-scoring one)")
    p.add_argument("--coverage", type=float, default=COVERAGE)
    p.add_argument("--reps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    sol = pd.read_csv(args.solution)
    if args.usage is not None:
        if "Usage" not in sol.columns:
            raise SystemExit(f"{args.solution} has no Usage column")
        sol = sol[sol["Usage"] == args.usage].reset_index(drop=True)
    sol = sol.sort_values("row_id", kind="stable").reset_index(drop=True)

    names, preds, confs = [], [], []
    for path in args.submissions:
        sub = pd.read_csv(path)
        sub = sub[sub["row_id"].isin(sol["row_id"])]
        sub = sub.sort_values("row_id", kind="stable").reset_index(drop=True)
        assert sub["row_id"].equals(sol["row_id"]), f"{path}: row_id mismatch"
        names.append(path.stem)
        preds.append(sub["pred"].to_numpy())
        confs.append(sub["confidence"].to_numpy(dtype=float))

    label = sol["label"].to_numpy()
    ref_loss = (sol["pred_ref"].to_numpy() != label).astype(float)
    m_col = sol["m"].to_numpy()
    sizes = tuple(sorted(set(int(v) for v in np.unique(m_col))))
    rng = np.random.default_rng(args.seed)

    # (n_subs, reps) accumulated over sizes, then divided: AvgRegAtCoverage is
    # the unweighted mean over sizes, so the replicate-wise mean is too.
    draws = np.zeros((len(names), args.reps))
    point = np.zeros(len(names))
    per_size = {}
    for m in sizes:
        sel = m_col == m
        rank_id = np.arange(int(sel.sum()))
        codes, _ = pd.factorize(sol.loc[sel, "id_test"])
        excess = [(preds[s][sel] != label[sel]).astype(float) - ref_loss[sel]
                  for s in range(len(names))]
        conf = [c[sel] for c in confs]
        pt = np.array([regret_at_coverage(excess[s], conf[s], rank_id,
                                          args.coverage)
                       for s in range(len(names))])
        per_size[m] = pt
        point += pt
        draws += paired_draws(excess, conf, rank_id, codes, args.coverage,
                              args.reps, rng)
    point /= len(sizes)
    draws /= len(sizes)

    ref = args.vs or names[int(point.argmin())]
    assert ref in names, f"--vs {ref} is not among {names}"
    r = names.index(ref)

    print(f"\nAvgRegAtCoverage (c = {args.coverage}), "
          f"{args.usage or 'all rows'}, {args.reps} paired replicates")
    print(f"reference for the paired difference: {ref}\n")
    print(f"{'submission':<18}{'score':>9} {'marginal 95% CI':>21}"
          f" {'paired diff vs ref':>21} {'wins':>6}")
    print("-" * 79)
    for s in np.argsort(point):
        lo, hi = np.quantile(draws[s], [ALPHA / 2, 1 - ALPHA / 2])
        if s == r:
            diff_txt, win = "(reference)", ""
        else:
            d = draws[s] - draws[r]
            dlo, dhi = np.quantile(d, [ALPHA / 2, 1 - ALPHA / 2])
            diff_txt = f"[{dlo:+.4f}, {dhi:+.4f}]"
            win = f"{float((d < 0).mean()):.0%}"
        print(f"{names[s]:<18}{point[s]:>+9.4f} [{lo:+.4f}, {hi:+.4f}]"
              f" {diff_txt:>21} {win:>6}")
    print("-" * 79)
    print("'wins' = fraction of replicates on which the row beats the reference.")
    print("A paired interval excluding 0 is the separation criterion; two")
    print("overlapping marginal intervals are not evidence against it.")

    print(f"\nper-size Reg@{args.coverage:g} (point estimates)")
    hdr = "".join(f"{m:>9}" for m in sizes)
    print(f"{'submission':<18}{hdr}")
    for s in np.argsort(point):
        row = "".join(f"{per_size[m][s]:>+9.4f}" for m in sizes)
        print(f"{names[s]:<18}{row}")
    spread = np.array([per_size[m].max() - per_size[m].min() for m in sizes])
    print(f"{'spread (all)':<18}" + "".join(f"{v:>+9.4f}" for v in spread))

    # The spread over everything is dominated by whichever entry is uniformly
    # bad, which separates at every size and so says nothing about grid design.
    # What decides whether a size earns its share of the metric is whether it
    # separates the *best* entries from each other -- the ranking near the top
    # of the leaderboard is the only part anyone competes over.
    zero = oracle_idx(per_size, sizes)
    contenders = [s for s in np.argsort(point) if s != zero]
    if len(contenders) >= 2:
        a, b = contenders[0], contenders[1]
        gap = np.array([per_size[m][b] - per_size[m][a] for m in sizes])
        tag = f"{names[b][:8]}-{names[a][:8]}"
        print(f"{tag:<18}" + "".join(f"{v:>+9.4f}" for v in gap))
        print("\nThe last row is the gap between the top two contenders at each")
        print(f"size -- here {names[a]} vs {names[b]}, which is the comparison")
        print("the whole design rests on. A size whose gap is ~0 is dead weight:")
        print("it costs its share of the metric and returns no ranking")
        print("information. (The 'spread (all)' row cannot show this, because a")
        print("uniformly bad entry separates at every size.)")


def oracle_idx(per_size, sizes) -> int:
    """Index of a submission that is identically zero at every size, if any.

    That is the true-prior reference line rather than a competitor, so it is
    excluded from the "does this size separate contenders" spread.
    """
    n = len(next(iter(per_size.values())))
    for s in range(n):
        if all(abs(per_size[m][s]) < 1e-12 for m in sizes):
            return s
    return -1


if __name__ == "__main__":
    main()
