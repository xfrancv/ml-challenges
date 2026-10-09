#!/usr/bin/env python3
"""Estimate the location prior ``w`` by EM on the development batches.

This is the step the competitors are meant to discover, done with exactly what
they have: the development labels (``solution.csv`` of ``organiser/dev``, which
is the published ``dev_solution.csv``) and the 9 location priors, i.e. the class
frequencies of each location in ``train.csv``. Given the labels, a batch's
likelihood under location ``l`` is ``prod_i pi_l(y_i)`` -- the images add
nothing, since ``p(x | y)`` is the same at every location -- so EM is exact
maximum likelihood (``chal.locprior``).

Also reports a batch bootstrap of the estimate: the check that the development
set is large enough to estimate ``w`` usefully.

    python estimate_location_prior.py out/final/kaggle/organiser/dev \\
        out/final/submissions/em_location_prior.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from chal.inference import batch_starts_from_ids
from chal.locprior import em_location_prior, label_loglik, \
    read_location_prior, total_variation, write_location_prior


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dev_dir", type=Path, help="organiser/dev of make_package.py")
    p.add_argument("out", type=Path, help="CSV receiving the estimate")
    p.add_argument("--reps", type=int, default=200,
                   help="batch-bootstrap replicates (default 200; 0 to skip)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    sol = pd.read_csv(args.dev_dir / "solution.csv").sort_values(
        ["id_test", "slot"], kind="stable").reset_index(drop=True)
    priors = pd.read_csv(args.dev_dir / "test_priors.csv").sort_values("id")
    theta = priors[[c for c in priors.columns if c.startswith("p")]].to_numpy()
    starts = batch_starts_from_ids(sol["id_test"].to_numpy())
    loglik = label_loglik(sol["label"].to_numpy(), starts, np.log(theta))
    w_hat, it = em_location_prior(loglik)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_location_prior(w_hat, args.out)

    L = len(w_hat)
    uniform = np.full(L, 1.0 / L)
    print(f"EM on {len(starts):,} development batches ({len(sol):,} rows), "
          f"{it} iterations")
    print("  estimated w : " + "  ".join(f"{v:.4f}" for v in w_hat))

    true_file = args.dev_dir / "location_prior.csv"
    w = read_location_prior(true_file) if true_file.exists() else None
    if w is not None:
        print("  true w      : " + "  ".join(f"{v:.4f}" for v in w))
        print(f"  TV(w_hat, w) = {total_variation(w_hat, w):.4f}   "
              f"TV(uniform, w) = {total_variation(uniform, w):.4f}")

    if args.reps > 0:
        rng = np.random.default_rng(args.seed)
        B = len(loglik)
        draws = np.stack([em_location_prior(loglik[rng.integers(0, B, B)])[0]
                          for _ in range(args.reps)])
        se = draws.std(axis=0)
        print("  bootstrap SE: " + "  ".join(f"{v:.4f}" for v in se)
              + f"   ({args.reps} batch resamples)")
        if w is not None:
            tv = np.array([total_variation(d, w) for d in draws])
            print(f"  TV(w_hat, w) over replicates: median {np.median(tv):.4f}, "
                  f"95% {np.quantile(tv, 0.95):.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
