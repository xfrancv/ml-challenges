#!/usr/bin/env python3
"""Score the baseline submissions and write a markdown leaderboard.

    python leaderboard_baseline.py
    python leaderboard_baseline.py --submissions out/final/submissions/ \\
        --out leaderboard_baseline.md

The baselines are the ``.csv`` files ``run_baselines.sh`` leaves directly in
the submissions folder, minus the per-size tables and the estimated location
prior, which are not submissions. Each one is scored with ``evaluate.py``, once
on the Public and once on the Private rows of the solution file, and is named
by its file name without the extension.

The score is ``AvgRegAtCoverage``; lower is better. The table is sorted by the
private score.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from leaderboard import HERE, USAGES, evaluate

# written into the submissions folder by run_baselines.sh, but not submissions
NOT_SUBMISSIONS = ("em_location_prior.csv",)


def find_submissions(root: Path) -> list[Path]:
    return sorted(f for f in root.glob("*.csv")
                  if f.name not in NOT_SUBMISSIONS
                  and not f.name.endswith("_per_size.csv"))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--submissions", type=Path,
                   default=HERE / "out/final/submissions",
                   help="folder holding the baseline submission .csv files")
    p.add_argument("--solution", type=Path,
                   default=HERE / "out/final/kaggle/solution.csv",
                   help="solution file with a Usage column")
    p.add_argument("--out", type=Path, default=HERE / "leaderboard_baseline.md")
    p.add_argument("--reps", type=int, default=0,
                   help="bootstrap replicates passed to evaluate.py (default 0: "
                        "the leaderboard only needs the headline number)")
    p.add_argument("--jobs", type=int, default=4)
    args = p.parse_args()

    root = args.submissions.resolve()
    solution = args.solution.resolve()
    if not solution.is_file():
        raise SystemExit(f"solution file {solution} not found")
    submissions = find_submissions(root)
    if not submissions:
        raise SystemExit(f"no submission .csv found in {root}")

    tasks = [(s, u) for s in submissions for u in USAGES]

    def run(task):
        sub, usage = task
        try:
            return evaluate(sub, solution, usage, args.reps)
        except RuntimeError as e:
            print(f"FAILED {sub} ({usage}): {e}", file=sys.stderr)
            return None

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        scores = dict(zip(tasks, pool.map(run, tasks)))

    rows = [(s.stem, s.name, scores[s, "Public"], scores[s, "Private"])
            for s in submissions]
    # failed submissions go last
    rows.sort(key=lambda r: (r[3] is None, r[3] or 0.0, r[2] is None, r[2] or 0.0))

    def fmt(v):
        return "failed" if v is None else f"{v:+.6f}"

    lines = ["# Baseline leaderboard", "",
             "Score is `AvgRegAtCoverage` (lower is better), sorted by the "
             "private score.", "",
             "| # | Baseline | File | Public | Private |",
             "|--:|:---------|:-----|-------:|--------:|"]
    for i, (name, file, public, private) in enumerate(rows, 1):
        lines.append(f"| {i} | {name} | `{file}` | {fmt(public)} | {fmt(private)} |")
    args.out.write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nwritten to {args.out}")
    if any(v is None for v in scores.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
