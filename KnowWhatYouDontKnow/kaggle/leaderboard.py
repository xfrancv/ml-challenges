#!/usr/bin/env python3
"""Score every submission under ``solutions/`` and write a markdown leaderboard.

    python leaderboard.py
    python leaderboard.py --solutions solutions/ --out leaderboard.md

Every ``submission.csv`` found anywhere below the solutions folder is scored
with ``evaluate.py``, once on the Public and once on the Private rows of the
solution file. A submission is named by the ``label.txt`` next to it if there
is one, otherwise by its folder path relative to the solutions folder.

The score is ``AvgRegAtCoverage``; lower is better. The table is sorted by the
private score.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
USAGES = ("Public", "Private")
# the line reads "AvgRegAtCoverage (c = 0.8) = +0.096438      lower is better"
SCORE_RE = re.compile(r"^AvgRegAtCoverage \([^)]*\)\s*=\s*([-+]?\d+\.\d+)",
                      re.MULTILINE)


def evaluate(submission: Path, solution: Path, usage: str, reps: int) -> float:
    """Run ``evaluate.py`` on one submission and return its headline score."""
    # evaluate.py writes figures next to the submission unless told otherwise;
    # they are of no use here, so send them to a throwaway folder
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [sys.executable, str(HERE / "evaluate.py"), str(submission),
               str(solution), "--usage", usage, "--reps", str(reps),
               "--out-dir", tmp]
        res = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    if res.returncode != 0:
        tail = (res.stderr.strip() or res.stdout.strip()).splitlines()[-1:]
        raise RuntimeError(tail[0] if tail else f"exit code {res.returncode}")
    found = SCORE_RE.search(res.stdout)
    if found is None:
        raise RuntimeError("no AvgRegAtCoverage line in the evaluate.py output")
    return float(found.group(1))


def submission_name(submission: Path, root: Path) -> str:
    label = submission.parent / "label.txt"
    if label.is_file():
        text = " ".join(label.read_text().split())
        if text:
            return text
    rel = submission.parent.relative_to(root)
    return rel.as_posix() if rel.parts else root.name


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--solutions", type=Path, default=HERE / "solutions",
                   help="folder searched recursively for submission.csv")
    p.add_argument("--solution", type=Path,
                   default=HERE / "out/final/kaggle/solution.csv",
                   help="solution file with a Usage column")
    p.add_argument("--out", type=Path, default=HERE / "leaderboard.md")
    p.add_argument("--reps", type=int, default=0,
                   help="bootstrap replicates passed to evaluate.py (default 0: "
                        "the leaderboard only needs the headline number)")
    p.add_argument("--jobs", type=int, default=4)
    args = p.parse_args()

    root = args.solutions.resolve()
    solution = args.solution.resolve()
    if not solution.is_file():
        raise SystemExit(f"solution file {solution} not found")
    submissions = sorted(root.rglob("submission.csv"))
    if not submissions:
        raise SystemExit(f"no submission.csv found under {root}")

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

    rows = [(submission_name(s, root), s.parent.relative_to(root).as_posix(),
             scores[s, "Public"], scores[s, "Private"]) for s in submissions]
    # failed submissions go last
    rows.sort(key=lambda r: (r[3] is None, r[3] or 0.0, r[2] is None, r[2] or 0.0))

    def fmt(v):
        return "failed" if v is None else f"{v:+.6f}"

    lines = ["# Leaderboard", "",
             "Score is `AvgRegAtCoverage` (lower is better), sorted by the "
             "private score.", "",
             "| # | Submission | Folder | Public | Private |",
             "|--:|:-----------|:-------|-------:|--------:|"]
    for i, (name, folder, public, private) in enumerate(rows, 1):
        name = name.replace("|", "\\|")
        lines.append(f"| {i} | {name} | `{folder}` | {fmt(public)} | {fmt(private)} |")
    args.out.write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nwritten to {args.out}")
    if any(v is None for v in scores.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
