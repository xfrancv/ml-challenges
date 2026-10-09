#!/usr/bin/env python3
"""Generate ``metric-template.ipynb`` from ``chal/metric.py``.

Kaggle runs a notebook, not a package, so the metric has to be a single
self-contained cell. Rather than maintaining that cell by hand next to the copy
``evaluate.py`` imports -- two copies that would eventually disagree, in the one
piece of code that cannot be debugged after launch -- the notebook is
**generated** from ``chal/metric.py``, which is why that module imports nothing
but numpy and pandas.

    python make_metric_notebook.py metric-template.ipynb
    python -m doctest chal/metric.py -v      # the notebook's tests, run locally
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HEADER = """# Competition metric: AvgRegAtCoverage
#
# GENERATED FILE -- do not edit here. Edit challenge/{source} and re-run
#     python make_metric_notebook.py {out} --source {source}
# so that this cell and the function evaluate.py imports stay the same code.
#
# Kaggle contract: a function named `score` whose first three arguments are
# (solution, submission, row_id_column_name), every argument annotated, and a
# single finite float returned. All solution columns except `Usage` are passed
# through, which is where `m`, `label` and `pred_ref` come from.
"""

FOOTER = '''

if __name__ == "__main__":
    import doctest
    results = doctest.testmod(verbose=False)
    print(f"doctests: {results.attempted - results.failed}"
          f"/{results.attempted} passed")
    assert results.failed == 0
'''


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path, nargs="?", default=Path("metric-template.ipynb"))
    p.add_argument("--source", type=Path, default=Path("chal/metric.py"))
    args = p.parse_args()

    src = args.source.read_text()
    # The module docstring describes the file's role in the repository, which is
    # noise inside the notebook; score()'s own docstring is what Kaggle renders.
    body = src.split('"""', 2)[2].lstrip("\n")
    body = body.replace("from __future__ import annotations\n\n", "")
    cell = HEADER.format(source=args.source, out=args.out) + "\n" \
        + body.rstrip() + FOOTER

    nb = {
        "metadata": {
            "kernelspec": {"language": "python", "display_name": "Python 3",
                           "name": "python3"},
            "language_info": {
                "name": "python", "version": "3.11.0",
                "mimetype": "text/x-python", "file_extension": ".py",
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
                "codemirror_mode": {"name": "ipython", "version": 3}},
            "kaggle": {"accelerator": "none", "dataSources": [],
                       "isInternetEnabled": False, "language": "python",
                       "sourceType": "notebook", "isGpuEnabled": False},
        },
        "nbformat": 4, "nbformat_minor": 4,
        "cells": [{"cell_type": "code", "source": cell, "metadata": {},
                   "outputs": [], "execution_count": None}],
    }
    args.out.write_text(json.dumps(nb))
    print(f"{args.out}: one cell, {len(cell.splitlines())} lines, "
          f"from {args.source}")


if __name__ == "__main__":
    main()
