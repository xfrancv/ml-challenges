"""The competition metric: AvgRegAtCoverage.

This is the exact function the organisers run on Kaggle -- ``evaluate.py`` calls
it, so a score you compute offline on the development batches is the score the
leaderboard would give you for those rows.

Read ``score()``'s docstring for what the metric does and for two worked
examples you can check by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COVERAGE = 0.8
EXPECTED_SIZES = (1, 2, 5, 10, 20, 50, 100)
NUM_CLASSES = 8


class ParticipantVisibleError(Exception):
    """Raised with a message the competitor is allowed to see."""


def score(solution: pd.DataFrame, submission: pd.DataFrame,
          row_id_column_name: str, coverage: float = COVERAGE,
          expected_sizes: tuple = EXPECTED_SIZES,
          num_classes: int = NUM_CLASSES) -> float:
    """Average selective regret at 80 % coverage, lower is better.

    Each test batch is a set of images from one location. For
    every image the competitor submits a predicted label ``pred`` and a real
    number ``confidence``; higher confidence means the prediction is more
    likely to be kept. Within each batch size ``m`` all rows are pooled and
    ranked by descending confidence, the most confident 80 % are kept, and the
    metric is the mean of

        1{pred != label} - 1{pred_ref != label}

    over the kept rows, where ``pred_ref`` is the prediction of the best
    possible predictor that knows which location the batch came from. The
    reported score averages these seven per-``m`` numbers with equal weight.

    No submission can beat the reference on average, so the expected score is
    at least zero; a slightly negative score can only come from chance. Even
    the best submission scores above zero at small ``m``, because a small
    batch says little about its location. Ties in ``confidence`` are broken by
    ascending ``row_id``, so a constant-confidence submission is still scored
    deterministically.

    >>> row_id_column_name = "row_id"
    >>> solution = pd.DataFrame({
    ...     "row_id":   [0, 1, 2, 3],
    ...     "id_test":  [0, 1, 2, 2],
    ...     "m":        [1, 1, 2, 2],
    ...     "label":    [0, 1, 3, 4],
    ...     "pred_ref": [0, 1, 3, 4]})
    >>> submission = pd.DataFrame({
    ...     "row_id":     [0, 1, 2, 3],
    ...     "pred":       [1, 1, 3, 4],
    ...     "confidence": [0.1, 0.9, 0.5, 0.5]})
    >>> score(solution.copy(), submission.copy(), row_id_column_name,
    ...       expected_sizes=(1, 2))
    0.25

    At ``m = 1`` both rows are kept (``ceil(0.8 * 2) = 2``), one of the two
    predictions is wrong where the reference is right, so that group scores
    0.5; at ``m = 2`` both predictions match the reference and the group scores
    0.0. The mean is 0.25.

    Coverage really does bite once a group has enough rows, and a competitor is
    rewarded for putting low confidence on the rows they get wrong:

    >>> solution = pd.DataFrame({
    ...     "row_id":   [0, 1, 2, 3, 4],
    ...     "id_test":  [0, 0, 0, 0, 0],
    ...     "m":        [5, 5, 5, 5, 5],
    ...     "label":    [0, 1, 2, 3, 4],
    ...     "pred_ref": [0, 1, 2, 3, 4]})
    >>> submission = pd.DataFrame({
    ...     "row_id":     [0, 1, 2, 3, 4],
    ...     "pred":       [0, 1, 2, 3, 9],
    ...     "confidence": [0.9, 0.8, 0.7, 0.6, 0.1]})
    >>> score(solution.copy(), submission.copy(), row_id_column_name,
    ...       expected_sizes=(5,), num_classes=10)
    0.0

    The single wrong row is ranked last and falls outside the 80 % budget, so
    the kept rows carry no regret at all.
    """
    sol = solution.copy()
    sub = submission.copy()

    for col in ("m", "label", "pred_ref"):
        if col not in sol.columns:
            raise ParticipantVisibleError(
                f"the solution file is missing column '{col}'")
    if row_id_column_name not in sol.columns:
        raise ParticipantVisibleError(
            f"the solution file is missing column '{row_id_column_name}'")
    for col in (row_id_column_name, "pred", "confidence"):
        if col not in sub.columns:
            raise ParticipantVisibleError(
                f"your submission is missing the column '{col}'; it needs "
                f"'{row_id_column_name}', 'pred' and 'confidence'")

    # --- align on row_id ---------------------------------------------------
    # Kaggle aligns the two frames before calling us, but the ranking and the
    # tie-break both depend on the alignment being exact, so it is re-done and
    # asserted here rather than trusted.
    if sub[row_id_column_name].duplicated().any():
        raise ParticipantVisibleError(
            f"your submission contains duplicate '{row_id_column_name}' values")
    if len(sub) != len(sol):
        raise ParticipantVisibleError(
            f"your submission has {len(sub):,} rows but {len(sol):,} were "
            f"expected -- predict every row of test_batches.csv exactly once")
    sol = sol.sort_values(row_id_column_name, kind="stable").reset_index(drop=True)
    sub = sub.sort_values(row_id_column_name, kind="stable").reset_index(drop=True)
    if not sol[row_id_column_name].equals(sub[row_id_column_name]):
        raise ParticipantVisibleError(
            f"the '{row_id_column_name}' values in your submission do not match "
            f"the expected ones")

    # --- validate the two submitted columns --------------------------------
    pred = pd.to_numeric(sub["pred"], errors="coerce").to_numpy(dtype=float)
    conf = pd.to_numeric(sub["confidence"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(pred).all():
        raise ParticipantVisibleError(
            "column 'pred' must be a whole number for every row (no blanks, "
            "no NaN, no text)")
    if not np.all(pred == np.round(pred)):
        raise ParticipantVisibleError("column 'pred' must contain whole numbers")
    pred = pred.astype(np.int64)
    if pred.min() < 0 or pred.max() >= num_classes:
        raise ParticipantVisibleError(
            f"column 'pred' must be a label in 0..{num_classes - 1}, but values "
            f"from {pred.min()} to {pred.max()} were submitted")
    if not np.isfinite(conf).all():
        raise ParticipantVisibleError(
            "column 'confidence' must be a finite real number for every row "
            "(no blanks, no NaN, no infinities)")

    label = sol["label"].to_numpy(dtype=np.int64)
    pred_ref = sol["pred_ref"].to_numpy(dtype=np.int64)
    m = sol["m"].to_numpy(dtype=np.int64)
    # row_id is the documented tie-break. Rank it by position in the sorted
    # frame so the rule works whatever dtype the column has.
    rank_id = np.arange(len(sol), dtype=np.int64)

    excess = (pred != label).astype(np.float64) - (pred_ref != label).astype(np.float64)

    sizes = tuple(sorted(set(int(v) for v in np.unique(m))))
    if expected_sizes is not None and sizes != tuple(sorted(expected_sizes)):
        raise ParticipantVisibleError(
            f"the solution file holds batch sizes {sizes}, expected "
            f"{tuple(sorted(expected_sizes))}")

    per_size = [
        _regret_at_coverage(excess[m == size], conf[m == size],
                            rank_id[m == size], coverage)
        for size in sizes
    ]
    result = float(np.mean(per_size))
    if not np.isfinite(result):
        raise ParticipantVisibleError("the metric did not evaluate to a finite number")
    return result


def _regret_at_coverage(excess: np.ndarray, conf: np.ndarray,
                        rank_id: np.ndarray, coverage: float) -> float:
    """Mean ``excess`` over the most confident ``ceil(coverage * B)`` rows.

    Sorted by descending confidence, ties broken by ascending ``rank_id``.
    ``np.lexsort`` orders by its last key first, so the keys are given as
    ``(-conf, rank_id)`` reversed: primary ``-conf``, secondary ``rank_id``.
    """
    B = len(excess)
    if B == 0:
        raise ParticipantVisibleError("a batch size group came out empty")
    k = max(1, int(np.ceil(coverage * B)))
    order = np.lexsort((rank_id, -conf))
    return float(excess[order[:k]].mean())
