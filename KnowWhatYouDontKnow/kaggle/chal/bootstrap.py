"""Bootstrap intervals for Reg@c, with the **batch** as the resampling unit.

Rows inside one batch share ``theta_*`` and the same adaptation evidence, so
they are strongly dependent; resampling rows would understate every interval
badly. Precision therefore tracks ``N(m)``, the number of batches, not ``B_m``.

Two things make this cheap enough to run on 42 000 rows per usage:

* **The sort happens once.** A replicate does not reorder anything -- it only
  changes each row's *multiplicity*, so the accepted set is the shortest prefix
  of the fixed confidence order whose multiplicities sum to ``k``, found with a
  cumulative sum.
* **Replicates are shared across submissions.** :func:`paired_draws` resamples
  the batches once per replicate and scores every submission on that same
  resample, which is what makes the *difference* between two submissions
  measurable. Comparing two independently-bootstrapped marginal intervals is
  the wrong test: two rejectors built on the same base predictor differ only in
  their ranking, and their per-replicate difference is far less variable than
  either one alone, so overlapping marginal bands say nothing.
"""

from __future__ import annotations

import numpy as np

# Replicates per vectorised chunk; caps peak memory at roughly
# CHUNK * B * 8 bytes per array, and B reaches 20 000 at m = 100.
_CHUNK = 64


def regret_at_coverage(excess: np.ndarray, conf: np.ndarray,
                       rank_id: np.ndarray, coverage: float) -> float:
    """Mean ``excess`` over the most confident ``ceil(coverage * B)`` rows.

    Descending confidence, ties broken by ascending ``rank_id`` -- the same
    rule ``chal.metric.score`` applies, so this must agree with it exactly.
    """
    k = max(1, int(np.ceil(coverage * len(excess))))
    order = np.lexsort((rank_id, -conf))
    return float(excess[order[:k]].mean())


def _prefix_mean(w: np.ndarray, exc_sorted: np.ndarray, k: int) -> np.ndarray:
    """Mean excess over the first ``k`` accepted rows, per replicate.

    ``w`` is ``(R, B)`` multiplicities in the fixed confidence order. The row at
    which the budget is reached is counted only for the part that fits.
    """
    R = len(w)
    cum = np.cumsum(w, axis=1)
    acc = np.cumsum(w * exc_sorted[None, :], axis=1)
    j = np.argmax(cum >= k, axis=1)
    rows = np.arange(R)
    prev = np.maximum(j - 1, 0)
    before = np.where(j > 0, acc[rows, prev], 0.0)
    before_n = np.where(j > 0, cum[rows, prev], 0)
    return (before + (k - before_n) * exc_sorted[j]) / k


def paired_draws(excess_per_sub: list[np.ndarray], conf_per_sub: list[np.ndarray],
                 rank_id: np.ndarray, batch_of_row: np.ndarray, coverage: float,
                 reps: int, rng: np.random.Generator) -> np.ndarray:
    """``(n_subs, reps)`` bootstrap draws of ``Reg@c`` on **shared** resamples.

    Every submission is scored on the same resampled batches in every
    replicate, so differences between rows of the result are paired.
    """
    n_subs = len(excess_per_sub)
    n_batches = int(batch_of_row.max()) + 1
    total = len(rank_id)
    k = max(1, int(np.ceil(coverage * total)))

    # One fixed confidence order per submission.
    orders = [np.lexsort((rank_id, -c)) for c in conf_per_sub]
    exc_sorted = [excess_per_sub[s][orders[s]] for s in range(n_subs)]
    batch_sorted = [batch_of_row[orders[s]] for s in range(n_subs)]

    out = np.empty((n_subs, reps))
    done = 0
    while done < reps:
        R = min(_CHUNK, reps - done)
        draw = rng.integers(0, n_batches, size=(R, n_batches))
        counts = np.zeros((R, n_batches), dtype=np.int64)
        np.add.at(counts, (np.repeat(np.arange(R), n_batches), draw.ravel()), 1)
        for s in range(n_subs):
            out[s, done:done + R] = _prefix_mean(
                counts[:, batch_sorted[s]], exc_sorted[s], k)
        done += R
    return out


def bootstrap_regret(excess: np.ndarray, conf: np.ndarray, rank_id: np.ndarray,
                     batch_of_row: np.ndarray, coverage: float, reps: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Single-submission convenience wrapper around :func:`paired_draws`."""
    return paired_draws([excess], [conf], rank_id, batch_of_row, coverage,
                        reps, rng)[0]
