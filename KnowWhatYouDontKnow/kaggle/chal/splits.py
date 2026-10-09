"""The Kaggle ``Usage`` pools of the test data.

The test pool (the original validation + test splits) is partitioned into three
disjoint pools, one per Kaggle ``Usage`` value. Disjointness is what stops
labels probed off the public leaderboard from being worth anything on the
private split.
"""

from __future__ import annotations

import numpy as np

USAGES = ("Public", "Private", "Ignored")
# Kaggle's own literal is "Ignored". A typo here silently drops every row of
# that pool from scoring, so it is asserted at generation time.


def assign_pools(y: np.ndarray, seed: int) -> np.ndarray:
    """Class-stratified partition of the test pool into ``len(USAGES)`` pools.

    Stratified by the *original* class, so that the pools look alike; the
    original labels are used for nothing else downstream.
    """
    rng = np.random.default_rng(seed)
    pool = np.empty(len(y), dtype=np.int64)
    k = len(USAGES)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        # Round-robin over a shuffled list: sizes differ by at most one.
        pool[idx] = np.arange(len(idx)) % k
    return pool
