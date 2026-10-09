"""The batch protocol: the grid, the batch counts, the sampler.

One *batch* is ``m`` rows from one location ``l``, drawn i.i.d. from the secret
location prior ``w``. Every row of every batch is drawn independently of all
others:

1. ``y ~ pi_l``, the location's label prior;
2. an image ``x`` from the pool ``P`` with probability ``q(y | x) / sum_P q(y | .)``,
   i.e. ``x ~ p_P(x | y)``.

No image carries a fixed label: the label of a row is the class drawn in step 1.
The same image may occur in several batches, or twice in one batch, each time
with its own independently drawn label, so linking copies of an image across
batches reveals nothing about their labels. With this procedure the posterior
of a row from location ``l`` is exactly

    p_l(y | x) ~ q(y | x) pi_l(y) / pibar_P(y),    pibar_P = mean_{x in P} q(. | x),

which is what the reference predictor maximises.

Every row is a submission row and every one is scored, so the pooled ranking at
size ``m`` has ``B_m = N(m) * m`` rows. ``N_MIN`` is the load-bearing constant:
rows inside one batch share the location and the same adaptation evidence, so
precision tracks the number of *batches*, not rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SIZE_GRID = (1, 2, 5, 10, 20, 50, 100)
N_MIN = 200
BATCH_SCALE = 2000          # N(m) = max(N_MIN, ceil(BATCH_SCALE / m))
# m_max must not exceed a tenth of the pool it is drawn from, or
# repeated images inside a batch stop being rare.
POOL_SIZE_RATIO = 10


def n_batches(m: int, n_min: int = N_MIN, scale: int = BATCH_SCALE) -> int:
    """``N(m) = max(N_min, ceil(scale / m))``."""
    assert m >= 1
    return int(max(n_min, -(-scale // m)))


def check_grid(pool_size: int, grid=SIZE_GRID) -> None:
    m_max = max(grid)
    assert m_max * POOL_SIZE_RATIO <= pool_size, (
        f"m_max = {m_max} exceeds |pool| / {POOL_SIZE_RATIO} = "
        f"{pool_size // POOL_SIZE_RATIO}; shrink the grid or enlarge the pool")


class PoolSampler:
    """Draws images of a given class from one pool: ``x ~ p_P(x | y)``.

    ``p_P(x | y) = q(y | x) / (|P| pibar_P(y))`` over the images of the pool,
    sampled by inverse CDF, one column per class.
    """

    def __init__(self, q_pool: np.ndarray):
        q = np.asarray(q_pool, dtype=np.float64)
        assert q.ndim == 2 and np.all(q >= 0)
        assert np.allclose(q.sum(axis=1), 1.0), "q(. | x) rows must sum to 1"
        self.n, self.Y = q.shape
        self.pi_bar = q.mean(axis=0)
        assert np.all(self.pi_bar > 0), "a class has zero mass in the pool"
        cdf = np.cumsum(q, axis=0) / q.sum(axis=0)
        cdf[-1] = 1.0
        self.cdf = cdf

    def images(self, cls: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """``(k,)`` pool indices, one ``x ~ p_P(x | cls[i])`` per entry."""
        cls = np.asarray(cls)
        u = rng.random(len(cls))
        idx = np.empty(len(cls), dtype=np.int64)
        for c in range(self.Y):
            sel = cls == c
            if sel.any():
                # side="right": index i is drawn when cdf[i-1] <= u < cdf[i],
                # i.e. with probability q(c | x_i) / sum_P q(c | .).
                idx[sel] = np.searchsorted(self.cdf[:, c], u[sel], side="right")
        return np.minimum(idx, self.n - 1)


@dataclass
class DrawnRows:
    """Row-level and batch-level arrays for one pool's worth of batches."""

    gen_batch: np.ndarray      # (n_rows,) generation index of the row's batch
    slot: np.ndarray           # (n_rows,) 0..m-1
    pool_row: np.ndarray       # (n_rows,) index into the pool
    label: np.ndarray          # (n_rows,) the class drawn for the row
    batch_m: np.ndarray        # (n_batches,)
    batch_location: np.ndarray  # (n_batches,)

    @property
    def n_rows(self) -> int:
        return len(self.slot)

    @property
    def n_batches(self) -> int:
        return len(self.batch_m)

    @property
    def m_of_row(self) -> np.ndarray:
        return self.batch_m[self.gen_batch]

    @property
    def location_of_row(self) -> np.ndarray:
        return self.batch_location[self.gen_batch]


def draw_rows(sampler: PoolSampler, theta: np.ndarray, w: np.ndarray,
              rng: np.random.Generator, grid=SIZE_GRID, n_min: int = N_MIN,
              scale: int = BATCH_SCALE) -> DrawnRows:
    """Every batch of one pool, flattened into rows, in grid order.

    ``theta`` is ``(L, Y)``, the label prior of every location; ``w`` is
    ``(L,)``, the location prior. Slots are ``0..m-1`` in draw order, which is
    uniformly random because the rows are i.i.d.
    """
    theta = np.asarray(theta, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    L, Y = theta.shape
    assert Y == sampler.Y and w.shape == (L,)
    assert np.all(w >= 0) and abs(w.sum() - 1.0) < 1e-9
    check_grid(sampler.n, grid)

    batch_m, batch_loc = [], []
    for m in grid:
        N = n_batches(m, n_min, scale)
        batch_m.append(np.full(N, m, dtype=np.int64))
        batch_loc.append(rng.choice(L, size=N, p=w))
    batch_m = np.concatenate(batch_m)
    batch_loc = np.concatenate(batch_loc)

    gen_batch = np.repeat(np.arange(len(batch_m)), batch_m)
    slot = np.concatenate([np.arange(m) for m in batch_m])
    loc_row = batch_loc[gen_batch]
    # y ~ pi_l by inverse CDF, row by row.
    cum = np.cumsum(theta, axis=1)
    u = rng.random(len(gen_batch))
    label = np.minimum((u[:, None] >= cum[loc_row]).sum(axis=1), Y - 1)
    pool_row = sampler.images(label, rng)
    return DrawnRows(gen_batch=gen_batch, slot=slot, pool_row=pool_row,
                     label=label.astype(np.int64), batch_m=batch_m,
                     batch_location=batch_loc)
