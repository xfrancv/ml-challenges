"""The secret location prior ``w`` and its EM estimate.

The location of every development and test batch is drawn i.i.d. from ``w``,
the same for every batch size and for the development, public and private
batches. Competitors are told only that; the intended solution estimates ``w``
by EM on the development batches.

**On the development batches EM needs no model.** A row of location ``l`` is
``(x, y)`` with ``p(x, y | l) = pi_l(y) p(x | y)``, and ``p(x | y)`` does not
depend on ``l``. So the likelihood of a batch's *labels* already carries all of
its information about ``l``:

    p(batch | l) ~ prod_i pi_l(y_i),

and EM over the mixture weights is exact maximum likelihood with the ``pi_l``
known -- they are the class frequencies of each location in ``train.csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from .priors import total_variation

# Neither uniform nor proportional to the training location sizes (location 8
# holds most of the training data), since both are the first guesses
# competitors will try. TV 0.42 from uniform, every entry >= 0.03. A milder
# w (TV 0.28) made the gain from estimating it by EM too small to separate from
# noise on the private split; this one separates on both (run_baselines.sh).
W_DEFAULT = (0.30, 0.03, 0.20, 0.03, 0.05, 0.25, 0.03, 0.08, 0.03)
W_MIN = 0.03


def check_location_prior(w: np.ndarray, location_sizes: np.ndarray) -> dict:
    """The guards on ``w``, returned as numbers for the report."""
    w = np.asarray(w, dtype=np.float64)
    L = len(w)
    assert abs(w.sum() - 1.0) < 1e-9, f"w sums to {w.sum()}"
    assert w.min() >= W_MIN, (
        f"w has an entry {w.min():.3f} < {W_MIN}; that location would barely "
        f"occur in the development data")
    sizes = np.asarray(location_sizes, dtype=np.float64)
    tv_uniform = total_variation(w, np.full(L, 1.0 / L))
    tv_sizes = total_variation(w, sizes / sizes.sum())
    assert tv_uniform >= 0.1, f"w is only TV {tv_uniform:.3f} from uniform"
    assert tv_sizes >= 0.1, (
        f"w is only TV {tv_sizes:.3f} from the training location shares")
    return {"tv_to_uniform": tv_uniform, "tv_to_training_shares": tv_sizes}


def label_loglik(label: np.ndarray, batch_starts: np.ndarray,
                 log_theta: np.ndarray) -> np.ndarray:
    """``(n_batches, L)``: ``sum_i log pi_l(y_i)`` over the rows of each batch.

    Rows must be grouped by batch, each batch contiguous from ``batch_starts``.
    """
    per_row = log_theta[:, np.asarray(label)].T                 # (n, L)
    return np.add.reduceat(per_row, batch_starts, axis=0)


def em_location_prior(loglik: np.ndarray, max_iter: int = 10000,
                      tol: float = 1e-12) -> tuple[np.ndarray, int]:
    """``(w_hat, iterations)``: the maximum-likelihood mixture weights.

    ``loglik[b, l] = log p(batch b | location l)`` up to a per-batch constant.
    Starts from uniform; the log-likelihood is concave in ``w``, so the
    fixed point is the global maximum.
    """
    B, L = loglik.shape
    w = np.full(L, 1.0 / L)
    for it in range(1, max_iter + 1):
        g = np.log(np.maximum(w, 1e-300))[None, :] + loglik
        resp = np.exp(g - logsumexp(g, axis=1, keepdims=True))
        w_new = resp.mean(axis=0)
        if np.abs(w_new - w).max() < tol:
            return w_new, it
        w = w_new
    return w, max_iter


def write_location_prior(w: np.ndarray, path: Path) -> None:
    pd.DataFrame({"location": np.arange(len(w)), "w": np.asarray(w)}).to_csv(
        path, index=False)


def read_location_prior(path: Path) -> np.ndarray:
    df = pd.read_csv(path).sort_values("location")
    w = df["w"].to_numpy(dtype=np.float64)
    assert np.array_equal(df["location"].to_numpy(), np.arange(len(w)))
    assert abs(w.sum() - 1.0) < 1e-6, f"{path}: w sums to {w.sum()}"
    return w / w.sum()
