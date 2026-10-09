"""Assigning the training data to locations.

Competitors are told that the training data come from ``L`` locations with
different class frequencies. Underneath, each location is one label prior:
locations ``0..K-1`` start from the ``K`` priors of ``base_priors.txt``, and
location ``K`` takes **every remaining training image**, so no training data
are discarded.

**Why the priors cannot be used as they are.** Each sliding-pair prior puts
0.35 on two classes and 0.05 on the other six, so summed over the ``K = 8``
locations every class has total probability exactly 1.0. A location of size
``n`` needs ``0.35 n`` or ``0.05 n`` images of every class, which caps every
location at the size of the rarest class and leaves the remainder location with
probability zero on it. The priors are therefore moved as little as possible.

**The linear program.** Let ``m[l, c]`` be the number of images of class ``c``
at location ``l < K`` and ``n_l = sum_c m[l, c]``. For a fixed ``delta``
every constraint is linear in ``m``:

* the remainder location is what is left: ``r_c = D_c - sum_l m[l, c]``;
* floor on every prior, the remainder's included: ``m[l, c] >= eps n_l`` and
  ``r_c >= eps sum_c r_c``;
* minimum location size: ``n_l >= n_min``;
* limited change: ``TV(m[l] / n_l, P[l]) <= delta``, written as
  ``sum_c |m[l, c] - n_l P[l, c]| <= 2 delta n_l``.

``delta`` is bisected down to the smallest feasible value, and at that
``delta`` the smallest of the ``n_l`` is maximised. The counts are then rounded
down; rounding only ever moves images into the remainder location.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

EPS_DEFAULT = 0.01
N_MIN_DEFAULT = 5000
DELTA_TOL = 1e-4


@dataclass
class LocationPlan:
    """How many images of each class every location receives."""

    counts: np.ndarray        # (K + 1, Y) int; the last row is the remainder
    base_priors: np.ndarray   # (K, Y) the priors locations 0..K-1 started from
    delta: float              # the smallest feasible max TV change (before rounding)
    eps: float
    n_min: int

    @property
    def L(self) -> int:
        return self.counts.shape[0]

    @property
    def sizes(self) -> np.ndarray:
        return self.counts.sum(axis=1)

    @property
    def priors(self) -> np.ndarray:
        """``(L, Y)``: each location's class frequencies -- its label prior."""
        return self.counts / self.sizes[:, None]

    @property
    def tv_change(self) -> np.ndarray:
        """``(K,)``: TV between each moved prior and the prior it started from."""
        return 0.5 * np.abs(self.priors[:-1] - self.base_priors).sum(axis=1)


def _solve(D: np.ndarray, P: np.ndarray, eps: float, n_min: float,
           delta: float):
    """One LP at a fixed ``delta``: maximise ``min_l n_l``.

    Variables, in order: ``m`` (K*Y), the TV slacks ``a`` (K*Y), and ``t``.
    """
    K, Y = P.shape
    T = float(D.sum())
    nm = K * Y
    nv = 2 * nm + 1

    def im(l, c):
        return l * Y + c

    def ia(l, c):
        return nm + l * Y + c

    rows, b = [], []

    def add(coef: dict, rhs: float) -> None:
        r = np.zeros(nv)
        for k, v in coef.items():
            r[k] += v
        rows.append(r)
        b.append(rhs)

    for l in range(K):
        n_l = {im(l, c): 1.0 for c in range(Y)}
        # t <= n_l and n_min <= n_l
        add({**{k: -1.0 for k in n_l}, nv - 1: 1.0}, 0.0)
        add({k: -1.0 for k in n_l}, -n_min)
        for c in range(Y):
            # eps n_l <= m[l, c]
            coef = {k: eps for k in n_l}
            coef[im(l, c)] -= 1.0
            add(coef, 0.0)
            # |m[l, c] - n_l P[l, c]| <= a[l, c], as two half-planes
            coef = {k: -P[l, c] for k in n_l}
            coef[im(l, c)] += 1.0
            add({**coef, ia(l, c): -1.0}, 0.0)
            add({**{k: -v for k, v in coef.items()}, ia(l, c): -1.0}, 0.0)
        # sum_c a[l, c] <= 2 delta n_l
        add({**{ia(l, c): 1.0 for c in range(Y)},
             **{k: -2.0 * delta for k in n_l}}, 0.0)

    for c in range(Y):
        # the remainder keeps eps of its own size:
        #   D_c - sum_l m[l, c] >= eps (T - sum_{l, k} m[l, k])
        coef = {im(l, k): -eps for l in range(K) for k in range(Y)}
        for l in range(K):
            coef[im(l, c)] += 1.0
        add(coef, D[c] - eps * T)

    obj = np.zeros(nv)
    obj[-1] = -1.0
    return linprog(obj, A_ub=np.array(rows), b_ub=np.array(b),
                   bounds=[(0, None)] * nv, method="highs")


def plan_locations(class_counts: np.ndarray, base_priors: np.ndarray,
                   eps: float = EPS_DEFAULT, n_min: int = N_MIN_DEFAULT,
                   tol: float = DELTA_TOL) -> LocationPlan:
    """The minimal change to ``base_priors`` that places every training image."""
    D = np.asarray(class_counts, dtype=float)
    P = np.asarray(base_priors, dtype=float)
    assert P.ndim == 2 and P.shape[1] == len(D)
    assert np.allclose(P.sum(axis=1), 1.0)
    share = D.min() / D.sum()
    assert eps < share, (
        f"eps = {eps:g} is not below the rarest class's share {share:.4f}; with "
        f"every image used, no set of priors can give each location that much "
        f"of it")

    K, Y = P.shape
    # Rounding down can take up to one image per class off a location, so the
    # LP is asked for Y more than n_min: the rounded sizes still honour n_min.
    n_lp = n_min + Y
    if _solve(D, P, eps, n_lp, 1.0).status != 0:
        raise ValueError(
            f"no assignment exists even with the priors free to move: n_min = "
            f"{n_min} is too large for {len(P)} locations with eps = {eps:g}")
    if _solve(D, P, eps, n_lp, 0.0).status == 0:
        lo = hi = 0.0
    else:
        lo, hi = 0.0, 1.0
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            if _solve(D, P, eps, n_lp, mid).status == 0:
                hi = mid
            else:
                lo = mid
    res = _solve(D, P, eps, n_lp, hi)
    assert res.status == 0, res.message

    m = res.x[:K * Y].reshape(K, Y)
    M = np.floor(m + 1e-6).astype(np.int64)
    rest = D.astype(np.int64) - M.sum(axis=0)
    assert np.all(rest >= 0), "rounding over-allocated a class"
    plan = LocationPlan(counts=np.vstack([M, rest]), base_priors=P,
                        delta=float(hi), eps=eps, n_min=n_min)
    _check(plan, D)
    return plan


def _check(plan: LocationPlan, D: np.ndarray) -> None:
    assert np.array_equal(plan.counts.sum(axis=0), D.astype(np.int64)), \
        "the locations do not hold every training image exactly once"
    assert np.all(plan.counts > 0), (
        "a location has no image of some class, so its prior has a zero and "
        "log theta is -inf")
    # Rounding down moves at most one image per cell, so the floor can slip by
    # about 1 / n_l -- far below anything that matters, but not zero.
    slack = 2.0 / plan.sizes.min()
    assert plan.priors.min() >= plan.eps - slack, (
        f"a prior entry {plan.priors.min():.5f} fell below the floor "
        f"{plan.eps:g}")
    assert plan.sizes[:-1].min() >= plan.n_min, \
        "a location came out smaller than n_min"


def assign_locations(y: np.ndarray, plan: LocationPlan,
                     rng: np.random.Generator) -> np.ndarray:
    """``(n,)`` location of every training image, realising ``plan.counts``.

    Within a class, which images go where is uniformly random, so a location
    differs from another only in its class frequencies -- never in the look of
    its images.
    """
    y = np.asarray(y)
    loc = np.full(len(y), -1, dtype=np.int64)
    for c in range(plan.counts.shape[1]):
        idx = rng.permutation(np.flatnonzero(y == c))
        assert len(idx) == plan.counts[:, c].sum()
        bounds = np.concatenate([[0], np.cumsum(plan.counts[:, c])])
        for l in range(plan.L):
            loc[idx[bounds[l]:bounds[l + 1]]] = l
    assert loc.min() >= 0
    return loc
