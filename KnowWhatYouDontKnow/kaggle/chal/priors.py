"""Label priors: the sliding-pair construction, the container, the file format.

A *location* of the challenge is one label prior. ``base_priors.txt`` holds the
8 priors locations 0..7 start from -- :func:`pair_priors` with ``tau = 0.35``.
``make_data.py`` moves them as little as possible so that every training image
can be placed (``chal.locations``), adds the remainder location, and writes the
resulting 9 location priors in the same file format.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

TAU_DEFAULT = 0.35
# Two priors closer than this in total variation count as the same prior.
TV_TOL = 1e-2


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum())


def pair_priors(Y: int, tau: float = TAU_DEFAULT) -> np.ndarray:
    """``(Y, Y)``: prior ``i`` puts ``tau`` on classes ``i`` and ``i + 1 (mod Y)``
    and spreads ``1 - 2 tau`` uniformly over the other ``Y - 2``.

    Every element is the same distance from its neighbours by construction.
    """
    assert Y >= 3, (
        f"a pair prior needs a third class to hold the remaining mass, Y = {Y}")
    assert 0.0 < tau < 0.5, (
        f"tau must lie in (0, 1/2), got {tau:g}: tau = 0 puts zero mass on the "
        f"pair and tau = 1/2 puts zero on every other class, and log theta is "
        f"-inf either way")
    theta = np.full((Y, Y), (1.0 - 2.0 * tau) / (Y - 2))
    for i in range(Y):
        theta[i, [i, (i + 1) % Y]] = tau
    return theta


@dataclass
class PriorSet:
    """A set of label priors, a label per element, and a reference prior."""

    theta: np.ndarray            # (C, Y), rows sum to 1, all entries > 0
    labels: list[str]
    train_prior: np.ndarray      # (Y,) the prior the TV column is measured from

    @property
    def C(self) -> int:
        return self.theta.shape[0]

    @property
    def Y(self) -> int:
        return self.theta.shape[1]

    @property
    def tv_to_train(self) -> np.ndarray:
        return np.array([total_variation(t, self.train_prior) for t in self.theta])

    def pairwise_tv(self) -> np.ndarray:
        C = self.C
        M = np.zeros((C, C))
        for a in range(C):
            for b in range(a + 1, C):
                M[a, b] = M[b, a] = total_variation(self.theta[a], self.theta[b])
        return M


# --- the priors file ------------------------------------------------------

_COLUMNS = "# index  label  TV(theta, theta_tr)  theta_1 ... theta_Y"


def write_prior_set(ps: PriorSet, path: Path, header: list[str]) -> None:
    """One tab-separated line per prior, after ``header`` as ``#`` comments."""
    tv = ps.tv_to_train
    off = ps.pairwise_tv()[np.triu_indices(ps.C, k=1)]
    lines = [f"# {h}" for h in header] + [
        f"# min pairwise TV = {off.min():.4f} (guard: >= {TV_TOL:g})",
        f"# TV to theta_tr in [{tv.min():.4f}, {tv.max():.4f}]",
        _COLUMNS,
    ]
    for i, (t, lab) in enumerate(zip(ps.theta, ps.labels)):
        vec = " ".join(f"{v:.10g}" for v in t)
        lines.append(f"{i}\t{lab}\t{tv[i]:.6f}\t{vec}")
    Path(path).write_text("\n".join(lines) + "\n")


def read_prior_set(path: Path, train_prior: np.ndarray) -> PriorSet:
    """Read a priors file; its TV column is recomputed from ``train_prior``."""
    thetas, labels = [], []
    for line in Path(path).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        _, label, _tv, vec = line.split("\t")
        thetas.append(np.fromstring(vec, sep=" "))
        labels.append(label)
    theta = np.stack(thetas)
    assert np.allclose(theta.sum(axis=1), 1.0, atol=1e-6), \
        f"{path}: prior rows do not sum to 1"
    return PriorSet(theta, labels, np.asarray(train_prior, float))
