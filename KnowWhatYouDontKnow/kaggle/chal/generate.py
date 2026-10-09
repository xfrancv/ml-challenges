"""The label model's side of generation: labels, the reference, the checks.

``make_data.py`` is the only caller. The functions live here so that
``selftest.py`` can check them on small synthetic problems.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .inference import plugin_for_prior
from .splits import USAGES

# Development ids start here, so they cannot be confused with test ids.
DEV_ID_OFFSET = 1_000_000


def tempered(log_q: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """``q(y | x)^(1 / T)``, renormalised, as float64 probabilities.

    ``T = 1`` is the calibrated label model itself. The result *is* the label
    model from then on -- labels are drawn from it and the reference predictor
    uses it -- so ``T`` only sets how noisy the labels are (the Bayes error).
    """
    assert temperature > 0
    z = np.asarray(log_q, dtype=np.float64) / temperature
    return np.exp(z - logsumexp(z, axis=1, keepdims=True))


def draw_labels(q: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One label per row, ``y ~ q(y | x)``."""
    cum = np.cumsum(q, axis=1)
    u = rng.random(len(q))
    return np.minimum((u[:, None] >= cum).sum(axis=1), q.shape[1] - 1)


def reference(q_rows: np.ndarray, pi_bar: np.ndarray,
              theta_rows: np.ndarray) -> np.ndarray:
    """``argmax_y q(y | x) pi_l(y) / pibar_P(y)``: the Bayes prediction given
    the location, for rows of one pool ``P``."""
    pred, _unc = plugin_for_prior(np.log(q_rows), np.log(pi_bar), theta_rows)
    return pred


def debiased_posterior(q_rows: np.ndarray, pi_bar_pool: np.ndarray,
                       pi_bar_split: np.ndarray) -> np.ndarray:
    """``q(y | x) pibar_S(y) / pibar_P(y)``, renormalised -- for the organiser files.

    The organiser's tools (``chal/predictors.py``) take one posterior per row
    and one base prior per split. The test split has three pools, each with its
    own ``pibar_P``; rescaling every row to the split's common ``pibar_S`` makes
    the plug-in rule under ``pibar_S`` reproduce :func:`reference` exactly:

        q pibar_S / pibar_P * pi_l / pibar_S  =  q pi_l / pibar_P.
    """
    lp = np.log(q_rows) + np.log(pi_bar_split) - np.log(pi_bar_pool)
    return np.exp(lp - logsumexp(lp, axis=1, keepdims=True))


def row_checks(id_test, slot, m_of_row, row_id, usage_of_row, pool_image,
               grid) -> dict:
    """The assertions on the test rows, run at generation time."""
    order = np.lexsort((slot, id_test))
    s, b, m = slot[order], id_test[order], m_of_row[order]
    starts = np.concatenate([[0], np.flatnonzero(np.diff(b)) + 1])
    ends = np.concatenate([starts[1:], [len(b)]])
    assert np.all(s[ends - 1] + 1 == m[starts]), "max(slot) + 1 != m somewhere"
    assert np.all(ends - starts == m[starts]), "a batch is missing rows"
    assert np.array_equal(s, np.concatenate(
        [np.arange(k) for k in m[starts]])), "slots are not 0..m-1"
    assert len(np.unique(row_id)) == len(row_id), "row_id is not unique"

    # The usage pools are disjoint at the image level.
    pools = [set(np.unique(pool_image[usage_of_row == i]).tolist())
             for i in range(len(USAGES))]
    for i in range(len(USAGES)):
        for j in range(i + 1, len(USAGES)):
            assert not (pools[i] & pools[j]), (
                f"pools {USAGES[i]} and {USAGES[j]} share images")
    usages_seen = sorted(np.array(USAGES, dtype=object)[
        np.unique(usage_of_row)].tolist())
    assert usages_seen == sorted(USAGES), f"usages present: {usages_seen}"
    assert "Ignored" in USAGES, "Kaggle's literal is 'Ignored', not 'Ignore'"
    sizes = sorted(set(int(v) for v in np.unique(m_of_row)))
    assert sizes == sorted(grid), f"sizes present {sizes} != grid {sorted(grid)}"
    return {"slots_are_0_to_m_minus_1": True, "row_id_unique": True,
            "usage_pools_disjoint": True, "all_grid_sizes_present": True}
