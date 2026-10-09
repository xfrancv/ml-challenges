"""Exact Bayesian label-prior adaptation over a finite set of priors ``Theta``.

A batch ``D`` is ``m`` images from one location, i.e. under one unknown prior
``theta`` out of the ``C`` location priors. With ``p_tr(y | x)`` the posterior
under the base prior ``p_tr(y)``:

    w(x, theta)        = sum_y (theta_y / p_tr(y)) p_tr(y | x)
    p_te(y | x, theta) = (theta_y / p_tr(y)) p_tr(y | x) / w(x, theta)
    p(theta | D)       ~ p(theta) prod_{j=1..m} w(x_j, theta)
    p(y | x_i, D)      = sum_theta p(theta | D) p_te(y | x_i, theta)

and from these, for every row,

    H(x, D)  = argmax_y p(y | x, D)                    the Bayes prediction
    T(x, D)  = 1 - max_y p(y | x, D)                   total uncertainty
    h(x, theta) = argmax_y p_te(y | x, theta)          the plug-in prediction
    A(x, D)  = E_{theta | D} [1 - max_y p_te(y | x, theta)]    aleatoric
    E(x, D)  = T - A  >= 0                             epistemic

``Theta`` is finite, so every integral is a sum over ``C`` terms, and all of it
runs in log space. The posterior over ``theta`` conditions on the **whole
batch**, the query row included; the MAP plug-in uses the same posterior.

Shapes: ``n`` rows, ``C`` priors, ``Y`` classes. Rows must be grouped by batch
and each batch contiguous -- ``batch_starts`` indexes the first row of each.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

# Rows per chunk of the (n, C, Y) intermediates; caps peak memory at roughly
# CHUNK * C * Y * 8 bytes per array.
_CHUNK = 65536


def log_weights(log_post: np.ndarray, log_train_prior: np.ndarray,
                log_theta: np.ndarray) -> np.ndarray:
    """``r[i, c] = log w(x_i, theta_c)``, in log space.

    ``w(x, theta) = p_te(x | theta) / p_tr(x) = sum_y (theta_y / p_tr(y)) p_tr(y | x)``
    -- the ratio in which the intractable ``p_tr(x)`` has already cancelled.
    """
    ratio = (log_theta - log_train_prior)[None, :, :]          # (1, C, Y)
    out = np.empty((len(log_post), log_theta.shape[0]))
    for a in range(0, len(log_post), _CHUNK):
        b = a + _CHUNK
        out[a:b] = logsumexp(log_post[a:b, None, :] + ratio, axis=2)
    return out


def batch_starts_from_ids(batch_id: np.ndarray) -> np.ndarray:
    """First-row index of each run in a row-sorted ``batch_id`` array."""
    assert len(batch_id) > 0
    change = np.flatnonzero(np.diff(batch_id)) + 1
    starts = np.concatenate([[0], change])
    assert np.all(np.diff(batch_id[starts]) > 0), (
        "rows must be sorted by batch id, each batch contiguous")
    return starts


@dataclass
class BatchInference:
    """Every per-row quantity of the module docstring, over all batches."""

    bayes_pred: np.ndarray     # (n,)  H(x, D)
    total: np.ndarray          # (n,)  T(x, D)
    aleatoric: np.ndarray      # (n,)  A(x, D)
    epistemic: np.ndarray      # (n,)  E(x, D) = T - A
    map_pred: np.ndarray       # (n,)  h(x, theta_map)
    map_unc: np.ndarray        # (n,)  1 - p_te(h | x, theta_map)
    map_index: np.ndarray      # (n,)  argmax_c p(theta_c | D), lowest c on ties
    pth: np.ndarray            # (n_batches, C) p(theta | D)


def infer(log_post: np.ndarray, batch_starts: np.ndarray,
          log_train_prior: np.ndarray, log_theta: np.ndarray,
          log_p_theta: np.ndarray) -> BatchInference:
    """Run the exact inference for every row of every batch.

    Cost is ``O(n C Y)``; nothing is per-batch except one ``reduceat``.
    """
    n, Y = log_post.shape
    C = log_theta.shape[0]
    assert log_theta.shape == (C, Y) and log_train_prior.shape == (Y,)

    r = log_weights(log_post, log_train_prior, log_theta)       # (n, C)

    # log p(theta) + log p_te(D | theta), up to a theta-free constant.
    S = np.add.reduceat(r, batch_starts, axis=0)                # (n_batches, C)
    g = log_p_theta[None, :] + S
    pth = np.exp(g - logsumexp(g, axis=1, keepdims=True))       # (n_batches, C)
    log_pth = np.log(pth)

    # Broadcast the per-batch posterior back over rows.
    row_batch = np.repeat(np.arange(len(batch_starts)),
                          np.diff(np.append(batch_starts, n)))
    map_index_b = g.argmax(axis=1)                              # lowest c on ties
    map_index = map_index_b[row_batch]

    ratio = (log_theta - log_train_prior)[None, :, :]
    bayes_pred = np.empty(n, dtype=np.int64)
    total = np.empty(n)
    aleatoric = np.empty(n)
    map_pred = np.empty(n, dtype=np.int64)
    map_unc = np.empty(n)

    for a in range(0, n, _CHUNK):
        b = min(a + _CHUNK, n)
        sl = slice(a, b)
        rows = np.arange(b - a)

        # log p_te(y | x_i, theta_c) = log_post + log(theta/p_tr) - r
        log_plug = log_post[sl, None, :] + ratio - r[sl, :, None]   # (k, C, Y)

        # --- label posterior and total uncertainty -------------------------
        log_lab = logsumexp(log_pth[row_batch[sl], :, None] + log_plug, axis=1)
        log_lab -= logsumexp(log_lab, axis=1, keepdims=True)
        label_post = np.exp(log_lab)
        pred = label_post.argmax(axis=1)                # ties -> lowest label
        bayes_pred[sl] = pred
        total[sl] = 1.0 - label_post[rows, pred]

        # --- per-theta plugin predictor and the aleatoric term -------------
        plugin_pred = log_plug.argmax(axis=2)                       # (k, C)
        plugin_max = np.exp(np.take_along_axis(
            log_plug, plugin_pred[:, :, None], axis=2)[:, :, 0])    # (k, C)
        plugin_unc = 1.0 - plugin_max
        # A = E_{theta ~ p(theta | D)} [1 - p_te(h(x, theta) | x, theta)]
        aleatoric[sl] = np.einsum("kc,kc->k", pth[row_batch[sl]], plugin_unc)

        # --- the MAP plugin -------------------------------------------------
        mi = map_index[sl]
        map_pred[sl] = plugin_pred[rows, mi]
        map_unc[sl] = plugin_unc[rows, mi]

    epistemic = total - aleatoric
    # E >= 0 because h(x, theta) minimises the per-theta conditional risk, so
    # A <= T(x, D, y_hat) for every y_hat. Only float noise may breach it.
    assert epistemic.min() > -1e-9, (
        f"epistemic uncertainty went negative ({epistemic.min():.3e}); that is "
        f"an arithmetic bug, not a modelling outcome")
    np.clip(epistemic, 0.0, None, out=epistemic)

    return BatchInference(bayes_pred=bayes_pred, total=total,
                          aleatoric=aleatoric, epistemic=epistemic,
                          map_pred=map_pred, map_unc=map_unc,
                          map_index=map_index, pth=pth)


def plugin_for_prior(log_post: np.ndarray, log_train_prior: np.ndarray,
                     theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(h(x, theta), 1 - max_y p_te(y | x, theta))`` for one prior per row.

    ``theta`` is either a single ``(Y,)`` prior or
    an ``(n, Y)`` array giving each row its own, which is how the true-prior
    reference predictor is evaluated (each row uses its batch's ``theta_*``).
    """
    theta = np.asarray(theta, float)
    if theta.ndim == 1:
        theta = np.broadcast_to(theta, log_post.shape)
    assert theta.shape == log_post.shape
    log_num = log_post + np.log(theta) - log_train_prior     # (n, Y)
    log_plug = log_num - logsumexp(log_num, axis=1, keepdims=True)
    pred = log_plug.argmax(axis=1)                           # ties -> lowest y
    unc = 1.0 - np.exp(log_plug[np.arange(len(pred)), pred])
    return pred, unc
