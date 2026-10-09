"""BCTS calibration of the label model, and its two calibration numbers.

The label model is calibrated by **bias-corrected temperature scaling with a
per-class temperature** -- ``softmax(z_k / T_k + b_k)``, all ``2Y`` parameters
fit by LBFGS on validation NLL. Validation NLL and ECE (15 **equal-mass**
bins) are reported before and after.

The calibrated posterior is ``log_softmax(z * scale + bias)`` with
``scale = 1 / T``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

N_ECE_BINS = 15


def log_softmax_np(logits: np.ndarray) -> np.ndarray:
    """Row-wise log-softmax of an (n, Y) array, in float64."""
    z = logits - logits.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def fit_bcts(logits: torch.Tensor, y: torch.Tensor,
             max_iter: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """Bias-corrected temperature scaling with a **per-class** temperature.

    Fits ``softmax(z_k / T_k + b_k)`` by minimising validation NLL
    (Alexandari et al. 2020, generalised from their shared ``T`` to one ``T_k``
    per class -- the extra freedom is what makes the correction bias-aware in
    both directions of the logit scale).

    Optimised over ``log(1 / T_k)`` so every temperature stays positive.
    Softmax is invariant to a constant shift of the bias, so ``b`` is returned
    mean-centred, which fixes the otherwise-unidentified offset.

    Returns ``(T, b)``, both shape ``(Y,)``.
    """
    Y = logits.shape[1]
    log_inv_t = torch.zeros(Y, requires_grad=True)
    b = torch.zeros(Y, requires_grad=True)
    opt = torch.optim.LBFGS([log_inv_t, b], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits * log_inv_t.exp() + b, y)
        loss.backward()
        return loss

    opt.step(closure)
    T = torch.exp(-log_inv_t.detach()).numpy().astype(np.float64)
    bd = b.detach().numpy().astype(np.float64)
    return T, bd - bd.mean()


def nll(log_post: np.ndarray, y: np.ndarray) -> float:
    """Mean negative log-likelihood of the labels under ``log_post`` (n, Y)."""
    return float(-log_post[np.arange(len(y)), y].mean())


def ece_equal_mass(post: np.ndarray, y: np.ndarray,
                   n_bins: int = N_ECE_BINS) -> float:
    """Expected calibration error with **equal-mass** (equal-count) bins.

    Equal-mass rather than equal-width because the confidence histogram of a
    trained network piles up near 1.0, which leaves most equal-width bins
    almost empty and makes the estimate noisy.
    """
    conf = post.max(axis=1)
    correct = (post.argmax(axis=1) == y).astype(float)
    order = np.argsort(conf, kind="stable")
    ece = 0.0
    n = len(y)
    for chunk in np.array_split(order, min(n_bins, n)):
        if len(chunk) == 0:
            continue
        ece += (len(chunk) / n) * abs(correct[chunk].mean() - conf[chunk].mean())
    return float(ece)


def calibration_summary(logits: np.ndarray, y: np.ndarray,
                        scale: np.ndarray, bias: np.ndarray) -> dict:
    """NLL and ECE before (raw softmax) and after ``z * scale + bias``.

    ``scale`` is the per-class ``1 / T`` of :func:`fit_bcts`.
    """
    def stats(log_post):
        return nll(log_post, y), ece_equal_mass(np.exp(log_post), y)

    nll_pre, ece_pre = stats(log_softmax_np(logits))
    nll_post, ece_post = stats(log_softmax_np(logits * scale + bias))
    return {"nll_before": nll_pre, "ece_before": ece_pre,
            "nll_after": nll_post, "ece_after": ece_post,
            "n_ece_bins": N_ECE_BINS}
