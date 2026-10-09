"""The reject-option baselines, built from an organiser directory.

``make_package.py`` writes ``organiser/test/`` and ``organiser/dev/``:
``predictions.csv`` (the TRUE label model, one row per batch row),
``test_batches.csv``, ``test_priors.csv`` (the 9 location priors, exactly what
a competitor reads off ``train.csv``) and ``train_prior.csv`` (the base prior
the posterior is under). The predictors here are therefore what a competitor
with a perfect model of ``q(y | x)`` would achieve. ``true_plugin`` needs the
location of every batch and reads ``batch_meta.csv``; it is the metric's own
reference, not a competitor.

**The location prior** ``p(theta)`` is uniform unless the caller passes one: the
secret ``w`` (``location_prior.csv``), or an estimate of it, such as the EM
estimate of ``estimate_location_prior.py``.

A reject-option predictor is a pair (base predictor, uncertainty score). The
submission format carries a **confidence**, so every score ``u`` is emitted as
``-u`` or as a rank; only the induced ranking is ever read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .inference import batch_starts_from_ids, infer, plugin_for_prior

# Name -> one-line description; notation of chal/inference.py.
PREDICTORS = {
    "base": "non-adapted base predictor, argmax_y q(y | x)",
    "map_plugin": "MAP plugin h(x, theta_map)",
    "bayes_total": "Bayesian base predictor, total uncertainty T",
    "bayes_epistemic": "Bayesian base predictor, epistemic uncertainty E",
    "true_plugin": "true-prior plugin -- oracle, zero regret",
    "bayes_aleatoric": "Bayesian base predictor, aleatoric uncertainty A",
}


def rank_confidence(*keys: np.ndarray) -> np.ndarray:
    """A confidence column reproducing a lexicographic **ascending** ordering.

    The metric reads only the ranking induced by ``confidence``, so a rejector
    whose score is a tuple -- epistemic uncertainty with ties broken by total
    uncertainty -- is expressed exactly by ranking the tuple and negating.
    Encoding the tie-break as ``-(E + eps * T)`` instead would need an ``eps``
    small enough not to reorder distinct ``E`` and large enough to survive
    float; a rank needs no such constant.

    Exact ties are the common case here, not a corner case: ``E(x, D)`` is
    exactly zero whenever every prior in ``Theta`` votes for the same label, so
    the tie-break governs a large part of the ranking.

    ``np.lexsort`` takes its *last* key as primary, so the keys are reversed.
    """
    order = np.lexsort(tuple(reversed(keys)))
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    return -rank.astype(np.float64)


class Problem:
    """One organiser directory, loaded and arranged for inference."""

    def __init__(self, organiser_dir: Path,
                 location_prior: np.ndarray | None = None):
        organiser_dir = Path(organiser_dir)
        self.dir = organiser_dir
        batches = pd.read_csv(organiser_dir / "test_batches.csv")
        preds = pd.read_csv(organiser_dir / "predictions.csv")
        train_prior = pd.read_csv(organiser_dir / "train_prior.csv").to_numpy()[0]
        priors = pd.read_csv(organiser_dir / "test_priors.csv")

        Y = len(train_prior)
        post = preds[[f"p{y}" for y in range(Y)]].to_numpy(dtype=np.float64)
        post = np.clip(post, 1e-300, None)
        # Renormalise so each row is exactly a distribution. This is a per-row
        # positive rescaling, which cancels everywhere downstream: it shifts
        # log_post by a row constant, hence r by a theta-independent constant,
        # which drops out of the softmax over theta and out of every plugin
        # posterior.
        post /= post.sum(axis=1, keepdims=True)
        log_post_all = np.log(post)

        # Inference needs rows grouped by batch, each batch contiguous.
        batches = batches.sort_values(["id_test", "slot"], kind="stable"
                                      ).reset_index(drop=True)
        # Positional lookup rather than a per-row dict.
        pos = pd.Index(preds["row_id"]).get_indexer(batches["row_id"])
        missing = int((pos < 0).sum())
        assert missing == 0, (
            f"{missing} rows have no entry in predictions.csv, e.g. "
            f"{batches['row_id'].to_numpy()[pos < 0][:3].tolist()}")

        self.batches = batches
        self.row_id = batches["row_id"].to_numpy()
        self.id_test = batches["id_test"].to_numpy()
        self.log_post = log_post_all[pos]
        self.train_prior = np.asarray(train_prior, float)
        self.log_train_prior = np.log(self.train_prior)
        self.theta = priors[[f"p{y}" for y in range(Y)]].to_numpy()
        self.log_theta = np.log(self.theta)
        C = len(self.theta)
        if location_prior is None:
            self.log_p_theta = np.full(C, -np.log(C))
        else:
            location_prior = np.asarray(location_prior, dtype=np.float64)
            assert location_prior.shape == (C,) and np.all(location_prior > 0)
            self.log_p_theta = np.log(location_prior / location_prior.sum())
        self.starts = batch_starts_from_ids(self.id_test)
        self._inf = None

    @property
    def n(self) -> int:
        return len(self.row_id)

    @property
    def C(self) -> int:
        return len(self.theta)

    @property
    def inference(self):
        """The exact inference, computed once and cached."""
        if self._inf is None:
            self._inf = infer(self.log_post, self.starts, self.log_train_prior,
                              self.log_theta, self.log_p_theta)
        return self._inf

    def theta_star_per_row(self) -> np.ndarray:
        """The location prior of each row's batch -- oracle use only."""
        meta = pd.read_csv(self.dir / "batch_meta.csv")
        lookup = dict(zip(meta["id_test"], meta["theta_star_index"]))
        return self.theta[np.array([lookup[i] for i in self.id_test])]


def predict(problem: Problem, name: str) -> tuple[np.ndarray, np.ndarray]:
    """``(pred, confidence)`` for one named predictor."""
    if name == "base":
        # argmax of the posterior under the base prior: no adaptation at all.
        pred = problem.log_post.argmax(axis=1)
        return pred, np.exp(problem.log_post[np.arange(problem.n), pred])

    if name == "true_plugin":
        pred, unc = plugin_for_prior(problem.log_post, problem.log_train_prior,
                                     problem.theta_star_per_row())
        return pred, 1.0 - unc

    inf = problem.inference
    if name == "map_plugin":
        return inf.map_pred, 1.0 - inf.map_unc
    if name == "bayes_total":
        return inf.bayes_pred, -inf.total
    if name == "bayes_epistemic":
        # Score by E, ties broken by T.
        return inf.bayes_pred, rank_confidence(inf.epistemic, inf.total)
    if name == "bayes_aleatoric":
        return inf.bayes_pred, rank_confidence(inf.aleatoric, inf.total)
    raise ValueError(f"unknown predictor {name!r}; known: {list(PREDICTORS)}")


def write_submission(path: Path, row_id: np.ndarray, pred: np.ndarray,
                     conf: np.ndarray) -> None:
    """Write a submission, sorted by ``row_id`` as Kaggle expects."""
    df = pd.DataFrame({"row_id": row_id, "pred": np.asarray(pred, dtype=np.int64),
                       "confidence": np.asarray(conf, dtype=np.float64)})
    assert np.isfinite(df["confidence"]).all(), "non-finite confidence"
    df.sort_values("row_id", kind="stable").to_csv(
        path, index=False, float_format="%.10g")
