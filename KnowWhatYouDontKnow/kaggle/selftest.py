#!/usr/bin/env python3
"""Checks that need no data and no network. Run before spending compute.

Four things are worth brute-forcing rather than trusting:

1. **The inference.** ``chal/inference.py`` computes ``H``, ``T``, ``A`` and
   ``E`` with chunked logsumexps over an ``(n, C, Y)`` array. Here the same
   quantities are recomputed from their definitions with explicit Python
   loops over ``theta`` and ``y``, on small random problems.
2. **The protocol.** ``N(m)``, the row counts, slots ``0..m-1``, and -- the
   load-bearing one -- that the sampler realises the posterior the reference
   predictor maximises: rows drawn at location ``l`` have
   ``p(y | x, l) ~ q(y | x) pi_l(y) / pibar_P(y)``.
3. **The metric.** ``score()`` against a direct implementation, including the
   tie-break, the coverage rounding, and the sign of the ranking.
4. **Generation.** The location linear program, the rotation, the location
   prior and its EM estimate, and that the organiser files reproduce the
   reference predictor exactly.

    python selftest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from chal.inference import batch_starts_from_ids, infer, plugin_for_prior
from chal.metric import score
from chal.predictors import rank_confidence
from chal.priors import pair_priors, read_prior_set
from chal.protocol import SIZE_GRID, n_batches

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


# --- 1. the inference, against its definitions ------------------------------

def brute_force(log_post, log_train_prior, log_theta, log_p_theta):
    """``H``, ``T``, ``A``, ``E``, MAP -- straight from the definitions."""
    n, Y = log_post.shape
    C = len(log_theta)
    post = np.exp(log_post)
    theta = np.exp(log_theta)
    p_tr = np.exp(log_train_prior)
    p_theta = np.exp(log_p_theta)

    # w(x_i, theta_c) = sum_y (theta_cy / p_tr(y)) p_tr(y | x_i)
    w = np.array([[sum(theta[c, y] / p_tr[y] * post[i, y] for y in range(Y))
                   for c in range(C)] for i in range(n)])
    # p_te(y | x_i, theta_c)
    plug = np.array([[[theta[c, y] / p_tr[y] * post[i, y] / w[i, c]
                       for y in range(Y)] for c in range(C)] for i in range(n)])

    # p(theta | D) ~ p(theta) prod_j w(x_j, theta)   -- the whole batch
    unnorm = np.array([p_theta[c] * np.prod(w[:, c]) for c in range(C)])
    pth = unnorm / unnorm.sum()

    label_post = np.array([[sum(pth[c] * plug[i, c, y] for c in range(C))
                            for y in range(Y)] for i in range(n)])
    H = label_post.argmax(axis=1)
    T = 1.0 - label_post[np.arange(n), H]

    h = plug.argmax(axis=2)                                    # (n, C)
    A = np.array([sum(pth[c] * (1.0 - plug[i, c, h[i, c]]) for c in range(C))
                  for i in range(n)])
    map_c = int(unnorm.argmax())
    map_pred = h[:, map_c]
    map_unc = np.array([1.0 - plug[i, map_c, h[i, map_c]] for i in range(n)])
    return H, T, A, T - A, map_pred, map_unc


def test_inference() -> None:
    print("inference against a brute-force implementation")
    rng = np.random.default_rng(0)
    worst = 0.0
    for trial in range(12):
        n = int(rng.integers(1, 9))
        Y = int(rng.integers(3, 7))
        C = int(rng.integers(2, 6))
        log_post = np.log(rng.dirichlet(np.ones(Y) * 0.7, size=n))
        p_tr = rng.dirichlet(np.ones(Y) * 3.0)
        theta = rng.dirichlet(np.ones(Y) * 0.8, size=C)
        log_p_theta = np.full(C, -np.log(C))

        got = infer(log_post, np.array([0]), np.log(p_tr), np.log(theta),
                    log_p_theta)
        H, T, A, E, mp, mu = brute_force(log_post, np.log(p_tr), np.log(theta),
                                         log_p_theta)
        worst = max(worst, np.abs(got.total - T).max(), np.abs(got.aleatoric - A).max())
        if not (np.array_equal(got.bayes_pred, H)
                and np.allclose(got.total, T) and np.allclose(got.aleatoric, A)
                and np.allclose(got.epistemic, E)
                and np.array_equal(got.map_pred, mp)
                and np.allclose(got.map_unc, mu)):
            check(f"trial {trial} (n={n}, Y={Y}, C={C})", False)
            return
    check("H, T, A, E and the MAP plugin on 12 random problems", True,
          f"max abs error {worst:.2e}")

    # E >= 0 always: h(x, theta) minimises the per-theta conditional risk.
    rng = np.random.default_rng(7)
    log_post = np.log(rng.dirichlet(np.ones(5) * 0.4, size=200))
    p_tr = rng.dirichlet(np.ones(5) * 3.0)
    theta = rng.dirichlet(np.ones(5) * 0.5, size=6)
    out = infer(log_post, np.arange(0, 200, 10), np.log(p_tr), np.log(theta),
                np.full(6, -np.log(6)))
    check("E(x, D) >= 0 on 200 rows", out.epistemic.min() >= 0.0,
          f"min {out.epistemic.min():.2e}")
    check("T = A + E exactly", np.allclose(out.total, out.aleatoric + out.epistemic))

    # A batch of one, under a Theta of one prior, is the plugin for that prior.
    theta1 = rng.dirichlet(np.ones(5) * 2.0, size=1)
    single = infer(log_post, np.arange(200), np.log(p_tr), np.log(theta1),
                   np.zeros(1))
    pred, unc = plugin_for_prior(log_post, np.log(p_tr), theta1[0])
    check("C = 1 collapses to the plugin predictor",
          np.array_equal(single.bayes_pred, pred) and np.allclose(single.total, unc))
    check("C = 1 leaves no epistemic uncertainty",
          float(np.abs(single.epistemic).max()) < 1e-12)

    # The full-batch posterior does not depend on which row is called the query.
    perm = rng.permutation(10)
    a = infer(log_post[:10], np.array([0]), np.log(p_tr), np.log(theta),
              np.full(6, -np.log(6)))
    b = infer(log_post[:10][perm], np.array([0]), np.log(p_tr), np.log(theta),
              np.full(6, -np.log(6)))
    check("the batch posterior is permutation-invariant",
          np.allclose(a.total[perm], b.total) and np.allclose(a.pth, b.pth))


# --- 2. the protocol -------------------------------------------------------

def test_protocol() -> None:
    print("\nprotocol")
    counts = {m: n_batches(m) for m in SIZE_GRID}
    check("N(m) on the default grid",
          list(counts.values()) == [2000, 1000, 400, 200, 200, 200, 200],
          str(list(counts.values())))
    rows = {m: counts[m] * m for m in SIZE_GRID}
    check("B_m on the default grid",
          list(rows.values()) == [2000, 2000, 2000, 2000, 4000, 10000, 20000])
    check("4200 batches and 42 000 rows per usage",
          sum(counts.values()) == 4200 and sum(rows.values()) == 42000,
          f"{sum(counts.values())} batches, {sum(rows.values())} rows")

    from chal.protocol import PoolSampler, draw_rows
    rng = np.random.default_rng(0)
    q_pool = rng.dirichlet(np.ones(8) * 0.5, size=1600)
    theta8 = pair_priors(8, 0.35)
    C = len(theta8)
    w = rng.dirichlet(np.ones(C) * 3.0)
    drawn = draw_rows(PoolSampler(q_pool), theta8, w, rng,
                      grid=(1, 2, 5, 10), n_min=4, scale=8)
    ok = True
    for b in range(drawn.n_batches):
        s = drawn.slot[drawn.gen_batch == b]
        ok &= np.array_equal(s, np.arange(drawn.batch_m[b]))
    check("slots run 0..m-1 in every batch", bool(ok))
    starts = batch_starts_from_ids(drawn.gen_batch)
    check("batches are contiguous and sorted", len(starts) == drawn.n_batches)

    # Locations i.i.d. from w; labels i.i.d. from the location's prior.
    many = draw_rows(PoolSampler(q_pool), theta8, w,
                     np.random.default_rng(1), grid=(10,), n_min=20000,
                     scale=10)
    freq = np.bincount(many.batch_location, minlength=C) / many.n_batches
    check("batch locations follow w", np.abs(freq - w).max() < 0.012,
          f"max |freq - w| {np.abs(freq - w).max():.4f}")
    loc = many.location_of_row
    lab = np.stack([np.bincount(many.label[loc == l], minlength=8)
                    / (loc == l).sum() for l in range(C)])
    check("row labels follow the location's prior",
          np.abs(lab - theta8).max() < 0.02,
          f"max |freq - pi_l| {np.abs(lab - theta8).max():.4f}")

    # The load-bearing one: given the image and the location, the drawn label
    # has exactly the posterior the reference predictor maximises. Checked on
    # a pool of 10 images, where every image is drawn often.
    from chal.generate import reference
    q10 = rng.dirichlet(np.ones(4) * 0.8, size=10)
    theta4 = np.array([[0.55, 0.25, 0.15, 0.05], [0.1, 0.2, 0.3, 0.4]])
    sampler = PoolSampler(q10)
    drawn = draw_rows(sampler, theta4, np.array([0.3, 0.7]),
                      np.random.default_rng(2), grid=(1,), n_min=400_000,
                      scale=1)
    # Judged by z-score, since the rarer (image, location) cells get only a
    # few thousand draws: max |z| over the 80 cells stays below 5 unless the
    # sampler is biased.
    worst = 0.0
    for l in range(2):
        want = q10 * theta4[l] / sampler.pi_bar
        want /= want.sum(axis=1, keepdims=True)
        for i in range(10):
            sel = (drawn.location_of_row == l) & (drawn.pool_row == i)
            n = int(sel.sum())
            got = np.bincount(drawn.label[sel], minlength=4) / n
            se = np.sqrt(want[i] * (1 - want[i]) / n)
            worst = max(worst, float((np.abs(got - want[i]) / se).max()))
    check("p(y | x, l) of the drawn rows is q pi_l / pibar, normalised",
          worst < 5.0, f"max |z| {worst:.2f} over 80 cells (Monte Carlo)")
    pred = reference(q10, sampler.pi_bar, np.repeat(theta4[1:], 10, axis=0))
    want = (q10 * theta4[1] / sampler.pi_bar).argmax(axis=1)
    check("reference() is that posterior's argmax", np.array_equal(pred, want))


# --- 3. the metric ---------------------------------------------------------

def naive_score(sol: pd.DataFrame, sub: pd.DataFrame, coverage: float) -> float:
    """The metric restated with Python loops, for comparison."""
    merged = sol.merge(sub, on="row_id")
    per_m = []
    for m in sorted(merged["m"].unique()):
        g = merged[merged["m"] == m]
        recs = sorted(((-r.confidence, r.row_id, r) for r in g.itertuples()),
                      key=lambda t: (t[0], t[1]))
        k = max(1, int(np.ceil(coverage * len(recs))))
        tot = 0.0
        for _, _, r in recs[:k]:
            tot += (r.pred != r.label) - (r.pred_ref != r.label)
        per_m.append(tot / k)
    return float(np.mean(per_m))


def test_metric() -> None:
    print("\nmetric")
    rng = np.random.default_rng(3)
    sizes = (1, 2, 5)
    n_per = 60
    rows = []
    for m in sizes:
        for i in range(n_per):
            rows.append({"row_id": len(rows), "id_test": len(rows) // m, "m": m,
                         "label": int(rng.integers(8)),
                         "pred_ref": int(rng.integers(8))})
    sol = pd.DataFrame(rows)
    sub = pd.DataFrame({"row_id": sol["row_id"],
                        "pred": rng.integers(0, 8, len(sol)),
                        "confidence": rng.random(len(sol))})
    got = score(sol.copy(), sub.copy(), "row_id", expected_sizes=sizes)
    want = naive_score(sol, sub, 0.8)
    check("score() matches a loop implementation", abs(got - want) < 1e-12,
          f"{got:+.6f} vs {want:+.6f}")

    # A submission that copies the reference has regret exactly zero, whatever
    # the confidences are. This is the true_plugin check, in miniature.
    sub_ref = sub.copy()
    sub_ref["pred"] = sol["pred_ref"]
    check("copying pred_ref scores exactly 0",
          score(sol.copy(), sub_ref.copy(), "row_id", expected_sizes=sizes) == 0.0)

    # Constant confidence must still be deterministic: ties break on row_id.
    flat = sub.copy()
    flat["confidence"] = 1.0
    a = score(sol.copy(), flat.copy(), "row_id", expected_sizes=sizes)
    b = score(sol.copy(), flat.sample(frac=1.0, random_state=1).copy(), "row_id",
              expected_sizes=sizes)
    check("constant confidence is scored deterministically", a == b, f"{a:+.6f}")

    # Higher confidence must mean "kept first": hiding the errors must help.
    wrong = (sub["pred"].to_numpy() != sol["label"].to_numpy())
    good = sub.copy()
    good["confidence"] = np.where(wrong, 0.0, 1.0)
    bad = sub.copy()
    bad["confidence"] = np.where(wrong, 1.0, 0.0)
    s_good = score(sol.copy(), good.copy(), "row_id", expected_sizes=sizes)
    s_bad = score(sol.copy(), bad.copy(), "row_id", expected_sizes=sizes)
    check("low confidence on errors scores better", s_good < s_bad,
          f"{s_good:+.6f} < {s_bad:+.6f}")

    # rank_confidence must reproduce the lexicographic order it was given.
    e = rng.choice([0.0, 0.0, 0.0, 0.1, 0.2], size=50)
    t = rng.random(50)
    conf = rank_confidence(e, t)
    want_order = np.lexsort((t, e))
    check("rank_confidence reproduces (E, then T) ascending",
          np.array_equal(np.argsort(-conf, kind="stable"), want_order))


# --- 4. generation ----------------------------------------------------------

def test_generation() -> None:
    print("\ngeneration")
    from chal.generate import debiased_posterior, draw_labels, reference, \
        tempered
    from chal.locations import assign_locations, plan_locations
    from chal.locprior import W_DEFAULT, check_location_prior, \
        em_location_prior, label_loglik
    from chal.protocol import PoolSampler, draw_rows
    from chal.transform import rotate

    # The location LP, on class counts shaped like TissueMNIST's but smaller.
    D = np.array([4800, 700, 530, 1390, 1060, 690, 3530, 2210])
    base = pair_priors(8, 0.35)
    on_file = read_prior_set(Path(__file__).resolve().parent / "base_priors.txt",
                             np.full(8, 1 / 8)).theta
    check("base_priors.txt holds the sliding-pair priors, tau = 0.35",
          on_file.shape == base.shape and np.allclose(on_file, base, atol=1e-12))
    plan = plan_locations(D, base, eps=0.01, n_min=500)
    check("every training image lands in exactly one location",
          np.array_equal(plan.counts.sum(axis=0), D))
    check("locations 0..7 honour n_min", plan.sizes[:-1].min() >= 500,
          f"sizes {plan.sizes.tolist()}")
    check("every prior clears the floor (up to rounding)",
          plan.priors.min() >= 0.01 - 2.0 / plan.sizes.min(),
          f"min entry {plan.priors.min():.4f}")
    check("the priors moved by at most delta (up to rounding)",
          plan.tv_change.max() <= plan.delta + 8.0 / plan.sizes[:-1].min(),
          f"max TV moved {plan.tv_change.max():.4f}, delta {plan.delta:.4f}")
    y = np.repeat(np.arange(8), D)
    loc = assign_locations(y, plan, np.random.default_rng(0))
    freq = np.stack([np.bincount(y[loc == l], minlength=8)
                     for l in range(plan.L)])
    check("class frequency per location is exactly the location's prior",
          np.allclose(freq / freq.sum(1, keepdims=True), plan.priors))

    # The rotation.
    rng = np.random.default_rng(1)
    X = rng.integers(0, 256, size=(300, 28, 28), dtype=np.uint8)
    Xt, k = rotate(X, np.random.default_rng(2))
    check("rotations are 90, 180 or 270 degrees, never 0",
          set(np.unique(k).tolist()) == {1, 2, 3})
    check("undoing the rotation restores the image exactly",
          all(np.array_equal(np.rot90(Xt[i], -k[i]), X[i]) for i in range(300)))

    # The label model: tempering and label draws.
    lq = np.log(rng.dirichlet(np.ones(8) * 0.6, size=4000))
    q = tempered(lq, 1.0)
    check("temperature 1 leaves q unchanged", np.allclose(q, np.exp(lq)))
    hot = tempered(lq, 2.0)
    check("temperature > 1 flattens q",
          hot.max(axis=1).mean() < q.max(axis=1).mean())
    y_draw = draw_labels(np.repeat(q[:1], 200_000, axis=0),
                         np.random.default_rng(3))
    check("draw_labels samples from q", np.abs(
        np.bincount(y_draw, minlength=8) / 200_000 - q[0]).max() < 0.005)

    # The location prior and its EM estimate.
    sizes = np.array([5000] * 8 + [60000])
    check("the default w passes its guards",
          bool(check_location_prior(np.array(W_DEFAULT), sizes)))
    rng = np.random.default_rng(4)
    theta = plan.priors
    w = np.array(W_DEFAULT)
    drawn = draw_rows(PoolSampler(q), theta, w, rng, grid=(20,), n_min=4000,
                      scale=20)
    starts = batch_starts_from_ids(drawn.gen_batch)
    w_hat, _ = em_location_prior(label_loglik(drawn.label, starts,
                                              np.log(theta)))
    check("EM on the labels recovers w", np.abs(w_hat - w).max() < 0.02,
          f"max |w_hat - w| {np.abs(w_hat - w).max():.4f}")

    # The organiser files reproduce the reference exactly: per-pool debiasing
    # onto the split's base prior, then the plug-in rule under that prior.
    pools = [q[:1500], q[1500:2700], q[2700:]]
    split_bar = q.mean(axis=0)
    bad = 0
    for qp in pools:
        s = PoolSampler(qp)
        d = draw_rows(s, theta, w, rng, grid=(5,), n_min=400, scale=5)
        rows = qp[d.pool_row]
        ref = reference(rows, s.pi_bar, theta[d.location_of_row])
        post = debiased_posterior(rows, s.pi_bar, split_bar)
        pred, _ = plugin_for_prior(np.log(post), np.log(split_bar),
                                   theta[d.location_of_row])
        bad += int((pred != ref).sum())
    check("true_plugin on the organiser files reproduces pred_ref", bad == 0,
          f"{bad} mismatches")


def main() -> None:
    print("challenge self-test -- no data, no network\n")
    test_inference()
    test_protocol()
    test_metric()
    test_generation()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all self-tests passed")


if __name__ == "__main__":
    main()
