#!/usr/bin/env python3
"""Generate every label, location and batch.

Needs the split of ``make_split.py`` and the label model ``q(y | x)`` of
``train_label_model.py``; needs no torch.

1. **Label model.** ``q(y | x)`` is read for every image of the training data,
   the development pool and the test pool, optionally tempered
   (``--temperature``). From here on ``q`` *is* the ground truth.
2. **Training labels.** Each training image gets one label, drawn once,
   ``y ~ q(y | x)``.
3. **Locations.** ``chal.locations`` solves the linear program that moves the 8
   priors of ``--base-priors`` as little as possible so that every training
   image can be placed, using the **generated** labels; location 8 takes the
   remainder. The 9 location priors ``pi_l`` are exactly the class frequencies
   of each location's training images, and are written to the priors file.
4. **Batches.** The location of every development and test batch is drawn
   i.i.d. from the secret location prior ``w`` (``--location-prior``). Every
   row is drawn independently: ``y ~ pi_l``, then an image of the pool with
   probability proportional to ``q(y | x)`` (``chal.protocol``). No image of
   the development or test pool carries a fixed label.
5. **Reference.** The reference prediction of a row of location ``l`` from
   pool ``P`` is ``argmax_y q(y | x) pi_l(y) / pibar_P(y)``, with
   ``pibar_P`` the mean of ``q`` over the pool -- the Bayes prediction given the
   location, exactly.

Outputs in ``out_dir`` (all organiser-only; ``make_package.py`` decides what is
published):

``data.npz``    images, labels, locations, batches, ids, reference predictions
``priors.txt``  the 9 location priors; its header holds the secret ``w``
``report.txt``  the labels, the location plan, the batches, the EM check

    python make_data.py out/final/data --split out/final/split \\
        --model out/final/model

A capped smoke run, for the plumbing only::

    python make_data.py out/smoke/data --split out/smoke/split \\
        --model out/smoke/model --n-min 20 --batch-scale 100
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from chal import data as chdata
from chal.generate import DEV_ID_OFFSET, debiased_posterior, draw_labels, \
    reference, row_checks, tempered
from chal.inference import batch_starts_from_ids
from chal.locations import EPS_DEFAULT, N_MIN_DEFAULT, assign_locations, \
    plan_locations
from chal.locprior import W_DEFAULT, check_location_prior, \
    em_location_prior, label_loglik, total_variation
from chal.priors import TV_TOL, PriorSet, read_prior_set, write_prior_set
from chal.protocol import BATCH_SCALE, N_MIN, POOL_SIZE_RATIO, SIZE_GRID, \
    PoolSampler, draw_rows
from chal.splits import USAGES


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--split", type=Path, required=True,
                   help="output directory of make_split.py")
    p.add_argument("--model", type=Path, required=True,
                   help="output directory of train_label_model.py")
    p.add_argument("--base-priors", type=Path,
                   default=Path(__file__).resolve().parent / "base_priors.txt",
                   help="the 8 priors locations 0..7 start from")
    p.add_argument("--eps", type=float, default=EPS_DEFAULT,
                   help=f"floor on every location prior (default {EPS_DEFAULT})")
    p.add_argument("--n-min-location", type=int, default=N_MIN_DEFAULT,
                   help=f"minimum size of locations 0..7 (default {N_MIN_DEFAULT})")
    p.add_argument("--location-prior", type=float, nargs="+",
                   default=list(W_DEFAULT),
                   help="the secret location prior w, one weight per location "
                        f"(default {' '.join(f'{v:g}' for v in W_DEFAULT)})")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="temper the label model, q^(1/T) renormalised; T > 1 "
                        "makes the labels noisier (default 1: q as calibrated)")
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--grid", type=int, nargs="+", default=list(SIZE_GRID))
    p.add_argument("--n-min", type=int, default=N_MIN,
                   help=f"test: minimum batches per size (default {N_MIN})")
    p.add_argument("--batch-scale", type=int, default=BATCH_SCALE,
                   help=f"test: N(m) = max(n_min, ceil(scale / m)) "
                        f"(default {BATCH_SCALE})")
    p.add_argument("--dev-n-min", type=int, default=150,
                   help="development: minimum batches per size (default 150)")
    p.add_argument("--dev-batch-scale", type=int, default=500)
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Y = chdata.NUM_CLASSES
    grid = tuple(int(m) for m in args.grid)

    split = np.load(args.split / "split.npz")
    lp = np.load(args.model / "log_post.npz", allow_pickle=True)
    assert str(lp["split_id"]) == str(split["run_id"]), (
        f"the label model was trained on split {lp['split_id']}, but "
        f"{args.split} holds split {split['run_id']}; retrain or point at the "
        f"right directory")
    if bool(lp["capped"]):
        print("!! the label model was trained with --max-fit: a smoke run, NOT "
              "for the competition")

    master = np.random.default_rng(args.seed)
    rng_label, rng_loc, rng_dev, rng_ids = master.spawn(4)
    usage_rngs = master.spawn(len(USAGES))

    # --- 1. the label model ------------------------------------------------
    q = {name: tempered(lp[f"log_post_{name}"], args.temperature)
         for name in ("train", "devpool", "test")}

    # --- 2. training labels, drawn once ------------------------------------
    y_train = draw_labels(q["train"], rng_label)

    # --- 3. locations, on the generated labels -----------------------------
    D = np.bincount(y_train, minlength=Y)
    pooled = D / D.sum()
    base = read_prior_set(args.base_priors, pooled).theta
    plan = plan_locations(D, base, eps=args.eps, n_min=args.n_min_location)
    loc_train = assign_locations(y_train, plan, rng_loc)
    theta = plan.priors
    K, L = len(base), plan.L
    labels = [f"location {l} (from prior {l} of {args.base_priors.name}, "
              f"TV moved {plan.tv_change[l]:.4f})" for l in range(K)]
    labels.append(f"location {K} (the remainder of the training data)")
    ps = PriorSet(theta, labels, pooled)
    off = ps.pairwise_tv()[np.triu_indices(ps.C, k=1)]
    assert off.min() >= TV_TOL, f"two locations are only TV {off.min():.4f} apart"
    # The competitor's route to the priors is the class frequencies of each
    # location in the released training data. They must be the priors exactly.
    for l in range(L):
        freq = np.bincount(y_train[loc_train == l], minlength=Y)
        assert np.allclose(freq / freq.sum(), theta[l], atol=1e-12)

    w = np.asarray(args.location_prior, dtype=np.float64)
    assert w.shape == (L,), f"--location-prior needs {L} weights, got {len(w)}"
    assert abs(w.sum() - 1.0) < 1e-6, f"--location-prior sums to {w.sum()}"
    w = w / w.sum()
    w_checks = check_location_prior(w, plan.sizes)

    header = [
        "Location priors of the challenge",
        f"L = {ps.C} locations over Y = {ps.Y} classes; each prior is the "
        f"exact class frequency of its location's GENERATED training labels",
        f"locations 0..{K - 1} are the priors of {args.base_priors.name} "
        f"moved by TV <= {plan.tv_change.max():.4f}; location {K} is the "
        f"remainder",
        f"eps = {args.eps:g}, n_min = {args.n_min_location}, seed = {args.seed}, "
        f"label-model temperature = {args.temperature:g}",
        "SECRET location prior w (batches draw their location i.i.d. from it): "
        + " ".join(f"{v:.4f}" for v in w),
        "TV to theta_tr below is against the pooled generated-label frequency",
    ]
    write_prior_set(ps, out_dir / "priors.txt", header=header)

    # --- 4a. development batches -------------------------------------------
    dev_sampler = PoolSampler(q["devpool"])
    dev_grid = tuple(m for m in grid if m * POOL_SIZE_RATIO <= dev_sampler.n)
    if dev_grid != grid:
        print(f"development grid truncated to {list(dev_grid)} (pool of "
              f"{dev_sampler.n:,})")
    dev = draw_rows(dev_sampler, theta, w, rng_dev, grid=dev_grid,
                    n_min=args.dev_n_min, scale=args.dev_batch_scale)
    dev_q = q["devpool"][dev.pool_row]
    dev_ref = reference(dev_q, dev_sampler.pi_bar, theta[dev.location_of_row])
    dev_batch_id = DEV_ID_OFFSET + rng_ids.permutation(dev.n_batches)
    dev_id_test = dev_batch_id[dev.gen_batch]
    dev_row_id = DEV_ID_OFFSET + rng_ids.permutation(dev.n_rows)
    dev_order = np.lexsort((dev.slot, dev_id_test))
    dev_border = np.argsort(dev_batch_id)

    # --- 4b. test batches, usage by usage ----------------------------------
    test_usage_of_image = split["test_usage"]
    pi_bar_split = q["test"].mean(axis=0)
    parts = {k: [] for k in ("gen_batch", "slot", "image", "label", "usage",
                             "ref", "post")}
    batch_usage, batch_m, batch_loc = [], [], []
    pi_bar_usage = np.zeros((len(USAGES), Y))
    next_batch = 0
    for u_i in range(len(USAGES)):
        pool_idx = np.flatnonzero(test_usage_of_image == u_i)
        sampler = PoolSampler(q["test"][pool_idx])
        pi_bar_usage[u_i] = sampler.pi_bar
        drawn = draw_rows(sampler, theta, w, usage_rngs[u_i], grid=grid,
                          n_min=args.n_min, scale=args.batch_scale)
        image = pool_idx[drawn.pool_row]
        q_rows = q["test"][image]
        parts["gen_batch"].append(drawn.gen_batch + next_batch)
        parts["slot"].append(drawn.slot)
        parts["image"].append(image)
        parts["label"].append(drawn.label)
        parts["usage"].append(np.full(drawn.n_rows, u_i, dtype=np.int8))
        parts["ref"].append(reference(q_rows, sampler.pi_bar,
                                      theta[drawn.location_of_row]))
        parts["post"].append(debiased_posterior(q_rows, sampler.pi_bar,
                                                pi_bar_split))
        batch_usage.append(np.full(drawn.n_batches, u_i, dtype=np.int8))
        batch_m.append(drawn.batch_m)
        batch_loc.append(drawn.batch_location)
        next_batch += drawn.n_batches
    t = {k: np.concatenate(v) for k, v in parts.items()}
    batch_usage = np.concatenate(batch_usage)
    batch_m = np.concatenate(batch_m)
    batch_loc = np.concatenate(batch_loc)
    m_of_row = batch_m[t["gen_batch"]]
    loc_of_row = batch_loc[t["gen_batch"]]
    n_rows, n_batches = len(t["slot"]), next_batch

    # Sequential ids would hand out the Usage partition.
    batch_public_id = rng_ids.permutation(n_batches)
    id_test = batch_public_id[t["gen_batch"]]
    row_id = rng_ids.permutation(n_rows)
    checks = row_checks(id_test, t["slot"], m_of_row, row_id, t["usage"],
                        t["image"], grid)
    order = np.lexsort((t["slot"], id_test))
    border = np.argsort(batch_public_id)

    run_id = "".join(f"{v:08x}" for v in master.integers(0, 1 << 32, 4))
    dev_img = dev.pool_row[dev_order]
    np.savez(
        out_dir / "data.npz",
        run_id=run_id, split_id=str(split["run_id"]), seed=args.seed,
        temperature=args.temperature, model_capped=bool(lp["capped"]),
        grid=np.array(grid), dev_grid=np.array(dev_grid),
        usages=np.array(USAGES), priors=theta, location_sizes=plan.sizes,
        location_prior=w, pooled_prior=pooled,
        # training data, in split order; location is the released column
        train_images=split["train_images"], train_label=y_train,
        train_location=loc_train, train_source=split["train_source"],
        train_rot=split["train_rot"], train_orig_label=split["train_orig_label"],
        # development rows, in public order: by id_test, then slot
        dev_images=split["devpool_images"][dev_img],
        dev_row_id=dev_row_id[dev_order], dev_id_test=dev_id_test[dev_order],
        dev_slot=dev.slot[dev_order], dev_m=dev.m_of_row[dev_order],
        dev_label=dev.label[dev_order],
        dev_location=dev.location_of_row[dev_order],
        dev_pred_ref=dev_ref[dev_order],
        # the development split has one pool, so q needs no debiasing
        dev_post=dev_q[dev_order], dev_pi_bar=dev_sampler.pi_bar,
        dev_image=dev_img, dev_source=split["devpool_source"][dev_img],
        dev_rot=split["devpool_rot"][dev_img],
        dev_orig_label=split["devpool_orig_label"][dev_img],
        dev_batch_id=dev_batch_id[dev_border],
        dev_batch_m=dev.batch_m[dev_border],
        dev_batch_location=dev.batch_location[dev_border],
        # test rows, in public order
        test_images=split["test_images"][t["image"][order]],
        test_row_id=row_id[order], test_id_test=id_test[order],
        test_slot=t["slot"][order], test_m=m_of_row[order],
        test_label=t["label"][order], test_location=loc_of_row[order],
        test_usage=t["usage"][order], test_pred_ref=t["ref"][order],
        test_post=t["post"][order], test_pi_bar=pi_bar_usage,
        test_pi_bar_split=pi_bar_split,
        test_image=t["image"][order],
        test_source=split["test_source"][t["image"][order]],
        test_rot=split["test_rot"][t["image"][order]],
        test_orig_label=split["test_orig_label"][t["image"][order]],
        test_batch_id=batch_public_id[border], test_batch_m=batch_m[border],
        test_batch_location=batch_loc[border],
        test_batch_usage=batch_usage[border],
    )

    # --- the EM check: can w be recovered from the development batches? ----
    dev_starts = batch_starts_from_ids(dev_id_test[dev_order])
    w_hat, em_iter = em_location_prior(
        label_loglik(dev.label[dev_order], dev_starts, np.log(theta)))

    # --- report -------------------------------------------------------------
    def err(pred, label):
        return float((np.asarray(pred) != np.asarray(label)).mean())

    def row(v):
        return "  ".join(f"{x:.4f}" for x in v)

    cls = "  ".join(f"{c:>6}" for c in range(Y))
    lines = [
        "Labels, locations, batches",
        "=" * 78,
        f"timestamp : {datetime.now().isoformat(timespec='seconds')}",
        f"command   : {' '.join(sys.argv)}",
        f"run_id    : {run_id}   (split {split['run_id']})",
        "-" * 78,
        f"label model: temperature {args.temperature:g}"
        + ("   (CAPPED smoke model, NOT for the competition)"
           if bool(lp["capped"]) else ""),
        f"  training data     : {len(y_train):,} images, one label each; "
        f"agrees with the original label on "
        f"{1 - err(y_train, split['train_orig_label']):.1%}",
        f"  mean max_y q(y|x) : train {q['train'].max(1).mean():.4f}   "
        f"dev pool {q['devpool'].max(1).mean():.4f}   "
        f"test pool {q['test'].max(1).mean():.4f}",
        f"  pibar dev pool    : {row(dev_sampler.pi_bar)}",
    ] + [
        f"  pibar {u:<11} : {row(pi_bar_usage[i])}"
        for i, u in enumerate(USAGES)
    ] + [
        "-" * 78,
        f"locations: eps = {args.eps:g}, n_min = {args.n_min_location}, "
        f"LP delta = {plan.delta:.4f}",
        f"  {'loc':>3} {'size':>7} {'share':>6} {'TV moved':>9} {'w':>6}  "
        f"class frequency {cls}",
    ] + [
        f"  {l:>3} {plan.sizes[l]:>7,} {plan.sizes[l] / len(y_train):>6.1%} "
        f"{(plan.tv_change[l] if l < K else float('nan')):>9.4f} "
        f"{w[l]:>6.3f}  {'':>15} " + row(theta[l])
        for l in range(L)
    ] + [
        "  pooled generated-label frequency                  " + row(pooled),
        f"  min pairwise TV between locations : {off.min():.4f}",
        f"  w: TV {w_checks['tv_to_uniform']:.3f} to uniform, "
        f"{w_checks['tv_to_training_shares']:.3f} to the training shares",
        "-" * 78,
        "batches (location i.i.d. from w; rows i.i.d. given the location)",
        f"  development : {dev.n_batches:,} batches, {dev.n_rows:,} rows, "
        f"sizes {list(dev_grid)}",
        f"  test        : {n_batches:,} batches, {n_rows:,} rows, "
        f"sizes {list(grid)}",
    ] + [
        f"    {u:<8}  {int((batch_usage == i).sum()):,} batches, "
        f"{int((t['usage'] == i).sum()):,} rows, "
        f"{len(np.unique(t['image'][t['usage'] == i])):,} distinct images"
        for i, u in enumerate(USAGES)
    ] + [
        f"  test batches per location : "
        f"{np.bincount(batch_loc, minlength=L).tolist()}",
        f"  dev batches per location  : "
        f"{np.bincount(dev.batch_location, minlength=L).tolist()}",
        f"  checks: {', '.join(k for k, v in checks.items() if v is True)}",
        "-" * 78,
        "reference predictor (Bayes given the location) -- error rate = Bayes "
        "error",
        f"  development : {err(dev_ref, dev.label):.4f}   "
        f"non-adapted argmax q : {err(dev_q.argmax(1), dev.label):.4f}",
        f"  test        : {err(t['ref'], t['label']):.4f}   "
        f"non-adapted argmax q : "
        f"{err(q['test'][t['image']].argmax(1), t['label']):.4f}",
        "-" * 78,
        f"EM on the development labels ({dev.n_batches:,} batches, "
        f"{em_iter} iterations)",
        f"  true w      : {row(w)}",
        f"  estimated w : {row(w_hat)}",
        f"  TV(w_hat, w) = {total_variation(w_hat, w):.4f}   "
        f"max |error| = {np.abs(w_hat - w).max():.4f}   "
        f"(uniform is TV {w_checks['tv_to_uniform']:.4f} away)",
        "  whether this gain is worth anything on the metric: run_baselines.sh",
        "",
    ]
    report = "\n".join(lines)
    (out_dir / "report.txt").write_text(report)
    print(report)
    print(f"wrote {out_dir}/data.npz, {out_dir}/report.txt, "
          f"{out_dir}/priors.txt")


if __name__ == "__main__":
    main()
