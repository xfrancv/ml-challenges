#!/usr/bin/env python3
"""Write the Kaggle upload.

Reads ``make_data.py``'s ``data.npz`` and writes three groups of files.
Needs no torch and no model: the labels and the reference predictions were
fixed at generation time.

**Published** (upload to Kaggle; ``row k`` of a CSV is ``image k`` of its
``.npy``):

``train.csv``, ``train_images.npy``          labeled training images and their
                                             location
``dev_test_batches.csv``, ``dev_images.npy`` one row per development image:
                                             its batch and the batch size
``dev_solution.csv``                         labels and the reference predictor
``dev_sample_submission.csv``
``test_batches.csv``, ``test_images.npy``    one row per test image: its batch
                                             and the batch size
``sample_submission.csv``

The sample submissions predict the most frequent training class with a
constant confidence. They show the format and nothing else.

**Given to Kaggle, hidden from students:** ``solution.csv``.

**Never uploaded** -- ``organiser/``:

``organiser/test/``, ``organiser/dev/``  each laid out for ``chal/predictors.py``
    (``predictions.csv``, ``test_batches.csv``, ``test_priors.csv``,
    ``train_prior.csv``, ``batch_meta.csv``, ``location_prior.csv``,
    ``solution.csv``). ``predictions.csv`` holds the TRUE label model, rescaled
    per pool so that the plug-in rule under ``train_prior.csv`` is exactly the
    reference predictor (``chal.generate.debiased_posterior``); the baselines
    built on it are therefore what a competitor with a perfect model achieves.
    ``location_prior.csv`` is the secret ``w``.
``organiser/row_source.csv``  which original image, rotation and original label
    every released development and test row came from
``manifest.json``

    python make_package.py out/final/data out/final/kaggle
    ./run_baselines.sh out/final/kaggle/organiser out/final/submissions
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from chal import data as chdata
from chal.inference import plugin_for_prior
from chal.locprior import write_location_prior
from chal.priors import read_prior_set


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data_dir", type=Path, help="output of make_data.py")
    p.add_argument("out_dir", type=Path, help="directory receiving the upload")
    p.add_argument("--seed", type=int, default=20260919,
                   help="shuffles the order of the training rows")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    Y = chdata.NUM_CLASSES

    hd = np.load(args.data_dir / "data.npz")
    if bool(hd["model_capped"]):
        print("!! the label model was trained with --max-fit: a smoke run, NOT "
              "for the competition")
    theta = hd["priors"]
    L = len(theta)
    ps = read_prior_set(args.data_dir / "priors.txt", hd["pooled_prior"])
    assert np.allclose(ps.theta, theta, atol=1e-9), \
        "priors.txt does not match data.npz"

    # --- the training data, as published -----------------------------------
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(hd["train_label"]))
    train_label = hd["train_label"][perm]
    train_location = hd["train_location"][perm]
    # A competitor's estimate of the location priors is the class frequency
    # of each location. It must be the prior the batches were drawn from.
    est = np.stack([np.bincount(train_label[train_location == l], minlength=Y)
                    for l in range(L)]).astype(float)
    est /= est.sum(axis=1, keepdims=True)
    assert np.allclose(est, theta, atol=1e-9), \
        "location class frequencies in train.csv differ from the priors"

    usages = [str(u) for u in hd["usages"]]
    test_usage = np.array(usages, dtype=object)[hd["test_usage"]]

    # The organiser files must reproduce the reference exactly: the plug-in
    # rule under each split's base prior, on its debiased posterior.
    for split, base in (("dev", hd["dev_pi_bar"]),
                        ("test", hd["test_pi_bar_split"])):
        pred, _ = plugin_for_prior(np.log(hd[f"{split}_post"]), np.log(base),
                                   theta[hd[f"{split}_location"]])
        n_bad = int((pred != hd[f"{split}_pred_ref"]).sum())
        assert n_bad == 0, f"{split}: {n_bad} rows where the organiser files " \
                           f"would not reproduce pred_ref"

    # --- published ------------------------------------------------------------
    pub = out_dir
    pub.mkdir(parents=True, exist_ok=True)
    np.save(pub / "train_images.npy", hd["train_images"][perm])
    pd.DataFrame({"label": train_label, "location": train_location}).to_csv(
        pub / "train.csv", index=False)

    majority = int(np.bincount(train_label, minlength=Y).argmax())

    def batch_files(prefix: str, split: str) -> pd.DataFrame:
        rows = pd.DataFrame({"row_id": hd[f"{split}_row_id"],
                             "id_test": hd[f"{split}_id_test"],
                             "slot": hd[f"{split}_slot"],
                             "m": hd[f"{split}_m"]})
        sample = pd.DataFrame({"row_id": rows["row_id"],
                               "pred": majority, "confidence": 0.0})
        rows.to_csv(pub / f"{prefix}test_batches.csv", index=False)
        np.save(pub / f"{split}_images.npy", hd[f"{split}_images"])
        sample.sort_values("row_id").to_csv(
            pub / f"{prefix}sample_submission.csv", index=False)
        return rows

    dev_rows = batch_files("dev_", "dev")
    test_rows = batch_files("", "test")

    dev_solution = dev_rows.assign(label=hd["dev_label"],
                                   pred_ref=hd["dev_pred_ref"])
    dev_solution.to_csv(pub / "dev_solution.csv", index=False)

    # --- hidden: Kaggle's solution file --------------------------------------
    solution = test_rows.assign(label=hd["test_label"],
                                pred_ref=hd["test_pred_ref"], Usage=test_usage)
    solution.to_csv(pub / "solution.csv", index=False)

    # --- organiser only -------------------------------------------------------
    org = pub / "organiser"
    for split, rows, sol, base in (
            ("test", test_rows, solution, hd["test_pi_bar_split"]),
            ("dev", dev_rows, dev_solution, hd["dev_pi_bar"])):
        d = org / split
        d.mkdir(parents=True, exist_ok=True)
        rows.to_csv(d / "test_batches.csv", index=False)
        sol.to_csv(d / "solution.csv", index=False)
        # Full precision, no float_format: true_plugin must reproduce pred_ref
        # exactly from this file.
        post = hd[f"{split}_post"]
        pd.DataFrame({"row_id": rows["row_id"],
                      **{f"p{y}": post[:, y] for y in range(Y)}}).to_csv(
            d / "predictions.csv", index=False)
        pd.DataFrame({"id": np.arange(L),
                      **{f"p{y}": est[:, y] for y in range(Y)}}).to_csv(
            d / "test_priors.csv", index=False)
        pd.DataFrame({f"p{y}": [base[y]] for y in range(Y)}).to_csv(
            d / "train_prior.csv", index=False)
        meta = {"id_test": hd[f"{split}_batch_id"], "m": hd[f"{split}_batch_m"]}
        if split == "test":
            meta["usage"] = np.array(usages, dtype=object)[hd["test_batch_usage"]]
        meta["theta_star_index"] = hd[f"{split}_batch_location"]
        pd.DataFrame(meta).to_csv(d / "batch_meta.csv", index=False)
        write_location_prior(hd["location_prior"], d / "location_prior.csv")

    pd.concat([
        pd.DataFrame({"split": "test", "row_id": hd["test_row_id"],
                      "source": "val+test", "source_index": hd["test_source"],
                      "quarter_turns": hd["test_rot"],
                      "orig_label": hd["test_orig_label"]}),
        pd.DataFrame({"split": "dev", "row_id": hd["dev_row_id"],
                      "source": "train", "source_index": hd["dev_source"],
                      "quarter_turns": hd["dev_rot"],
                      "orig_label": hd["dev_orig_label"]}),
    ]).to_csv(org / "row_source.csv", index=False)

    def err(pred, label):
        return float((pred != label).mean())

    manifest = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "data_run_id": str(hd["run_id"]),
        "split_id": str(hd["split_id"]),
        "data_seed": int(hd["seed"]),
        "temperature": float(hd["temperature"]),
        "model_capped": bool(hd["model_capped"]),
        "locations": int(L),
        "location_sizes": [int(v) for v in hd["location_sizes"]],
        "location_prior": [float(v) for v in hd["location_prior"]],
        "n_train": int(len(train_label)),
        "dev": {"batches": int(len(hd["dev_batch_id"])),
                "rows": int(len(dev_rows))},
        "test": {"batches": int(len(hd["test_batch_id"])),
                 "rows": int(len(test_rows)),
                 "rows_per_usage": {u: int((test_usage == u).sum())
                                    for u in usages}},
        "batches_per_location": [
            int(v) for v in np.bincount(hd["test_batch_location"], minlength=L)],
        "bayes_error_given_location": {
            "dev": err(hd["dev_pred_ref"], hd["dev_label"]),
            "test": err(hd["test_pred_ref"], hd["test_label"])},
        "sample_submission": f"class {majority}, constant confidence",
    }
    (pub / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"{len(train_label):,} training images in {L} locations; "
          f"{manifest['dev']['batches']:,} development batches "
          f"({manifest['dev']['rows']:,} rows); "
          f"{manifest['test']['batches']:,} test batches "
          f"({manifest['test']['rows']:,} rows)")
    print(f"Bayes error given the location : dev "
          f"{manifest['bayes_error_given_location']['dev']:.4f}   test "
          f"{manifest['bayes_error_given_location']['test']:.4f}")
    print("\nupload to Kaggle : train.csv train_images.npy test_batches.csv "
          "test_images.npy\n                   sample_submission.csv "
          "dev_test_batches.csv dev_images.npy\n                   "
          "dev_solution.csv dev_sample_submission.csv")
    print("give Kaggle only : solution.csv")
    print("keep local       : organiser/ manifest.json")
    print(f"\nbaselines        : ./run_baselines.sh {org} <subs_dir>")


if __name__ == "__main__":
    main()
