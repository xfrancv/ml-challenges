#!/usr/bin/env python3
"""Train and calibrate the label model ``q(y | x)``.

The label model *defines* the ground truth of the challenge: the labels of the
training data, of every development row and of every test row are drawn from
it, and the reference predictor is its Bayes rule given the location. It is a
ResNet-18 fitted to the **original** TissueMNIST labels of the **secret set**
of ``make_split.py`` -- images the competitors never see:

1. the secret set is split, stratified by class, into a weight-fitting part and
   a validation part (``--val-fraction``);
2. the weights are fitted on the weight-fitting part, the epoch is selected on
   the validation part;
3. the posterior is calibrated (BCTS, ``softmax(z_k / T_k + b_k)``) on the
   validation part.

The calibrated network is then applied to every image of the training data, the
development pool and the test pool -- as rotated, exactly as released.

Whether ``q`` is a good model of the real tissue classes does not matter for
correctness: by construction it is the true posterior of the data-generating
process. Calibration matters for realism only -- it sets how noisy the labels
are -- and ``make_data.py --temperature`` can change that afterwards.

Outputs in ``out_dir``:

``model.pt``      weights, the calibration map, normalisation
``log_post.npz``  calibrated ``log q(y | x)`` for the validation part, the
                  training data, the development pool and the test pool -- so no
                  later script needs torch
``report.txt``    splits, calibration numbers, per-class error
``learning_curves.png``

Run with::

    python train_label_model.py out/final/model --split out/final/split
    python train_label_model.py out/smoke/model --split out/smoke/split \\
        --epochs 1 --max-fit 3000                              # smoke only
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from chal import data as chdata
from chal.calibration import calibration_summary, fit_bcts, log_softmax_np

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None


def make_model(num_classes: int, in_channels: int = 1) -> nn.Module:
    """ResNet-18 with the small-input stem (3x3 conv, stride 1, no max-pool)."""
    from torchvision.models import resnet18
    m = resnet18(weights=None, num_classes=num_classes)
    m.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1,
                        padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


def to_tensor(x: np.ndarray, mean: float, std: float) -> torch.Tensor:
    """uint8 (N, H, W) -> normalised float32 (N, 1, H, W)."""
    x = x.astype(np.float32) / 255.0
    x = (x - mean) / std
    return torch.from_numpy(x[:, None, :, :].copy())


@torch.no_grad()
def collect_logits(model: nn.Module, X: torch.Tensor, device: torch.device,
                   batch_size: int = 512) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        out.append(model(X[i:i + batch_size].to(device)).cpu())
    return torch.cat(out).numpy().astype(np.float64)


def collect_logits_np(model: nn.Module, X: np.ndarray, mean: float, std: float,
                      device: torch.device, chunk: int = 16384) -> np.ndarray:
    """``collect_logits`` on uint8 images, normalised a chunk at a time.

    The three pools hold ~187 000 images; as one float32 tensor they would
    take 600 MB for nothing.
    """
    return np.concatenate([
        collect_logits(model, to_tensor(X[i:i + chunk], mean, std), device)
        for i in range(0, len(X), chunk)])


def per_class_error(y: np.ndarray, pred: np.ndarray, Y: int) -> np.ndarray:
    err = np.full(Y, np.nan)
    for c in range(Y):
        mask = y == c
        if mask.any():
            err[c] = float((pred[mask] != c).mean())
    return err


def make_curves_figure(history: dict, best_epoch: int, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = np.arange(1, len(history["fit_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, label, tr, va in (
            (axes[0], "loss", history["fit_loss"], history["cal_loss"]),
            (axes[1], "classification error", history["fit_err"], history["cal_err"])):
        ax.plot(epochs, tr, lw=1.8, color="C0", marker="o", ms=3,
                label="fit part (subsample)")
        ax.plot(epochs, va, lw=1.8, color="C1", marker="s", ms=3,
                label="validation part")
        ax.axvline(best_epoch, color="0.4", ls="--", lw=1,
                   label=f"best epoch ({best_epoch})")
        ax.set_xlabel("epoch")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Label model: training curves")
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curves.png", dpi=130)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_dir", type=Path, help="directory receiving all outputs")
    p.add_argument("--split", type=Path, required=True,
                   help="output directory of make_split.py")
    p.add_argument("--val-fraction", type=float, default=0.20,
                   help="portion of the secret set used for epoch selection "
                        "and BCTS (default 0.20)")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"),
                   help="'auto' (default) uses the GPU when torch sees one")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    p.add_argument("--curve-sample", type=int, default=10000,
                   help="fit-part examples scored each epoch for the learning "
                        "curve (default 10000); model selection uses the "
                        "validation part, so this is diagnostic only")
    p.add_argument("--threads", type=int, default=0,
                   help="torch CPU threads (0 = leave torch's default)")
    p.add_argument("--max-fit", type=int, default=0,
                   help="cap the fit part at this many examples (0 = no cap). "
                        "For smoke runs; a capped run is marked in report.txt "
                        "and must not be used for the competition.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("error: --device cuda requested but CUDA is not available")
    device = torch.device(args.device)

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- data ---------------------------------------------------------------
    Y = chdata.NUM_CLASSES
    split = np.load(Path(args.split) / "split.npz")
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        split["secret_images"], split["secret_orig_label"],
        test_size=args.val_fraction, stratify=split["secret_orig_label"],
        random_state=args.seed)
    # name -> images of every pool scored after training
    targets = {name: split[f"{name}_images"]
               for name in ("train", "devpool", "test")}
    capped = bool(args.max_fit and args.max_fit < len(y_fit))
    if capped:
        # Stratified cap, so the class frequency is unchanged.
        rng = np.random.default_rng(args.seed)
        keep = np.concatenate([
            rng.permutation(np.flatnonzero(y_fit == c))[
                :max(1, round(args.max_fit * (y_fit == c).mean()))]
            for c in range(Y)])
        X_fit, y_fit = X_fit[keep], y_fit[keep]

    xf = X_fit.astype(np.float32) / 255.0
    norm_mean = float(xf.mean())
    norm_std = float(xf.std()) + 1e-7
    del xf

    Xt_fit = to_tensor(X_fit, norm_mean, norm_std)
    Xt_cal = to_tensor(X_cal, norm_mean, norm_std)
    yt_fit = torch.from_numpy(y_fit)
    yt_cal = torch.from_numpy(y_cal)

    fit_prior = np.bincount(y_fit, minlength=Y).astype(float)
    fit_prior /= fit_prior.sum()
    assert np.all(fit_prior > 0), "some class is absent from the fit part"

    # --- training ----------------------------------------------------------
    model = make_model(Y).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    # Plain unweighted CE and a uniform shuffle: the calibrated posterior is
    # then the posterior under the secret set's class frequency.
    assert criterion.weight is None, "class-weighted loss"
    assert criterion.reduction == "mean"

    @torch.no_grad()
    def evaluate(X, y):
        model.eval()
        losses, preds = 0.0, []
        for i in range(0, len(X), 512):
            xb, yb = X[i:i + 512].to(device), y[i:i + 512].to(device)
            logits = model(xb)
            losses += F.cross_entropy(logits, yb, reduction="sum").item()
            preds.append(logits.argmax(dim=1).cpu())
        preds = torch.cat(preds)
        return losses / len(X), float((preds != y).float().mean())

    history = {k: [] for k in ("fit_loss", "fit_err", "cal_loss", "cal_err")}
    best_err, best_epoch, best_state = float("inf"), 0, None
    n_fit = len(Xt_fit)

    # The fit-part curve is diagnostic only -- model selection uses the
    # validation part -- so it is measured on a fixed subsample.
    curve_n = min(args.curve_sample, n_fit)
    curve_idx = torch.from_numpy(
        np.random.default_rng(args.seed).choice(n_fit, curve_n, replace=False))
    Xt_curve, yt_curve = Xt_fit[curve_idx], yt_fit[curve_idx]

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n_fit)
        batches = range(0, n_fit, args.batch_size)
        if tqdm is not None:
            batches = tqdm(batches, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for i in batches:
            idx = perm[i:i + args.batch_size]
            xb, yb = Xt_fit[idx].to(device), yt_fit[idx].to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        tr_loss, tr_err = evaluate(Xt_curve, yt_curve)
        ca_loss, ca_err = evaluate(Xt_cal, yt_cal)
        for k, v in zip(history, (tr_loss, tr_err, ca_loss, ca_err)):
            history[k].append(v)
        print(f"epoch {epoch:3d}/{args.epochs}  fit loss {tr_loss:.4f} "
              f"err {tr_err:.4f}   cal loss {ca_loss:.4f} err {ca_err:.4f}",
              flush=True)
        if ca_err < best_err:
            best_err, best_epoch = ca_err, epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    # --- calibration on the validation part ---------------------------------
    cal_logits = collect_logits(model, Xt_cal, device)
    temperature, bias = fit_bcts(torch.from_numpy(cal_logits).float(), yt_cal)
    assert np.all(temperature > 0), "non-positive temperature from calibration"
    scale = 1.0 / temperature

    def calibrate(logits: np.ndarray) -> np.ndarray:
        return log_softmax_np(logits * scale + bias)

    calib = calibration_summary(cal_logits, y_cal, scale, bias)

    # --- calibrated posteriors of every pool ---------------------------------
    lp_cal = calibrate(cal_logits)
    lps = {name: calibrate(collect_logits_np(model, X, norm_mean, norm_std,
                                             device))
           for name, X in targets.items()}
    for name, lp in (("cal", lp_cal), *lps.items()):
        assert np.all(np.isfinite(lp)), f"non-finite log-posterior on {name}"

    np.savez_compressed(
        out_dir / "log_post.npz",
        log_post_cal=lp_cal.astype(np.float32), y_cal=y_cal,
        fit_prior=fit_prior, seed=args.seed, capped=capped,
        class_names=np.array(chdata.CLASS_NAMES, dtype=object),
        **{f"log_post_{name}": lp.astype(np.float32)
           for name, lp in lps.items()},
        # make_data.py refuses posteriors of a different split.
        split_id=str(split["run_id"]),
    )

    torch.save({
        "model_state": best_state, "num_classes": Y,
        "calibration": "bcts", "temperature": temperature,
        "calib_scale": scale, "calib_bias": bias,
        "calibration_metrics": calib, "fit_prior": fit_prior,
        "split": str(args.split), "split_id": str(split["run_id"]),
        "norm_mean": norm_mean, "norm_std": norm_std,
        "best_epoch": best_epoch, "seed": args.seed,
        "val_fraction": args.val_fraction,
        "max_fit": args.max_fit,
    }, out_dir / "model.pt")

    cal_pred = lp_cal.argmax(axis=1)
    cls_err = per_class_error(y_cal, cal_pred, Y)
    # Agreement with the ORIGINAL labels, for context: the challenge's labels
    # are drawn from q, so these are not its error rates.
    orig = {name: split[f"{name}_orig_label"] for name in targets}
    lines = [
        "Label model q(y | x): training and calibration",
        "=" * 78,
        f"timestamp   : {datetime.now().isoformat(timespec='seconds')}",
        f"command     : {' '.join(sys.argv)}",
        f"device      : {args.device}   seed: {args.seed}",
        f"epochs      : {args.epochs}  batch {args.batch_size}  lr {args.lr:g}",
        f"curve sample: {curve_n:,} fit examples scored per epoch (diagnostic)",
        f"split       : {args.split} (run {split['run_id']})",
        "-" * 78,
        f"  {'secret set -> fit':<26} : {len(y_fit):,}"
        + ("   (CAPPED by --max-fit; NOT for the competition)" if capped else ""),
        f"  {'secret set -> validation':<26} : {len(y_cal):,}   (epoch "
        f"selection and BCTS)",
    ] + [
        f"  {'scored: ' + name:<26} : {len(X):,}"
        for name, X in targets.items()
    ] + [
        f"best epoch  : {best_epoch}  (validation error {best_err:.4f})",
        "-" * 78,
        "calibration : bcts   softmax(z_k / T_k + b_k)",
        f"  NLL : {calib['nll_before']:.4f} -> {calib['nll_after']:.4f}",
        f"  ECE : {calib['ece_before']:.4f} -> {calib['ece_after']:.4f}"
        f"   ({calib['n_ece_bins']} equal-mass bins)",
    ] + ([
        "  !! ECE got worse after calibration; check the validation size."
    ] if calib["ece_after"] > calib["ece_before"] + 1e-3 else []) + [
        "-" * 78,
        "fit-part class frequency (original labels):",
        "  " + "  ".join(f"{v:.4f}" for v in fit_prior),
        "",
        f"error against the original labels, validation : "
        f"{float((cal_pred != y_cal).mean()):.4f}",
    ] + [
        f"error against the original labels, {name:<10} : "
        f"{float((lps[name].argmax(axis=1) != orig[name]).mean()):.4f}"
        for name in targets
    ] + [
        "",
        "per-class error against the original labels, validation:",
    ] + [
        f"  {c}  {chdata.CLASS_NAMES[c][:44]:<44} {cls_err[c]:.4f}"
        for c in range(Y)
    ] + [""]

    report = "\n".join(lines)
    (out_dir / "report.txt").write_text(report)
    print(report)
    make_curves_figure(history, best_epoch, out_dir)
    print(f"outputs in {out_dir}/: model.pt, log_post.npz, report.txt, "
          f"learning_curves.png")


if __name__ == "__main__":
    main()
