# Challenge: *Know what you don't know: nine locations*

The scripts that generate the Kaggle competition on TissueMNIST under a
location shift: the data, the metric, the starter kit, the Kaggle pages and the
pre-launch checks. The directory is self-contained — nothing here imports from
the parent project.

## The task

Competitors get **images**. The training images each carry a class and one of
9 **locations**; the locations differ only in their class frequencies. Test
images arrive in **batches** of `m ∈ {1, 2, 5, 10, 20, 50, 100}`, each batch
from one location they are not told. For every image they submit a label and a
confidence; the organiser keeps the most confident 80 % per batch size and
scores the **regret against the Bayes predictor that knows the location**,
averaged over the seven sizes. Lower is better.

Two things make the reference exact and the task hard:

1. **Synthetic labels.** Every label — training, development and test — is
   drawn from a secret label model `q(y | x)`, a calibrated CNN fitted to a
   *secret* 30 % of the TissueMNIST training split that is never released. The
   reference predictor is therefore the true Bayes predictor given the
   location, not an estimate of it: the expected score of any submission is
   `>= 0`, and the best achievable one is strictly positive at small `m`.
2. **Secret location prior.** The location of every development and test batch
   is drawn i.i.d. from a secret, non-uniform `w`. Competitors are told only
   that the development batches follow the same process.

The intended solution estimates the 9 location priors from `train.csv`,
estimates `w` by EM on the development batches, adapts to each batch by exact
Bayesian inference over the 9 locations, and ranks by **epistemic** rather than
total uncertainty (why: the docstring of `baseline_solutions.py`). None of this
is told to the competitors.

## Setup

A machine with an NVIDIA GPU; only the label model needs it, the rest is
NumPy/SciPy/pandas. Install torch **first**, from an explicit CUDA index, then
the rest:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv   # the GPU row of the table
nvidia-smi | grep "CUDA Version"                        # the driver limit
CU=cu128                                                # from the table below
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/$CU
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_arch_list())"
python -c "import torch; c = torch.nn.Conv2d(3, 8, 3).cuda(); print(c(torch.randn(2, 3, 28, 28, device='cuda')).shape)"
python selftest.py                    # ~15 s, needs no data and no network
```

The first check must print `True` and list the GPU's `sm_XY` (compute
capability X.Y); the convolution must print `torch.Size([2, 8, 26, 26])`.

**Picking `CU`.** The wheel has to fit two things, and
`torch.cuda.is_available()` checks only the first:

* **Driver.** The wheel's CUDA must not exceed the "CUDA Version" `nvidia-smi`
  prints, or torch reports *"The NVIDIA driver on your system is too old"*.
* **GPU architecture.** The wheel must contain kernels for the GPU's compute
  capability. If it does not, `is_available()` is still `True` but the first
  convolution fails with *"no kernel image is available for execution on the
  device"*.

| GPU (compute capability) | index | needs driver CUDA |
| :-- | :-- | :-- |
| Blackwell (10.0, 12.0) | `cu130` (or `cu128`) | ≥ 13.0 (≥ 12.8) |
| Turing – Hopper (7.5 – 9.0) | `cu130`, `cu128` or `cu126` | ≥ 13.0, 12.8, 12.6 |
| Volta, V100 (7.0) | `cu126` (dropped from `cu128`+) | ≥ 12.6 |

| Machine | GPU | `CU=` |
| :-- | :-- | :-- |
| `cml` | GeForce RTX 4050 Laptop (8.9) | `cu128` |
| `gauss` | RTX PRO 4000 Blackwell (12.0) | `cu130` |
| `n26` (cluster) | Tesla V100-SXM2 (7.0) | `cu126` |

Notes:

* Use `--index-url`, not `--extra-index-url`: the latter lets pip pick the
  PyPI wheel, which is the newest CUDA build. With torch already installed,
  `requirements.txt` leaves it alone.
* To replace a wrong wheel, uninstall first — pip will not swap it by itself:
  `pip uninstall -y torch torchvision`, then install again.
* On Ubuntu, `python3 -m venv` needs the `python3-venv` package.
* `nvidia-smi` failing with *"Driver/library version mismatch"* means the
  driver was updated without a reboot; reboot.

## Run

```bash
./run_all.sh              # everything, into out/final
./run_all.sh --smoke      # capped plumbing run, into out/smoke
```

`run_all.sh` runs the nine steps below from the activated virtualenv, logs to
`<out-dir>/run_all.log`, and asserts the pre-launch checks at the end. `--smoke`
trains the label model for one epoch on 3 000 images and draws few batches: it
checks the plumbing and produces nothing usable (its artifacts are marked, and
its pre-launch checks may fail). `--out-dir`, `--epochs`, `--device` and
`--clean` are described at the top of the script.

The steps, one by one:

```bash
python selftest.py
python download_data.py                                            # ~125 MB
python make_split.py out/final/split                               # rotate + split
python train_label_model.py out/final/model --split out/final/split
python make_data.py out/final/data --split out/final/split --model out/final/model
python make_package.py out/final/data out/final/kaggle
python make_metric_notebook.py metric-template.ipynb
python make_student_bundle.py out/final/kaggle_code --zip
./run_baselines.sh out/final/kaggle/organiser out/final/submissions
```

`out/` and `data/` are gitignored. A regenerated competition is not bit-identical
to an earlier one: GPU training is not deterministic, and the stratified split
is drawn by scikit-learn, whose shuffling may differ between versions.
`make_data.py` refuses a model whose `split_id` does not match the split it is
given, so a model and a split from different runs cannot be combined by
accident.

## What lands where

| Uploaded to Kaggle | Given to Kaggle, hidden from students | Never uploaded |
| :-- | :-- | :-- |
| `train.csv`, `train_images.npy`, `test_batches.csv`, `test_images.npy`, `sample_submission.csv`, `dev_test_batches.csv`, `dev_images.npy`, `dev_solution.csv`, `dev_sample_submission.csv` | `solution.csv`, `metric-template.ipynb` | `organiser/`, `manifest.json`, `out/final/split/`, `out/final/model/`, `out/final/data/` |

The first two columns are files of `out/final/kaggle/` (the notebook is in the
repo root). The starter kit is `out/final/kaggle_code.zip`; the texts of the
Kaggle pages are in `description/`.

`out/final/data/priors.txt` holds the 9 location priors and, in its header, the
secret `w`. It is an output, so it is not kept in the repo.

`organiser/test/` and `organiser/dev/` are laid out for `chal/predictors.py`:
`predictions.csv` holds the **true** label model, rescaled per pool so that the
plug-in rule under `train_prior.csv` reproduces `pred_ref` exactly
(`chal.generate.debiased_posterior`); `test_priors.csv` the 9 location priors;
`batch_meta.csv` the location of every batch; `location_prior.csv` the secret
`w`. The baselines built on it are what a competitor with a perfect model of
`q` achieves.

To fetch the solution file of the launched competition back from Kaggle:

```bash
kaggle competitions solution status know-what-you-dont-know-under-a-new-prior --json
```

## How the data are generated

**Rotation and split** (`make_split.py`). Every image is rotated once by
90/180/270° and never again; no noise is added. The original training split is
divided, stratified by the original class, into the secret set (30 %), the
training data (70 % × 90 % ≈ 104 000) and the development pool
(70 % × 10 % ≈ 11 600). The test pool is the original val + test (70 920),
partitioned into the Kaggle `Public` / `Private` / `Ignored` pools. The
original labels are used only for the stratification and for fitting the label
model: matching a released image against TissueMNIST reveals its original
label, and the competition's label is drawn independently of that given the
image.

**Label model** (`train_label_model.py`). ResNet-18 on the secret set, split
into a weight-fitting and a validation part; epoch selection and BCTS
calibration on the validation part. It scores every training,
development-pool and test-pool image. Whether `q` models the real tissue
classes well does not matter: by construction it is the true posterior of the
data-generating process. `make_data.py --temperature T` tempers it
(`q^(1/T)`), which only sets the Bayes error.

**Labels, locations, batches** (`make_data.py`).

* Each training image gets one label, `y ~ q(y | x)`. A linear program
  (`chal/locations.py`) then assigns the training images to locations using
  these **generated** labels: locations 0–7 are the 8 priors of
  `base_priors.txt` moved as little as possible, location 8 takes the
  remainder. A location's class frequency in `train.csv` *is* its prior
  `pi_l`, exactly.
* Every development and test row is drawn independently: the batch's location
  `l ~ w`, then `y ~ pi_l`, then an image of the pool `P` with probability
  `∝ q(y | x)` (`chal/protocol.py`). No pool image carries a fixed label; the
  same image can appear in several rows with different labels, so linking
  copies across batches reveals nothing.
* With this sampler the posterior of a row from location `l` is exactly
  `p_l(y | x) ∝ q(y | x) pi_l(y) / pibar_P(y)`, with `pibar_P` the mean of `q`
  over the pool. The reference prediction is its argmax — the Bayes rule given
  the location. `selftest.py` checks the sampler against it.

**The location prior** `w` defaults to `chal/locprior.py:W_DEFAULT` — TV 0.42
from uniform, every weight `>= 0.03`, far from the training location shares. A
milder `w` (TV 0.28) was tried first: estimating it by EM paid on the public
split but not detectably on the private one. `make_data.py` refuses a `w` that
violates those guards and reports how well EM on the development labels
recovers it. The development set has 1 500 batches (`--dev-n-min 150`) to make
that estimate usable; reusing development images costs nothing since every row
has a fresh label.

## Pre-launch checks

`run_baselines.sh` estimates `w` by EM on `organiser/dev` exactly as a
competitor would (`estimate_location_prior.py`: development labels plus the
location priors), then runs every baseline under a uniform, the EM-estimated
and the true location prior, and compares them on shared bootstrap resamples.
Read, in order:

1. **`true_plugin` must score exactly 0.000000.** It *is* the reference; any
   other number means generation and scoring have diverged.
2. **No score may be clearly negative.** The reference is the Bayes predictor
   given the location; a clearly negative score is a bug.
3. **`bayes_epistemic_em` must beat `bayes_epistemic_uniform`** with a paired
   interval excluding 0. Otherwise the secret prior is not worth discovering;
   move `w` further from uniform (`make_data.py --location-prior`) and
   regenerate.
4. **`bayes_epistemic` must beat `bayes_total`** (same prior), again with a
   paired interval excluding 0. Otherwise ranking by epistemic uncertainty is
   not rewarded.

`run_all.sh` asserts the point-estimate version of these four on the private
split and exits non-zero if one fails.

**The metric cannot be debugged after launch.** `chal/metric.py` is the only
copy: `evaluate.py` imports it, `metric-template.ipynb` is generated from it by
`make_metric_notebook.py`, and `make_student_bundle.py` copies it. Never
hand-edit the notebook. The docstring of `score()` is rendered to competitors,
so the bundle audit also applies to it.

**The bootstrap unit is the batch, not the row.** Rows inside a batch share
the location and the same adaptation evidence; `N(m)`, not `B_m`, sets the
width of every interval.

## Layout

| Script | Does |
| :-- | :-- |
| `run_all.sh` | the whole pipeline, with the pre-launch checks asserted |
| `selftest.py` | brute-force checks of the inference, the sampler, the metric, the generation |
| `download_data.py` | fetch and verify the TissueMNIST archive |
| `make_split.py` | rotate every image once; secret / training / development / test split |
| `train_label_model.py` | fit and calibrate the label model on the secret set; score every pool |
| `make_data.py` | generated labels, locations, location priors, batches, reference predictions |
| `make_package.py` | the Kaggle upload, `solution.csv` and `organiser/` |
| `make_metric_notebook.py` | generate `metric-template.ipynb` from `chal/metric.py` |
| `make_student_bundle.py` | assemble `student/` + the metric into the starter kit, audited for leaks |
| `run_baselines.sh` | the four below, in the order the pre-launch checks need |
| `estimate_location_prior.py` | EM estimate of `w` from the development labels, with a bootstrap |
| `baseline_solutions.py` | the baselines under a chosen location prior, the intended optimum among them |
| `compare_baselines.py` | paired bootstrap comparison of submissions |
| `evaluate.py` | score a submission, per-size table with bootstrap CIs, plot |

| Module | Holds |
| :-- | :-- |
| `chal/data.py` | download and load TissueMNIST |
| `chal/transform.py` | the one-off rotation |
| `chal/splits.py` | the Kaggle `Usage` pools of the test pool |
| `chal/calibration.py` | BCTS, NLL, equal-mass ECE |
| `chal/priors.py` | the sliding-pair priors and the priors file format |
| `chal/locations.py` | the location linear program and the assignment |
| `chal/locprior.py` | the secret location prior, its guards, EM |
| `chal/protocol.py` | the grid, `N(m)`, the sampler `x ~ p_P(x \| y)` |
| `chal/generate.py` | tempering, label draws, the reference predictor, the organiser posterior |
| `chal/inference.py` | exact Bayesian inference over the locations: `H`, `T`, `A`, `E`, the MAP plug-in |
| `chal/predictors.py` | the reject-option baselines, built from `organiser/` |
| `chal/bootstrap.py` | batch-level bootstrap of the metric |
| `chal/metric.py` | `score()` — the single source for Kaggle, `evaluate.py` and the starter kit |

| Other | Holds |
| :-- | :-- |
| `base_priors.txt` | the 8 priors locations 0–7 start from |
| `metric-template.ipynb` | the metric as Kaggle runs it — generated, do not edit |
| `student/` | sources of the starter kit |
| `description/` | texts of the Kaggle pages |
