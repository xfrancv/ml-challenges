#!/usr/bin/env bash
# Generate the whole competition from scratch, on a machine with a GPU.
#
# Everything EXCEPT the install: set up and activate the virtualenv first, as
# README.md "Setup" describes, then run this.
#
#   ./run_all.sh                      # out/final, 30 epochs, cuda
#   ./run_all.sh --smoke              # capped plumbing run into out/smoke
#   ./run_all.sh --clean              # erase out/ first  (asks to confirm)
#   ./run_all.sh --out-dir out/b --epochs 40 --device cpu
#
# Steps: selftest, download, split, label model, data, package, metric
# notebook, student bundle, baselines. Everything is logged to
# <out-dir>/run_all.log, and the pre-launch checks are asserted at the end.
set -euo pipefail
cd "$(dirname "$0")"

OUT=out/final
DEVICE=cuda
EPOCHS=30
CLEAN=0
SMOKE=0
PY=${PYTHON:-python}

while [[ $# -gt 0 ]]; do
  case $1 in
    --clean)    CLEAN=1; shift ;;
    --smoke)    SMOKE=1; OUT=out/smoke; EPOCHS=1; shift ;;
    --out-dir)  OUT=$2; shift 2 ;;
    --device)   DEVICE=$2; shift 2 ;;
    --epochs)   EPOCHS=$2; shift 2 ;;
    -h|--help)  sed -n '2,15p' "$0"; exit 0 ;;
    *)          echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# A smoke run must never be mistaken for the real thing: it caps the label
# model and the batch counts, and every artifact it writes is marked.
command -v "$PY" >/dev/null || {
  echo "no '$PY' on PATH -- activate the virtualenv first (README.md, Setup)" >&2
  exit 1
}

if [[ $SMOKE == 1 ]]; then
  TRAIN_EXTRA=(--max-fit 3000)
  DATA_EXTRA=(--n-min 20 --batch-scale 100)
else
  TRAIN_EXTRA=()
  DATA_EXTRA=()
fi

# --- 0. checks before anything expensive ------------------------------------
if [[ $CLEAN == 1 ]]; then
  if [[ -d out ]]; then
    echo "About to delete out/ ($(du -sh out 2>/dev/null | cut -f1))."
    read -r -p "Type 'yes' to continue: " reply
    [[ $reply == yes ]] || { echo "aborted"; exit 1; }
    rm -rf out
  fi
fi
mkdir -p "$OUT"
LOG=$OUT/run_all.log
exec > >(tee -a "$LOG") 2>&1

echo "=== run_all.sh   $(date -Is)"
echo "out=$OUT device=$DEVICE epochs=$EPOCHS smoke=$SMOKE"
echo "python: $($PY -V), $($PY -c 'import sys; print(sys.prefix)')"

$PY - "$DEVICE" <<'EOF'
import sys
device = sys.argv[1]
try:
    import torch
except ModuleNotFoundError:
    sys.exit("torch is not installed -- see README.md, Setup")
print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, "
      f"available {torch.cuda.is_available()}")
if device == "cuda":
    if not torch.cuda.is_available():
        sys.exit("--device cuda, but torch sees no GPU: the wheel's CUDA is "
                 "probably newer than the driver. Reinstall torch from an "
                 "older index (README.md, Setup), or pass --device cpu.")
    print("GPU:", torch.cuda.get_device_name(0))
    # is_available() does not notice a wheel without kernels for this GPU;
    # the first convolution does.
    torch.nn.Conv2d(1, 8, 3).cuda()(torch.randn(2, 1, 28, 28, device="cuda"))
EOF

step() { echo; echo "=== $* "; }

step "1/9 selftest (no data, no network)"
$PY selftest.py

step "2/9 download TissueMNIST"
$PY download_data.py

step "3/9 split: rotate every image once, secret / train / dev / test"
$PY make_split.py "$OUT/split"

step "4/9 label model on the secret set   (the slow step)"
$PY train_label_model.py "$OUT/model" --split "$OUT/split" --device "$DEVICE" \
    --epochs "$EPOCHS" ${TRAIN_EXTRA[@]+"${TRAIN_EXTRA[@]}"}

step "5/9 labels, locations, batches"
$PY make_data.py "$OUT/data" --split "$OUT/split" --model "$OUT/model" \
    ${DATA_EXTRA[@]+"${DATA_EXTRA[@]}"}

step "6/9 the Kaggle upload"
$PY make_package.py "$OUT/data" "$OUT/kaggle"

step "7/9 the metric notebook"
$PY make_metric_notebook.py metric-template.ipynb

step "8/9 the student bundle"
$PY make_student_bundle.py "$OUT/kaggle_code" --zip

step "9/9 baselines and the pre-launch checks   (several minutes)"
./run_baselines.sh "$OUT/kaggle/organiser" "$OUT/submissions"

# --- the pre-launch checks, asserted ----------------------------------------
step "verdict"
$PY - "$OUT" <<'EOF'
import sys, glob
from pathlib import Path
import pandas as pd

out = Path(sys.argv[1])
bad = []

def check(ok, msg):
    print(("ok    " if ok else "FAIL  ") + msg)
    if not ok:
        bad.append(msg)

# 1. the oracle scores exactly zero at every batch size
t = pd.read_csv(out / "submissions/Private/true_plugin_per_size.csv")
check(bool((t["reg_at_c"].abs() < 1e-12).all()),
      "true_plugin scores exactly 0 (generation and scoring agree)")

# 2. no clearly negative score
worst, where = 0.0, ""
for f in glob.glob(str(out / "submissions/Private/*_per_size.csv")):
    v = pd.read_csv(f)["reg_at_c"].min()
    if v < worst:
        worst, where = v, Path(f).stem
check(worst > -0.005, f"no clearly negative score (worst {worst:+.5f} {where})")

# 3. and 4. the two comparisons that decide whether the task has structure
scores = {}
for f in glob.glob(str(out / "submissions/Private/*_per_size.csv")):
    scores[Path(f).stem.replace("_per_size", "")] = \
        float(pd.read_csv(f)["reg_at_c"].mean())
for name, val in sorted(scores.items(), key=lambda kv: kv[1]):
    print(f"        {name:<26} {val:+.4f}")
em, uni = scores.get("bayes_epistemic_em"), scores.get("bayes_epistemic_uniform")
tot = scores.get("bayes_total_em")
if em is None or uni is None or tot is None:
    check(False, "the expected baselines are missing")
else:
    check(em < uni, f"estimating w pays: bayes_epistemic_em {em:+.4f} < "
                    f"uniform {uni:+.4f}")
    check(em < tot, f"epistemic beats total ranking: {em:+.4f} < {tot:+.4f}")

print("\n(point estimates on Private; the paired bootstrap intervals that "
      "decide\nchecks 3 and 4 are in the run_baselines.sh output above)")
if bad:
    sys.exit(f"\n{len(bad)} PRE-LAUNCH CHECK(S) FAILED -- do not launch")
print("\nall pre-launch checks passed")
EOF

cat <<NOTE

=== done. Upload to Kaggle, from $OUT/kaggle:
      train.csv train_images.npy test_batches.csv test_images.npy
      sample_submission.csv dev_test_batches.csv dev_images.npy
      dev_solution.csv dev_sample_submission.csv
    give Kaggle only : solution.csv, metric-template.ipynb
    starter kit      : $OUT/kaggle_code.zip
    never upload     : $OUT/kaggle/organiser/, $OUT/data, $OUT/model, $OUT/split
    location priors  : $OUT/data/priors.txt  (organiser only; header holds w)
    full log         : $LOG
NOTE
