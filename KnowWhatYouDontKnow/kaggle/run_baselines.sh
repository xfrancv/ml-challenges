#!/usr/bin/env bash
# Compute every baseline and score it -- the pre-launch check that decides
# whether the competition has discoverable structure at all.
#
# Two comparisons matter, both on shared bootstrap resamples
# (compare_baselines.py), because two independently bootstrapped marginal
# intervals are the wrong test:
#   * bayes_total vs bayes_epistemic -- same predictor, different ranking;
#   * bayes_epistemic under a uniform location prior vs under the EM estimate
#     of w from the development batches -- the "does EM pay?" check.
# The runs under the true w show how much of the gap EM closes.
#
#   ./run_baselines.sh                               # out/final/kaggle/organiser
#   ./run_baselines.sh out/smoke/kaggle/organiser out/smoke/submissions
set -euo pipefail
cd "$(dirname "$0")"

ORG=${1:-out/final/kaggle/organiser}
SUBS=${2:-out/final/submissions}
PY=${PYTHON:-python}
command -v "$PY" >/dev/null || {
  echo "no '$PY' on PATH -- activate the virtualenv first (README.md, Setup)" >&2
  exit 1
}
BAYES=(map_plugin bayes_total bayes_epistemic bayes_aleatoric)

mkdir -p "$SUBS"
$PY estimate_location_prior.py "$ORG/dev" "$SUBS/em_location_prior.csv"

$PY baseline_solutions.py "$ORG/test" "$SUBS" --only base true_plugin
$PY baseline_solutions.py "$ORG/test" "$SUBS" --only "${BAYES[@]}" \
    --location-prior uniform --suffix _uniform
$PY baseline_solutions.py "$ORG/test" "$SUBS" --only "${BAYES[@]}" \
    --location-prior "$SUBS/em_location_prior.csv" --suffix _em
$PY baseline_solutions.py "$ORG/test" "$SUBS" --only "${BAYES[@]}" \
    --location-prior true --suffix _true

# The submissions are the .csv files in $SUBS, minus the per-size tables that
# evaluate.py writes and the location-prior estimate: those have no row_id.
shopt -s nullglob
SUBMISSIONS=()
for f in "$SUBS"/*.csv; do
  [[ $f == *_per_size.csv || $f == */em_location_prior.csv ]] || SUBMISSIONS+=("$f")
done

for usage in Public Private; do
  echo
  echo "################################################################"
  echo "# $usage -- paired comparison against bayes_epistemic_uniform"
  echo "################################################################"
  $PY compare_baselines.py "$ORG/test/solution.csv" "${SUBMISSIONS[@]}" \
      --usage "$usage" --vs bayes_epistemic_uniform
done

echo
echo "################################################################"
echo "# per-submission detail and figures (Private)"
echo "################################################################"
for f in "${SUBMISSIONS[@]}"; do
  $PY evaluate.py "$f" "$ORG/test/solution.csv" --usage Private \
      --out-dir "$SUBS/Private"
done

cat <<'NOTE'

Read the output in this order:
  1. true_plugin must be exactly +0.000000. Anything else means generation and
     scoring have diverged; stop and fix that before reading anything else.
  2. Every other score must be >= 0 up to noise: the reference is the Bayes
     predictor given the location. A clearly negative score is a bug.
  3. bayes_epistemic_em vs bayes_epistemic_uniform: the paired interval must
     exclude 0, or the secret location prior is not worth discovering and w
     should be moved further from uniform.
  4. bayes_epistemic vs bayes_total (same prior): the paired interval must
     exclude 0 too, or ranking by epistemic uncertainty is not rewarded.
NOTE
