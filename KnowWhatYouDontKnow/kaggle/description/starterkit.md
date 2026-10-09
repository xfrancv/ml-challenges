The starter kit is optional. It loads the images and tables, writes a
submission in the right format, and — most usefully — lets you **score yourself
offline** on the development batches before spending a submission.

## Contents

| File | What it is |
| :-- | :-- |
| `competition_data.py` | loads the images and tables into numpy / pandas |
| `make_sample_submission.py` | writes a valid submission; replace its `predict_batch` |
| `metric.py` | the competition metric, the same code the leaderboard runs |
| `evaluate.py` | scores a submission against `dev_solution.csv` |

## Install

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The kit needs numpy and pandas; matplotlib only for `evaluate.py --plot`.
Training a model on the images is up to you, with any framework you like. On a
Kaggle notebook everything is preinstalled.

## Use

Point `DATA` at the competition data:

```
DATA=/kaggle/input/<competition>
```

**Write the sample submission and score it.** It takes seconds and proves the
whole loop works:

```
python make_sample_submission.py --data-dir $DATA --split dev --out dev_submission.csv
python evaluate.py dev_submission.csv $DATA/dev_solution.csv
```

You get a per-batch-size table and one number: what the leaderboard would give
this submission on those rows.

**Add your predictor.** `make_sample_submission.py` calls
`predict_batch(images, ...)` once per batch, with all of that batch's images.
Replace it and rerun.

**Produce a submission.** Same predictor, on the test batches:

```
python make_sample_submission.py --data-dir $DATA --split test --out submission.csv
```

## How the files join up

Row *k* of every CSV describes image *k* of the matching `.npy`:
`train.csv` ↔ `train_images.npy`, `dev_test_batches.csv` ↔ `dev_images.npy`,
`test_batches.csv` ↔ `test_images.npy`. Submissions are keyed on `row_id`.
Rows sharing an `id_test` form one batch and come from the same location. Rows
in different batches are independent; see the rules for what you may not do
with them.
