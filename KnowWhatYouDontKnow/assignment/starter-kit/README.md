# Starter code — *Know what you don't know: nine locations*

Everything here is optional. The kit loads the competition data, writes a
submission in the right format, and — the useful part — **scores you offline**
on the development batches before you spend a submission.

## The task in one paragraph

Images of kidney tissue cells were collected at 9 locations, all with the same
device. You get labeled training images, each tagged with its location, and
test images arriving in **batches**: every batch comes from one of the 9
locations, drawn at random from a fixed distribution that is not disclosed, and
you are not told which location a batch came from. For each test image you
submit a label and a confidence. The organiser pools all images of the same
batch size, sorts by descending confidence, keeps the most confident 80 %, and
measures how much more often your kept labels are wrong than those of the best
possible predictor that knows the batch's location. Lower is better; nobody
reaches zero.

## Files

| File | What it is |
| :-- | :-- |
| `competition_data.py` | loads the images and tables into numpy / pandas |
| `make_sample_submission.py` | writes a valid submission; replace its `predict_batch` |
| `metric.py` | the competition metric, the same code the leaderboard runs |
| `evaluate.py` | scores a submission against `dev_solution.csv` |

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The kit needs numpy and pandas; matplotlib only for `evaluate.py --plot`.
Training a model on the images is up to you, with any framework you like. On a
Kaggle notebook everything is preinstalled.

## Use

Set `DATA` to wherever the competition data lives:

```bash
DATA=../competition-data          # or /kaggle/input/<competition>
```

**1. Write the sample submission and score it.** It takes seconds and proves
the whole loop works:

```bash
python make_sample_submission.py --data-dir $DATA --split dev --out dev_submission.csv
python evaluate.py dev_submission.csv $DATA/dev_solution.csv
```

You get a per-batch-size table and one number at the bottom — what the
leaderboard would give this submission on those rows. It predicts one class for
everything, so it scores badly; beating it by a lot is the easy part.

**2. Add your predictor.** `make_sample_submission.py` calls
`predict_batch(images, ...)` once per batch with all of that batch's images.
Replace it, rerun step 1, and compare.

```bash
python evaluate.py dev_submission.csv $DATA/dev_solution.csv --plot
```

**3. Submit.** Same predictor, on the test batches:

```bash
python make_sample_submission.py --data-dir $DATA --split test --out submission.csv
```

## Loading the data

```python
from competition_data import load_training, load_batches, batch_slices

X, labels, locations = load_training(DATA)       # (n, 28, 28) uint8, (n,), (n,)
X_dev, dev_rows = load_batches(DATA, "dev")      # rows: row_id, id_test, slot, m,
                                                 #       label, pred_ref
X_test, test_rows = load_batches(DATA, "test")   # rows: row_id, id_test, slot, m

for id_test, sl in batch_slices(test_rows):      # one batch at a time
    batch_images = X_test[sl]
```

Row *k* of every CSV describes image *k* of the matching `.npy`. Every image
was randomly rotated by a multiple of 90°; this holds for training,
development and test images alike. The labels are synthetic: they are not
TissueMNIST's.

## Scoring yourself: what you can and cannot do

* `test_batches.csv` — the real test set. You submit predictions for these; the
  labels are not in the data.
* `dev_test_batches.csv` — batches built by the identical procedure, with the
  same distribution over locations, from other images; their labels and
  reference predictions are in `dev_solution.csv`.
  `evaluate.py` scores them exactly as the leaderboard would.

The development set is smaller than the test set, so its numbers are noisier —
do not read small differences as real.

## Four things that cost people points

**The confidence column is a ranking, not a probability.** Any strictly
increasing transformation of it gives exactly the same score. What it has to do
is put your *likely-wrong* predictions at the bottom.

**Confidences are pooled across batches.** Within one batch size, every image
from every batch goes into one ranking. Normalising confidences within a batch
throws away the difference between batches.

**Ties are broken by ascending `row_id`.** A constant confidence column is
scored deterministically, and badly.

**You are not asked to reject anything.** Submit a label for every row; the
organiser sweeps the coverage. There is no way to abstain in the file format.
