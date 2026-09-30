The images are 28×28 greyscale pictures of human kidney cortex cells in 8
tissue classes, derived from TissueMNIST (MedMNIST v2). They were collected at
9 locations; see **Description**.

For every row of `test_batches.csv` you predict a class `0`–`7` and a
confidence; see **Evaluation** for the submission format.

## How the images and the CSV files fit together

The images are stored in three NumPy files. Each is an array of shape
`(n, 28, 28)` and dtype `uint8`, and each has exactly one CSV file that
describes it, with exactly `n` data rows:

| Images | Described by | Rows |
| :-- | :-- | --: |
| `train_images.npy` | `train.csv` | 104,243 |
| `dev_images.npy` | `dev_test_batches.csv` | 28,750 |
| `test_images.npy` | `test_batches.csv` | 126,000 |

**The link is the position.** The *k*-th data row of the CSV (counting from 0,
not counting the header line) describes image `images[k]` of its `.npy` file.
For example, the first data row of `test_batches.csv` describes
`test_images[0]`, and the last one describes `test_images[125999]`. Do not
re-sort a CSV before pairing it with its images, or the pairing is lost.

**`row_id` is not an image index.** It is an arbitrary identifier, for example
`1000769`, and does not say where the image is stored. Use it only to join
the per-image tables with each other: your submission, `dev_solution.csv` and
the batch listings are all keyed by `row_id`.

```python
import numpy as np
import pandas as pd

train = pd.read_csv("train.csv")               # row k: label, location
X_train = np.load("train_images.npy")          # X_train[k]: that row's image

test = pd.read_csv("test_batches.csv")         # row k: row_id, id_test, slot, m
X_test = np.load("test_images.npy")            # X_test[k]: that row's image
test["image_index"] = np.arange(len(test))     # keep the position before any
                                               # sorting or merging

dev = pd.read_csv("dev_test_batches.csv")
X_dev = np.load("dev_images.npy")
dev["image_index"] = np.arange(len(dev))
# labels and reference predictions: join on row_id, never by position
dev = dev.merge(pd.read_csv("dev_solution.csv")[["row_id", "label", "pred_ref"]],
                on="row_id")

# all images of one test batch
rows = test[test["id_test"] == test["id_test"].iloc[0]]
batch_images = X_test[rows["image_index"].to_numpy()]
```

The same chain, written out:

*   a **training image** `X_train[k]` has the class `train.csv` row *k*
    `label` and the location `train.csv` row *k* `location`;
*   a **test image** `X_test[k]` has the identifier `row_id`, belongs to the
    batch `id_test`, and that batch has `m` images — all three given in row
    *k* of `test_batches.csv`;
*   a **development image** `X_dev[k]` is described the same way by row *k* of
    `dev_test_batches.csv`; its true class and the reference prediction are in
    `dev_solution.csv`, joined on `row_id`;
*   a **submission** has one row per `row_id` of `test_batches.csv`, in any
    order.

`competition_data.py` in the starter kit does all of this for you:
`load_training()` returns the training images with their labels and locations,
and `load_batches()` returns the images of `"dev"` or `"test"` with a table
aligned to them row for row, sorted by batch.

## Files

### Training data

*   **`train_images.npy`** — 104,243 images, array `(104243, 28, 28)`.
*   **`train.csv`** — one row per training image: its `label` and `location`.
    Row *k* belongs to `train_images[k]`.

### Test set

*   **`test_batches.csv`** — one row per test image; 126,000 rows in 12,600
    batches. Says which batch each image belongs to and how large that batch
    is. Row *k* belongs to `test_images[k]`.
*   **`test_images.npy`** — the test images, array `(126000, 28, 28)`, in the
    row order of `test_batches.csv`.
*   **`sample_submission.csv`** — a valid submission that predicts the most
    frequent class for everything. It shows the format; it is not a serious
    baseline.

| batch size `m` | 1 | 2 | 5 | 10 | 20 | 50 | 100 |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| batches | 6,000 | 3,000 | 1,200 | 600 | 600 | 600 | 600 |
| images | 6,000 | 6,000 | 6,000 | 6,000 | 12,000 | 30,000 | 60,000 |

### Development set, for scoring yourself offline

Built by the identical procedure, from images that appear neither in the
training data nor in any test batch.

*   **`dev_test_batches.csv`** — one row per development image; 28,750 rows in
    1,500 batches, with the same batch sizes as the test set. Row *k* belongs
    to `dev_images[k]`.
*   **`dev_images.npy`** — the development images, array `(28750, 28, 28)`, in
    the row order of `dev_test_batches.csv`.
*   **`dev_solution.csv`** — the answers for those rows: the true class and the
    reference predictor's prediction. Join it to `dev_test_batches.csv` on
    `row_id`.
*   **`dev_sample_submission.csv`** — the sample submission, on the development
    batches.

## Columns

### `train.csv`

*   `label` — the true class, `0`–`7`.
*   `location` — where the image was collected, `0`–`8`.

### `test_batches.csv`, `dev_test_batches.csv`

*   `row_id` — identifies one image; globally unique. **This is the key your
    submission must use.** It is not a position in the `.npy` file: the image
    of a row is found by the row's position, as described above.
*   `id_test` — which batch the image belongs to. All images sharing an
    `id_test` come from the same location.
*   `slot` — position within the batch, `0` to `m-1`. The order is arbitrary and
    carries no information.
*   `m` — how many images the batch contains, repeated on every row of the
    batch. A batch's rows are the rows sharing its `id_test`; there are
    exactly `m` of them.

### `sample_submission.csv`, `dev_sample_submission.csv`

*   `row_id` — one row for every image.
*   `pred` — predicted class, integer `0`–`7`.
*   `confidence` — any finite real number; larger means "keep this prediction".

### `dev_solution.csv`

*   `row_id`, `id_test`, `slot`, `m` — as in `dev_test_batches.csv`.
*   `label` — the true class.
*   `pred_ref` — the class predicted by the **reference predictor** for the
    batch's location. The metric measures how much more often your `pred` is
    wrong than this column is.

## Classes

| label | tissue |
| --: | :-- |
| 0 | Collecting Duct, Connecting Tubule |
| 1 | Distal Convoluted Tubule |
| 2 | Glomerular endothelial cells |
| 3 | Interstitial endothelial cells |
| 4 | Leukocytes |
| 5 | Podocytes |
| 6 | Proximal Tubule Segments |
| 7 | Thick Ascending Limb |

Some classes are close to indistinguishable at 28×28. Some error is
irreducible, which is why the metric compares you against a reference predictor
rather than against perfection.

## A note on the images

Every image was rotated once by a random multiple of 90°; training,
development and test images alike. The same image can occur in more than one
batch, or twice in one batch. Each occurrence is a separate row whose label was
drawn independently, so two copies of an image need not carry the same class.
Rows of different batches are independent, and the rules forbid linking them.

### Dataset attribution

The images of this challenge are derived from the TissueMNIST dataset from
MedMNIST v2. **The labels are synthetic**: every class label, in the training,
development and test data alike, was generated by the organisers from a model
of the tissue classes. They are not TissueMNIST's labels.

TissueMNIST is licensed under the Creative Commons Attribution 4.0
International License (CC BY 4.0).

TissueMNIST is derived from BBBC051 from the Broad Bioimage Benchmark
Collection (BBBC), originally released under CC BY 3.0.

References:

J. Yang et al., "MedMNIST v2: A Large-Scale Lightweight Benchmark for
2D and 3D Biomedical Image Classification," Scientific Data, 2023.

A. Woloshuk et al., "In Situ Classification of Cell Types in Human Kidney
Tissue Using 3D Nuclear Staining," Cytometry, 2020.
