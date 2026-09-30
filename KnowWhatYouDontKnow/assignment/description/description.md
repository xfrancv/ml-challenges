## The setting

The images are 28×28 greyscale pictures of human kidney cortex cells, in 8
classes. They were collected at $L = 9$ locations — think of 9 hospitals or
labs. Every location captured its images with the same kind of device, so a
cell of a given class looks the same wherever it was imaged. What differs is
**how often each class occurs**: the class frequencies can vary considerably
from one location to another.

### Training data

`train.csv` and `train_images.npy` hold 104,243 labeled images. Each carries its
class and the **location** it was collected at, `0`–`8`. The locations
contributed very different numbers of images.

### Test data

The test images arrive in **batches** of $m \in \{1, 2, 5, 10, 20, 50, 100\}$
images. All images of one batch come from **the same location**, which you are
not told. The location of each batch was drawn at random, independently of all
other batches, from a fixed distribution over the 9 locations. That
distribution is not disclosed.

Different batches are independent: knowing where one batch came from tells you
nothing about the next.

### Development data

`dev_test_batches.csv` holds 1,500 further batches, built by exactly the same
procedure — including the same distribution over locations — from images that
are in neither the training data nor the test batches. For these you get every
image's true class **and** the reference predictor's prediction (see
**Evaluation**), so you can score yourself offline exactly as the leaderboard
will.

## What you submit

For **every** test image, a predicted class `0`–`7` and a confidence. The
leaderboard keeps the 80 % of your predictions you were most confident about,
within each batch size, and compares them with the reference predictor. See
**Evaluation**.

## Why the batch matter

You may use all images of a batch to predict any image in it — including the
image you are currently classifying. Nothing in the format prevents it and
nothing in the scoring penalises it.

How much a batch reveals varies across the competition. A batch of 100 images
says a great deal about where it came from. A batch of 1 says almost nothing.
Most of the difficulty sits in between, and your confidence should reflect it.
