A classifier is trained on images collected at several sites and then used on
new images from those same sites. The images all come from the same kind of
device, but the sites are not alike: some kinds of cells are common at one site
and rare at another. And when new images arrive, nobody tells you which site
sent them.

In this competition you get labeled training images of kidney tissue from **9
locations**, each image tagged with where it was collected. The test images
arrive in **batches**. Every batch comes from one of the 9 locations, but you
are not told which. A batch of 100 images says a lot about where it came from;
a batch of 1 says almost nothing.

**Goal:** for every test image, predict its class **and** attach a confidence
score. We keep the 80 % of your predictions you were most confident about, and
measure how much worse they are than the best possible predictor that knows
the location each batch came from. Lower is better.

The interesting part is the confidence, not only the labels. Two images can be
equally hard to classify while the batches they arrived in differ completely in
how well they reveal their location, and the scoring is built so that noticing
the difference pays.
