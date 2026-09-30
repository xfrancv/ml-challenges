## Competition Rules

*This is a course competition. There is no monetary prize; the reward is extra
course credit. Entering constitutes acceptance of these rules. Dates are on the
**Overview → Timeline** page.*

1. **Competition data only.** You may access and use the competition data solely
   for participating in this competition and on the competition's Kaggle forums.
   This is the central rule. You must not use any data or information other
   than what is published on this competition's Data page — in particular, and
   without limitation, not the TissueMNIST or MedMNIST datasets in any form, no
   other external dataset, and no model whose weights were trained, pre-trained
   or fine-tuned on anything other than the competition data.

2. **No recovering the test labels.** You may not attempt to find out the true
   class of any test image by any means other than predicting it from the
   competition data — in particular, not by matching test images against
   TissueMNIST, MedMNIST or any other source, whether exactly, by nearest
   neighbours, or after undoing rotations. (It would not help: the labels of
   this competition are not TissueMNIST's.)

3. **Each test batch is for predicting its own rows.** A test batch is the set
   of rows sharing an `id_test`, together with their images. You may use the
   whole of a batch to predict the rows in it — that is the task. You may not
   carry anything from one test batch to another, and you may not compute
   anything over the test set as a whole: no training, fine-tuning,
   statistics, pseudo-labelling, model selection or matching of images across
   batches, in any form. Anything you want to fit, tune or select, fit on the
   training and development data, which are provided for exactly that.

4. **Work independently.** Discuss freely; write and submit your own solution.

5. **Team size is 1.**

6. **Code and a short description are part of your entry.** With your final
   submissions you hand in the code that produced them, the trained weights,
   and a short description of what you did (a paragraph to a page, as a PDF).
   The code must run on the competition data as published and reproduce your
   submission. The winning entries and a random sample of the others are
   checked by hand: the code is read, and re-run on a differently drawn test
   set. An entry without code and description is not ranked.

7. **Breaking a rule disqualifies the entry.** If you are unsure whether
   something is allowed, ask on the forum before you rely on it. Asking is never
   penalised.

8. **Submissions.** At most 5 per day; you choose 2 to be scored finally.
