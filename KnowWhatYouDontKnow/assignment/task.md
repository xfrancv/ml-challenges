# Goal

Create the best possible submission for the challenge on label prior shift adaptation with a reject option.

# Context

Read these before starting:

- `description/overview.md`: the challenge overview.
- `description/description.md`: the problem formulation.
- `description/evaluation.md`: the evaluation metric.
- `description/dataset_description.md`: the data files and their columns.
- `description/rules.md`: the challenge rules.
- `description/starterkit.md`: the starter kit. Its code is in `start_kit/`.

Earlier solutions are stored in `solutions/solutionN/`, where N is the solution ID. Read their descriptions and dev scores. They are the baseline to beat.

# Task

Implement an algorithm that solves the challenge and scores better on the development data than the existing solutions. Deliver two scripts:

1. **`make_submission.py`** reads the challenge data and writes a submission file in the challenge format (`row_id,pred,confidence`, one row for each of the 126,000 rows in `test_batches.csv`), ready to upload to Kaggle. Example usage:
   ```
   python make_submission.py --data-dir ../../data --out submission.csv
   ```

2. **`evaluate_on_dev_data.py`** runs the same algorithm on the development batches (`dev_test_batches.csv`, `dev_predictions.csv`) and scores it against `dev_solution.csv` with the official metric. Example usage:
   ```
   python evaluate_on_dev_data.py --data-dir ../../data
   ```
   Model it on `start_kit/evaluate.py`, and compute the score with `start_kit/metric.py` without changing that file. It writes two outputs to the solution folder:
   1. a text file with `AvgRegAtCoverage` and the per-batch-size breakdown, as printed by `evaluate.py`;
   2. the regret-coverage figure that `evaluate.py --plot` produces.

Both scripts must run the same prediction code, so the dev score is a faithful estimate of the test score.

# Rules

- First, create the output folder `solutions/solutionM/`, where M is the smallest unused solution ID. If the folder `solutions/` is empty, set M=1.
- Create or modify files only inside `solutions/solutionM/`. Do not touch the data, the starter kit or other solutions.
- Follow the challenge rules (see `rules.md` and `CLAUDE.md`). In particular:
  - Each test batch may use only its own rows.
  - Fit or tune only on `dev.csv` or the dev batches, never on the test set.
  - Use no external data, and hardcode no row or batch ids.
- Write `solutions/solutionM/DESCRIPTION.md` with a short summary of the algorithm, the dev score, and a comparison against the earlier solutions.
