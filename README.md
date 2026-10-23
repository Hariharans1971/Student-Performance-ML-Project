# Student Performance Prediction — Machine Learning

A beginner-friendly machine-learning project exploring the UCI-style student performance dataset with a reproducible evaluation pipeline and a separate interactive exploration script.

## Problem

The dataset records student participation, school context, and an outcome label (`Class`). This project explores associations in that dataset; it does not establish that any feature causes an academic outcome.

## What is included

- `Project.py`: legacy interactive visual exploration and model-comparison walkthrough.
- `evaluate_model.py`: non-interactive, reproducible evaluation using all non-target columns.
- `tests/test_evaluate_model.py`: tests for reproducibility, metric bounds, confusion-matrix totals, target separation, and invalid datasets.
- GitHub Actions: syntax validation, automated tests, and an evaluation run against the tracked CSV.

## Evaluation design

The reproducible pipeline:
1. Validates the dataset and separates `Class` from the predictor columns.
2. Uses a **stratified 75%/25% train/test split** with random seed `42`.
3. Fits categorical one-hot encoding inside a pipeline, so preprocessing is learned from the training split.
4. Compares a majority-class baseline with a Random Forest classifier using the same split.
5. Reports accuracy, macro precision, macro recall, macro F1, confusion matrices, class counts, and the Random Forest's metric difference from baseline.

Run it locally:

```bash
python -m pip install -r requirements.txt
python evaluate_model.py
```

Save a JSON report when you need to attach the exact results to a portfolio review:

```bash
python evaluate_model.py --json-out evaluation_results.json
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

The benchmark output produced by the command and CI run is the source of the actual metric values; the repository does not hard-code an unverified accuracy claim.

## Limitations

- Metrics describe this dataset and one fixed holdout split; they are not a guarantee of generalization.
- No repeated cross-validation or external validation is included yet.
- The dataset is observational; results do not demonstrate causation.
- This is an educational experiment and must not be used to make consequential decisions about individual students.
- `Project.py` remains an interactive exploratory script. The reproducible evaluation path is `evaluate_model.py`.

## Next steps

- Add repeated stratified cross-validation and per-class error analysis.
- Record a versioned evaluation report with dataset provenance and checksum.
- Compare a calibrated model against the same baseline on a separately held-out dataset.

## Author

Hariharan S. — B.Tech Computer Science and Engineering (AI & ML), 2026
