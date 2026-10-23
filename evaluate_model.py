"""Reproducible, non-interactive evaluation for the student performance dataset.

Run from any working directory:
    python evaluate_model.py
    python evaluate_model.py --json-out evaluation_results.json

The evaluation uses a fixed stratified holdout split, a majority-class baseline,
and a Random Forest pipeline. Results are dataset-specific and should not be
interpreted as evidence that the model is suitable for real educational decisions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = ROOT / "AI-Data.csv"
TARGET_COLUMN = "Class"
RANDOM_STATE = 42
TEST_SIZE = 0.25


def _validate_frame(frame: pd.DataFrame, target_column: str = TARGET_COLUMN) -> tuple[pd.DataFrame, pd.Series]:
    """Validate the input schema and return predictors and a clean target."""
    if frame.empty:
        raise ValueError("The input dataset is empty.")
    if target_column not in frame.columns:
        raise ValueError(f"Required target column {target_column!r} was not found.")
    if frame.columns.duplicated().any():
        duplicate_names = sorted(set(frame.columns[frame.columns.duplicated()].astype(str)))
        raise ValueError(f"Duplicate column names are not supported: {duplicate_names}")

    target = frame[target_column].astype("string").str.strip()
    if target.isna().any() or target.eq("").any():
        raise ValueError(f"Target column {target_column!r} contains missing or blank labels.")

    predictors = frame.drop(columns=[target_column]).copy()
    if predictors.shape[1] == 0:
        raise ValueError("At least one predictor column is required.")
    if predictors.isna().all(axis=None):
        raise ValueError("All predictor values are missing.")

    return predictors, target.astype(str)


def _make_preprocessor(predictors: pd.DataFrame) -> ColumnTransformer:
    """Build transformations from training-set column types."""
    numeric_columns = predictors.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in predictors.columns if column not in numeric_columns]
    transformers: list[tuple[str, Any, list[str]]] = []

    if numeric_columns:
        transformers.append(("numeric", "passthrough", numeric_columns))
    if categorical_columns:
        transformers.append(
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _pipeline(predictors: pd.DataFrame, model: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", _make_preprocessor(predictors)),
            ("classifier", model),
        ]
    )


def _model_metrics(model: Pipeline, x_train: pd.DataFrame, x_test: pd.DataFrame,
                   y_train: pd.Series, y_test: pd.Series, labels: list[str]) -> dict[str, Any]:
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "macro_precision": round(float(precision_score(y_test, predictions, labels=labels, average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(y_test, predictions, labels=labels, average="macro", zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_test, predictions, labels=labels, average="macro", zero_division=0)), 4),
        "confusion_matrix_labels": labels,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
    }


def evaluate_frame(frame: pd.DataFrame, target_column: str = TARGET_COLUMN) -> dict[str, Any]:
    """Evaluate a baseline and Random Forest on the same fixed stratified split."""
    predictors, target = _validate_frame(frame, target_column=target_column)
    labels = sorted(target.unique().tolist())
    if len(labels) < 2:
        raise ValueError("Classification evaluation requires at least two target classes.")

    x_train, x_test, y_train, y_test = train_test_split(
        predictors,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    baseline = _pipeline(predictors, DummyClassifier(strategy="most_frequent"))
    forest = _pipeline(
        predictors,
        RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced_subsample",
            n_jobs=1,
        ),
    )

    baseline_metrics = _model_metrics(baseline, x_train, x_test, y_train, y_test, labels)
    forest_metrics = _model_metrics(forest, x_train, x_test, y_train, y_test, labels)
    return {
        "dataset_rows": int(len(frame)),
        "predictor_count": int(predictors.shape[1]),
        "predictors": predictors.columns.astype(str).tolist(),
        "target_column": target_column,
        "class_distribution": {
            str(key): int(value) for key, value in target.value_counts().sort_index().items()
        },
        "split": {
            "method": "stratified train_test_split",
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
        },
        "models": {
            "majority_class_baseline": baseline_metrics,
            "random_forest": forest_metrics,
        },
        "random_forest_minus_baseline": {
            metric: round(forest_metrics[metric] - baseline_metrics[metric], 4)
            for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1")
        },
        "limitations": [
            "Metrics apply only to this dataset and fixed holdout split.",
            "The dataset is observational; results do not establish causation.",
            "Do not use this educational model to make consequential decisions about individual students.",
            "No external validation or repeated cross-validation is performed by this script.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate student performance classifiers reproducibly.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="CSV dataset path (default: AI-Data.csv beside this script).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional output path for the full JSON metrics report.",
    )
    args = parser.parse_args()

    if not args.data.is_file():
        parser.error(f"Dataset file not found: {args.data}")

    result = evaluate_frame(pd.read_csv(args.data))
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
