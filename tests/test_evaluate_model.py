import unittest

import pandas as pd

from evaluate_model import evaluate_frame


def sample_frame(rows_per_class: int = 12) -> pd.DataFrame:
    rows = []
    for class_label, offset in (("L", 0), ("M", 10), ("H", 20)):
        for index in range(rows_per_class):
            rows.append(
                {
                    "participation": offset + index,
                    "resources": (offset + index) % 7,
                    "grade_band": ["lower", "middle", "upper"][offset // 10],
                    "Class": class_label,
                }
            )
    return pd.DataFrame(rows)


class ReproducibleEvaluationTests(unittest.TestCase):
    def test_metrics_are_reproducible_and_confusion_matrix_covers_test_rows(self):
        frame = sample_frame()
        first = evaluate_frame(frame)
        second = evaluate_frame(frame)

        self.assertEqual(first["split"], second["split"])
        self.assertEqual(first["models"], second["models"])
        self.assertEqual(first["dataset_rows"], len(frame))
        self.assertEqual(first["split"]["train_rows"] + first["split"]["test_rows"], len(frame))
        for model_name in ("majority_class_baseline", "random_forest"):
            metrics = first["models"][model_name]
            self.assertEqual(sum(map(sum, metrics["confusion_matrix"])), first["split"]["test_rows"])
            for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
                self.assertGreaterEqual(metrics[metric], 0.0)
                self.assertLessEqual(metrics[metric], 1.0)

    def test_target_is_not_used_as_a_predictor(self):
        result = evaluate_frame(sample_frame())
        self.assertNotIn("Class", result["predictors"])
        self.assertEqual(result["target_column"], "Class")

    def test_missing_target_column_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "target column"):
            evaluate_frame(pd.DataFrame({"feature": [1, 2, 3]}))

    def test_empty_dataset_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            evaluate_frame(pd.DataFrame(columns=["feature", "Class"]))

    def test_single_class_target_is_rejected(self):
        frame = pd.DataFrame({"feature": range(10), "Class": ["M"] * 10})
        with self.assertRaisesRegex(ValueError, "at least two"):
            evaluate_frame(frame)


if __name__ == "__main__":
    unittest.main()
