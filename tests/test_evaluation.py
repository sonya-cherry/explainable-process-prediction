"""
Tests for evaluation metric utilities.
"""

from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.modeling.evaluation import (
    build_evaluation_table,
    compute_basic_metrics,
    compute_metrics_with_roc_auc,
    evaluate_model,
    save_confusion_matrix_plot,
    save_model_comparison_plot,
    save_roc_curve,
    save_roc_curves,
)


def test_compute_basic_metrics_returns_expected_values():
    """Evaluation metrics should return correct values on a small known example."""
    # Arrange
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 0, 0]

    # Act
    metrics = compute_basic_metrics(y_true, y_pred)

    # Assert
    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert round(metrics["f1"], 4) == 0.6667


def test_compute_metrics_with_roc_auc_returns_auc():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    y_score = [0.05, 0.20, 0.80, 0.95]

    metrics = compute_metrics_with_roc_auc(y_true, y_pred, y_score)

    assert metrics["roc_auc"] == 1.0


def test_save_confusion_matrix_and_roc_curve_create_files(tmp_path):
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    y_score = [0.10, 0.60, 0.70, 0.90]

    confusion_path = save_confusion_matrix_plot(
        y_true=y_true,
        y_pred=y_pred,
        model_name="toy_model",
        output_path=tmp_path / "confusion_matrix.png",
    )
    roc_path = save_roc_curve(
        y_true=y_true,
        y_score=y_score,
        model_name="toy_model",
        output_path=tmp_path / "roc_curve.png",
    )

    assert confusion_path.exists()
    assert roc_path.exists()


def test_build_evaluation_table_and_comparison_plot(tmp_path):
    table = build_evaluation_table(
        [
            {
                "model": "baseline",
                "accuracy": 0.5,
                "precision": 0.5,
                "recall": 1.0,
                "f1": 0.67,
                "roc_auc": 0.5,
            },
            {
                "model": "random_forest",
                "accuracy": 0.8,
                "precision": 0.75,
                "recall": 1.0,
                "f1": 0.86,
                "roc_auc": 0.9,
            },
        ]
    )

    plot_path = save_model_comparison_plot(
        evaluation_table=table,
        output_path=tmp_path / "model_comparison.png",
    )

    assert list(table.columns) == [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]
    assert plot_path.exists()


def test_evaluate_model_returns_all_metric_keys():
    """evaluate_model should return a dict with the model name and all metric keys."""
    X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0]])
    y = [0, 0, 1, 1]
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)

    result = evaluate_model(model, X, y, "rf_model")

    assert result["model"] == "rf_model"
    assert set(result.keys()) == {"model", "accuracy", "f1", "precision", "recall", "roc_auc"}
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["roc_auc"] <= 1.0


def test_evaluate_model_falls_back_to_predictions_without_predict_proba():
    """evaluate_model should use predictions as the score when predict_proba is absent."""
    class _ThresholdClassifier:
        def predict(self, X):
            return np.array([1 if row[0] > 1.5 else 0 for row in X])

    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = [0, 0, 1, 1]
    model = _ThresholdClassifier()

    result = evaluate_model(model, X, y, "threshold_model")

    assert result["accuracy"] == 1.0
    assert result["roc_auc"] == 1.0


def test_save_roc_curves_creates_file_for_multiple_models(tmp_path):
    """save_roc_curves should produce one figure containing curves for all given models."""
    y_true = [0, 0, 1, 1]
    model_scores = {
        "Model A": np.array([0.1, 0.2, 0.8, 0.9]),
        "Model B": np.array([0.2, 0.3, 0.7, 0.85]),
    }

    output_path = save_roc_curves(
        model_scores=model_scores,
        y_true=y_true,
        output_path=tmp_path / "roc_curves.png",
    )

    assert output_path.exists()
