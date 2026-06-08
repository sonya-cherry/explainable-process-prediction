

"""
Tests for evaluation metric utilities.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.modeling.evaluation import (
    build_evaluation_table,
    compute_basic_metrics,
    compute_metrics_with_roc_auc,
    save_confusion_matrix_plot,
    save_model_comparison_plot,
    save_roc_curve,
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
