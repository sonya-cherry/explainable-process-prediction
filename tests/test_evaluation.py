

"""
Tests for evaluation metric utilities.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.modeling.evaluation import compute_basic_metrics


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