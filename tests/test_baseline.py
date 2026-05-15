"""
Tests for baseline model utilities.
"""

from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csr_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.modeling.baseline import train_majority_baseline


def test_majority_baseline_predicts_most_frequent_class():
    """Majority baseline should always predict the most frequent training label."""
    # Arrange
    X_train = csr_matrix([[0], [1], [2], [3]])
    y_train = np.array([1, 1, 1, 0])

    # Act
    model = train_majority_baseline(X_train, y_train)
    predictions = model.predict(X_train)

    # Assert
    assert len(predictions) == len(y_train)
    assert set(predictions) == {1}
