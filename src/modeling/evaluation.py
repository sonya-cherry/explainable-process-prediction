from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

"""
Evaluation utilities for process outcome prediction models.
"""


def compute_basic_metrics(y_true, y_pred) -> dict[str, float]:
    """
    Compute binary classification metrics for binary outcome prediction.
    """
    metrics = dict()

    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)

    return metrics
