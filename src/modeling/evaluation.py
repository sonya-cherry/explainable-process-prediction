
from pathlib import Path

from typing import Union

import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)



"""
Evaluation utilities for process outcome prediction models.

This module provides basic classification metrics, model comparison tables,
and Sprint 2 evaluation visualizations, including ROC curve and confusion
matrix plots.
"""







def compute_basic_metrics(y_true, y_pred) -> dict[str, float]:
    """
    Compute basic binary classification metrics.

    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels.

    Returns:
        Dictionary containing accuracy, F1-score, precision, and recall.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }



def compute_metrics_with_roc_auc(y_true, y_pred, y_score) -> dict[str, float]:
    """
    Compute binary classification metrics including ROC AUC.

    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels.
        y_score: Predicted probabilities or decision scores for the positive class.

    Returns:
        Dictionary containing basic metrics and ROC AUC.
    """
    metrics = compute_basic_metrics(y_true, y_pred)
    metrics["roc_auc"] = roc_auc_score(y_true, y_score)
    return metrics


# --- Model evaluation and comparison table utilities ---

def evaluate_model(model, X, y, model_name: str) -> dict:
    """
    Evaluate a fitted classification model.

    Args:
        model: Fitted sklearn-compatible classifier.
        X: Feature matrix used for evaluation.
        y: True binary labels.
        model_name: Human-readable model name for the result table.

    Returns:
        Dictionary containing the model name and evaluation metrics.
    """
    y_pred = model.predict(X)

    # ROC AUC should be computed from probabilities when available.
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:, 1]
    else:
        y_score = y_pred

    metrics = compute_metrics_with_roc_auc(y, y_pred, y_score)
    metrics["model"] = model_name
    return metrics


def build_evaluation_table(results: list[dict]) -> pd.DataFrame:
    """
    Build a model comparison table from metric dictionaries.

    Args:
        results: List of dictionaries produced by evaluate_model.

    Returns:
        DataFrame with one row per model and standard Sprint 2 metrics.
    """
    columns = ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
    return pd.DataFrame(results)[columns]


def save_roc_curve(
    y_true,
    y_score,
    model_name: str = "model",
    output_path: Union[str, Path] = "figures/roc_curve_sprint2.png",
) -> Path:
    """
    Create and save a ROC curve figure.

    Args:
        y_true: True binary labels.
        y_score: Predicted probabilities or decision scores for the positive class.
        model_name: Name of the evaluated model.
        output_path: Destination path for the ROC curve figure.

    Returns:
        Path to the saved figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"{model_name} (AUC = {roc_auc:.3f})",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")

    ax.set_title("ROC Curve - Sprint 2")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


def save_confusion_matrix_plot(
    y_true,
    y_pred,
    model_name: str = "model",
    output_path: Union[str, Path] = "figures/confusion_matrix_sprint2.png",
) -> Path:
    """
    Create and save a confusion matrix plot.

    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels.
        model_name: Name of the evaluated model.
        output_path: Destination path for the confusion matrix figure.

    Returns:
        Path to the saved figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Class 0: negative", "Class 1: positive"],
    )
    display.plot(ax=ax, values_format="d", colorbar=False)

    ax.set_title(f"Confusion Matrix - {model_name}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path

