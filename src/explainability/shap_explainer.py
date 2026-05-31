from pathlib import Path
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

"""
SHAP explanation utilities for process outcome prediction models.

This module does not train models. It only explains an already trained model
using a feature matrix and aligned feature names.

Recommended input for this project:
- model: trained Random Forest or Gradient Boosting model
- X: no-leakage feature matrix, for example X_test_no_leakage
- feature_columns: matching feature names, for example feature_columns_no_leakage
"""


def prepare_shap_dataframe(
    X: Any,
    feature_columns: list[str],
    max_samples: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Convert a dense or sparse feature matrix into a pandas DataFrame for SHAP.

    Args:
        X: Feature matrix. Supports numpy arrays, pandas DataFrames, and sparse matrices.
        feature_columns: Feature names aligned with X.
        max_samples: Optional maximum number of rows to keep for SHAP runtime control.
        random_state: Random seed used when sampling rows.

    Returns:
        DataFrame with readable feature names.
    """
    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
    elif hasattr(X, "toarray"):
        X_df = pd.DataFrame(X.toarray(), columns=feature_columns)
    else:
        X_df = pd.DataFrame(X, columns=feature_columns)

    if X_df.shape[1] != len(feature_columns):
        raise ValueError(
            "Number of feature columns does not match the feature matrix. "
            f"Got {X_df.shape[1]} matrix columns and {len(feature_columns)} feature names."
        )

    if max_samples is not None and len(X_df) > max_samples:
        X_df = X_df.sample(n=max_samples, random_state=random_state)

    return X_df


def _get_positive_class_shap_values(shap_values: Any) -> np.ndarray:
    """
    Extract SHAP values for the positive class from different SHAP return formats.

    SHAP may return:
    - a list with one array per class
    - a 2D array with shape (n_samples, n_features)
    - a 3D array with shape (n_samples, n_features, n_classes)
    """
    if isinstance(shap_values, list):
        return shap_values[1]

    values = np.asarray(shap_values)

    if values.ndim == 3:
        return values[:, :, 1]

    return values


def _get_positive_class_base_value(explainer: Any) -> float:
    """Extract the expected value for the positive class if available."""
    expected_value = explainer.expected_value

    if isinstance(expected_value, (list, tuple, np.ndarray)):
        return float(expected_value[1])

    return float(expected_value)


def save_global_shap_summary(
    model: Any,
    X: Any,
    feature_columns: list[str],
    output_path: Union[str, Path] = "figures/explanations/shap_summary_sprint2.png",
    max_samples: int = 500,
) -> Path:
    """
    Create and save a global SHAP summary plot for a tree-based model.

    Args:
        model: Trained tree-based model, for example RandomForestClassifier.
        X: Feature matrix to explain.
        feature_columns: Feature names aligned with X.
        output_path: Destination path for the SHAP summary plot.
        max_samples: Maximum number of cases used for SHAP calculation.

    Returns:
        Path to the saved figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    X_df = prepare_shap_dataframe(
        X=X,
        feature_columns=feature_columns,
        max_samples=max_samples,
    )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)
    values_to_plot = _get_positive_class_shap_values(shap_values)

    shap.summary_plot(
        values_to_plot,
        X_df,
        show=False,
        max_display=20,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def save_global_shap_importance_table(
    model: Any,
    X: Any,
    feature_columns: list[str],
    output_path: Union[str, Path] = "reports/shap_feature_importance_sprint3.csv",
    max_samples: int = 500,
) -> pd.DataFrame:
    """
    Save a table with mean absolute SHAP importance per feature.

    Args:
        model: Trained tree-based model.
        X: Feature matrix to explain.
        feature_columns: Feature names aligned with X.
        output_path: Destination CSV path.
        max_samples: Maximum number of cases used for SHAP calculation.

    Returns:
        DataFrame sorted by mean absolute SHAP value.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    X_df = prepare_shap_dataframe(
        X=X,
        feature_columns=feature_columns,
        max_samples=max_samples,
    )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)
    positive_values = _get_positive_class_shap_values(shap_values)

    importance_df = pd.DataFrame(
        {
            "feature": X_df.columns,
            "mean_abs_shap": np.abs(positive_values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)

    importance_df.to_csv(output_path, index=False)
    return importance_df


def save_global_shap_importance_plot(
    model: Any,
    X: Any,
    feature_columns: list[str],
    output_path: Union[str, Path] = "figures/explanations/shap_importance_bar_sprint2.png",
    max_samples: int = 500,
    top_n: int = 15,
) -> Path:
    """
    Create and save a clean global SHAP feature-importance bar plot.

    This plot is easier to read than the standard SHAP beeswarm plot because it
    only shows the average absolute SHAP impact per feature.

    Args:
        model: Trained tree-based model.
        X: Feature matrix to explain.
        feature_columns: Feature names aligned with X.
        output_path: Destination path for the SHAP importance plot.
        max_samples: Maximum number of cases used for SHAP calculation.
        top_n: Number of most important features to display.

    Returns:
        Path to the saved figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    X_df = prepare_shap_dataframe(
        X=X,
        feature_columns=feature_columns,
        max_samples=max_samples,
    )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)
    positive_values = _get_positive_class_shap_values(shap_values)

    importance_df = pd.DataFrame(
        {
            "feature": X_df.columns,
            "mean_abs_shap": np.abs(positive_values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)

    top_features = importance_df.head(top_n).sort_values(
        "mean_abs_shap",
        ascending=True,
    )

    fig_height = max(5, 0.45 * len(top_features) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    bars = ax.barh(
        top_features["feature"],
        top_features["mean_abs_shap"],
    )

    ax.bar_label(
        bars,
        labels=[f"{value:.4f}" for value in top_features["mean_abs_shap"]],
        padding=4,
    )

    ax.set_title("Global SHAP Feature Importance")
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_ylabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    max_value = top_features["mean_abs_shap"].max()
    ax.set_xlim(0, max_value * 1.20 if max_value > 0 else 1)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path

def save_selected_local_explanations(
    model: Any,
    X: Any,
    feature_columns: list[str],
    y_true: Any,
    y_pred: Any,
    output_dir: Union[str, Path] = "figures/explanations",
) -> dict[str, Path]:
    """
    Save local SHAP explanations for representative prediction cases.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    indices = select_local_explanation_indices(y_true=y_true, y_pred=y_pred)
    saved_paths: dict[str, Path] = {}

    for case_type, index in indices.items():
        if index is None:
            continue

        saved_paths[case_type] = save_local_shap_bar_plot(
            model=model,
            X=X,
            feature_columns=feature_columns,
            row_index=index,
            output_path=output_dir / f"shap_local_{case_type}.png",
        )

    return saved_paths


def save_local_shap_bar_plot(
    model: Any,
    X: Any,
    feature_columns: list[str],
    row_index: int,
    output_path: Union[str, Path],
    max_display: int = 10,
) -> Path:
    """
    Create and save a local SHAP bar plot for one selected case.

    Args:
        model: Trained tree-based model.
        X: Feature matrix to explain.
        feature_columns: Feature names aligned with X.
        row_index: Row index of the case to explain.
        output_path: Destination path for the local explanation plot.
        max_display: Number of features shown in the local explanation.

    Returns:
        Path to the saved figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    X_df = prepare_shap_dataframe(X=X, feature_columns=feature_columns)

    if row_index < 0 or row_index >= len(X_df):
        raise IndexError(
            f"row_index {row_index} is outside the valid range 0 to {len(X_df) - 1}."
        )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)
    positive_values = _get_positive_class_shap_values(shap_values)
    base_value = _get_positive_class_base_value(explainer)

    explanation = shap.Explanation(
        values=positive_values[row_index],
        base_values=base_value,
        data=X_df.iloc[row_index].values,
        feature_names=list(X_df.columns),
    )

    shap.plots.bar(
        explanation,
        max_display=max_display,
        show=False,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def select_local_explanation_indices(y_true: Any, y_pred: Any) -> dict[str, Optional[int]]:
    """
    Select representative cases for local explanations.

    The function tries to select:
    - one correctly predicted positive case
    - one correctly predicted negative case
    - one misclassified case

    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels.

    Returns:
        Dictionary with row indices or None if a case type is not available.
    """
    y_true_series = pd.Series(y_true).reset_index(drop=True)
    y_pred_series = pd.Series(y_pred).reset_index(drop=True)

    correct_positive = y_true_series[
        (y_true_series == 1) & (y_pred_series == 1)
    ].index

    correct_negative = y_true_series[
        (y_true_series == 0) & (y_pred_series == 0)
    ].index

    misclassified = y_true_series[
        y_true_series != y_pred_series
    ].index

    return {
        "correct_positive": int(correct_positive[0]) if len(correct_positive) > 0 else None,
        "correct_negative": int(correct_negative[0]) if len(correct_negative) > 0 else None,
        "misclassified": int(misclassified[0]) if len(misclassified) > 0 else None,
    }

