from pathlib import Path
import sys

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.explainability.shap_explainer import (
    prepare_shap_dataframe,
    save_global_shap_importance_plot,
    save_global_shap_importance_table,
    save_global_shap_summary,
    save_local_shap_bar_plot,
    select_local_explanation_indices,
)


def _toy_tree_model():
    X = pd.DataFrame(
        {
            "urgency": [0, 0, 1, 1, 2, 2],
            "reassignments": [0, 1, 0, 1, 0, 1],
            "open_duration": [1, 2, 2, 3, 4, 5],
        }
    )
    y = [0, 0, 0, 1, 1, 1]
    model = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42)
    model.fit(X, y)
    return model, X, list(X.columns)


def test_prepare_shap_dataframe_keeps_feature_names():
    _, X, feature_columns = _toy_tree_model()

    X_df = prepare_shap_dataframe(X.to_numpy(), feature_columns=feature_columns)

    assert list(X_df.columns) == feature_columns
    assert X_df.shape == X.shape


def test_global_shap_outputs_are_created_on_small_subset(tmp_path):
    model, X, feature_columns = _toy_tree_model()

    summary_path = save_global_shap_summary(
        model=model,
        X=X,
        feature_columns=feature_columns,
        output_path=tmp_path / "shap_summary.png",
        max_samples=4,
    )
    importance = save_global_shap_importance_table(
        model=model,
        X=X,
        feature_columns=feature_columns,
        output_path=tmp_path / "shap_importance.csv",
        max_samples=4,
    )

    assert summary_path.exists()
    assert not importance.empty
    assert {"feature", "mean_abs_shap"}.issubset(importance.columns)


def test_local_shap_bar_plot_is_created(tmp_path):
    model, X, feature_columns = _toy_tree_model()

    output_path = save_local_shap_bar_plot(
        model=model,
        X=X,
        feature_columns=feature_columns,
        row_index=0,
        output_path=tmp_path / "local_shap.png",
    )

    assert output_path.exists()


def test_save_global_shap_importance_plot_creates_file(tmp_path):
    """save_global_shap_importance_plot should write a PNG bar chart."""
    model, X, feature_columns = _toy_tree_model()

    plot_path = save_global_shap_importance_plot(
        model=model,
        X=X,
        feature_columns=feature_columns,
        output_path=tmp_path / "shap_importance_bar.png",
        max_samples=4,
        top_n=3,
    )

    assert plot_path.exists()


def test_select_local_explanation_indices_identifies_all_case_types():
    """All three case types should be found when the predictions are mixed."""
    y_true = [1, 1, 0, 0, 1]
    y_pred = [1, 0, 0, 1, 1]

    indices = select_local_explanation_indices(y_true=y_true, y_pred=y_pred)

    assert indices["correct_positive"] is not None
    assert indices["correct_negative"] is not None
    assert indices["misclassified"] is not None
    # Sanity-check each returned index against the input labels.
    assert y_true[indices["correct_positive"]] == 1
    assert y_pred[indices["correct_positive"]] == 1
    assert y_true[indices["correct_negative"]] == 0
    assert y_pred[indices["correct_negative"]] == 0
    assert y_true[indices["misclassified"]] != y_pred[indices["misclassified"]]


def test_select_local_explanation_indices_returns_none_when_no_misclassified():
    """misclassified should be None when every prediction matches the true label."""
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]

    indices = select_local_explanation_indices(y_true=y_true, y_pred=y_pred)

    assert indices["correct_positive"] is not None
    assert indices["correct_negative"] is not None
    assert indices["misclassified"] is None


def test_prepare_shap_dataframe_samples_rows_when_max_samples_is_set():
    """prepare_shap_dataframe should truncate to max_samples rows."""
    _, X, feature_columns = _toy_tree_model()

    X_df = prepare_shap_dataframe(X, feature_columns=feature_columns, max_samples=3)

    assert len(X_df) == 3
    assert list(X_df.columns) == feature_columns


def test_save_local_shap_bar_plot_raises_on_out_of_bounds_row_index(tmp_path):
    """save_local_shap_bar_plot should raise IndexError for a row index beyond the data."""
    model, X, feature_columns = _toy_tree_model()

    with pytest.raises(IndexError):
        save_local_shap_bar_plot(
            model=model,
            X=X,
            feature_columns=feature_columns,
            row_index=999,
            output_path=tmp_path / "shap_invalid.png",
        )

