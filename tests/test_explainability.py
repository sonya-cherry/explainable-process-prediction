from pathlib import Path
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.explainability.shap_explainer import (
    prepare_shap_dataframe,
    save_global_shap_importance_table,
    save_global_shap_summary,
    save_local_shap_bar_plot,
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

