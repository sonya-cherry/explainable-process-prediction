from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import shap

from src.explainability.shap_explainer import (
    _get_positive_class_shap_values,
    prepare_shap_dataframe,
    save_local_shap_bar_plot,
)


def _positive_class_probability(model: Any, row: pd.DataFrame) -> Optional[float]:
    """Return the positive class probability when the model supports it."""
    if not hasattr(model, "predict_proba"):
        return None

    probabilities = model.predict_proba(row)
    if probabilities.shape[1] > 1:
        return float(probabilities[0, 1])
    return float(probabilities[0, 0])


def _top_shap_features(
    model: Any,
    row: pd.DataFrame,
    top_n: int,
) -> list[dict[str, float | str]]:
    """Compute top local SHAP contributors for one feature row."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(row)
    positive_values = _get_positive_class_shap_values(shap_values)

    contributions = pd.DataFrame(
        {
            "feature": row.columns,
            "value": row.iloc[0].to_numpy(),
            "shap_value": np.asarray(positive_values)[0],
        }
    )
    contributions["abs_shap_value"] = contributions["shap_value"].abs()
    contributions = contributions.sort_values("abs_shap_value", ascending=False)

    return contributions.head(top_n)[
        ["feature", "value", "shap_value", "abs_shap_value"]
    ].to_dict("records")


def predict_and_explain_case(
    model: Any,
    case_features: Any,
    feature_columns: list[str],
    case_id: Optional[Any] = None,
    top_n: int = 5,
    local_plot_path: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    """
    Predict one case or feature row and return a compact local explanation.

    Args:
        model: Fitted sklearn-compatible tree model.
        case_features: One-row feature matrix, Series, DataFrame, dense array, or sparse row.
        feature_columns: Feature names aligned with the model input.
        case_id: Optional identifier included in the returned dictionary.
        top_n: Number of top SHAP features to return.
        local_plot_path: Optional destination for a local SHAP bar plot.

    Returns:
        Dictionary with case ID, predicted class, probability, top SHAP features,
        and optionally the saved local plot path.
    """
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    X_df = prepare_shap_dataframe(case_features, feature_columns=feature_columns)
    if len(X_df) != 1:
        raise ValueError(
            "predict_and_explain_case expects exactly one feature row. "
            f"Received {len(X_df)} rows."
        )

    prediction = int(model.predict(X_df)[0])
    probability = _positive_class_probability(model, X_df)
    top_features = _top_shap_features(model=model, row=X_df, top_n=top_n)

    result = {
        "case_id": case_id,
        "predicted_class": prediction,
        "prediction_probability": probability,
        "top_features": top_features,
        "local_plot_path": None,
    }

    if local_plot_path is not None:
        saved_path = save_local_shap_bar_plot(
            model=model,
            X=X_df,
            feature_columns=feature_columns,
            row_index=0,
            output_path=local_plot_path,
            max_display=top_n,
        )
        result["local_plot_path"] = str(saved_path)

    return result

