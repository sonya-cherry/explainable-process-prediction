from pathlib import Path
import sys

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.prototype.predict_explain import predict_and_explain_case


def _toy_model():
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
    return model, X


def test_predict_and_explain_case_returns_prediction_and_top_features(tmp_path):
    model, X = _toy_model()

    result = predict_and_explain_case(
        model=model,
        case_features=X.iloc[[0]],
        feature_columns=list(X.columns),
        case_id="case_1",
        top_n=2,
        local_plot_path=tmp_path / "case_1_shap.png",
    )

    assert result["case_id"] == "case_1"
    assert result["predicted_class"] in {0, 1}
    assert 0 <= result["prediction_probability"] <= 1
    assert len(result["top_features"]) == 2
    assert Path(result["local_plot_path"]).exists()


def test_predict_and_explain_case_requires_one_row():
    model, X = _toy_model()

    with pytest.raises(ValueError, match="exactly one feature row"):
        predict_and_explain_case(
            model=model,
            case_features=X.iloc[:2],
            feature_columns=list(X.columns),
        )

