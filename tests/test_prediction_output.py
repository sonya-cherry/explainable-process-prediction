from pathlib import Path
import sys

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.output.prediction_output import (
    REQUIRED_OUTPUT_COLUMNS,
    create_prediction_output,
    load_prediction_output,
    save_model_prediction_output,
)


def test_create_prediction_output_includes_sprint_metadata():
    output = create_prediction_output(
        case_ids=["case_1", "case_2"],
        y_true=[0, 1],
        predictions=[0, 1],
        probabilities=[0.2, 0.9],
        model_name="Random Forest",
        model_type="black_box",
        dataset_split="test",
        threshold=0.5,
        sprint="sprint3",
    )

    assert list(output.columns) == REQUIRED_OUTPUT_COLUMNS
    assert output["model"].tolist() == ["Random Forest", "Random Forest"]
    assert output["model_type"].tolist() == ["black_box", "black_box"]
    assert output["dataset_split"].tolist() == ["test", "test"]
    assert output["sprint"].tolist() == ["sprint3", "sprint3"]


def test_prediction_output_rejects_invalid_probability():
    with pytest.raises(ValueError, match="Probabilities must be between 0 and 1"):
        create_prediction_output(
            case_ids=["case_1"],
            y_true=[1],
            predictions=[1],
            probabilities=[1.5],
        )


def test_save_model_prediction_output_uses_model_probabilities(tmp_path):
    X = pd.DataFrame(
        {
            "feature_a": [0.0, 1.0, 2.0, 3.0],
            "feature_b": [1.0, 1.0, 0.0, 0.0],
        }
    )
    y = [0, 0, 1, 1]
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)

    output_path = tmp_path / "predictions.csv"
    output = save_model_prediction_output(
        model=model,
        features=X,
        case_ids=["case_1", "case_2", "case_3", "case_4"],
        y_true=y,
        model_name="Random Forest",
        model_type="black_box",
        sprint="sprint3",
        output_path=output_path,
    )

    loaded = load_prediction_output(output_path)
    assert output_path.exists()
    assert output["probability"].between(0, 1).all()
    assert loaded.shape == output.shape

