from pathlib import Path
from typing import Any, Iterable, Optional, Union

import pandas as pd

"""
This module defines the structure and validation for prediction outputs in Sprint 1.
The output is a DataFrame with one row per case, containing:
- case_id: Unique identifier for each case.
- y_true: The true label for the case (if available).
- prediction: The predicted class for the case.
- probability: The predicted probability for the positive class (if available).
- model: The name of the model used to generate the predictions.
"""


REQUIRED_OUTPUT_COLUMNS = [
    "case_id",
    "y_true",
    "prediction",
    "probability",
    "model",
]


def _to_series(values: Iterable[Any], name: str) -> pd.Series:
    """Convert an iterable to a pandas Series with a stable name."""
    if isinstance(values, pd.Series):
        return values.reset_index(drop=True).rename(name)
    return pd.Series(list(values), name=name)


def _validate_equal_lengths(columns: dict[str, pd.Series]) -> None:
    """Validate that all output columns have the same number of rows."""
    lengths = {column_name: len(values) for column_name, values in columns.items()}
    unique_lengths = set(lengths.values())

    if len(unique_lengths) != 1:
        raise ValueError(
            "Prediction output columns must have the same length. "
            f"Received lengths: {lengths}"
        )


def _validate_prediction_output(output_df: pd.DataFrame) -> None:
    """Validate that the prediction output is complete and usable for inspection."""
    missing_columns = [
        column for column in REQUIRED_OUTPUT_COLUMNS
        if column not in output_df.columns
    ]
    """check for missing required columns"""
    if missing_columns:
        raise ValueError(f"Missing required output columns: {missing_columns}")

    if output_df.empty:
        raise ValueError("Prediction output is empty.")

    if output_df["case_id"].isna().any():
        raise ValueError("Prediction output contains missing case_id values.")

    if output_df["prediction"].isna().any():
        raise ValueError("Prediction output contains missing prediction values.")

    if output_df["model"].isna().any() or (output_df["model"].astype(str).str.strip() == "").any():
        raise ValueError("Prediction output contains missing or empty model names.")


def create_prediction_output(
    case_ids: Iterable[Any],
    y_true: Optional[Iterable[Any]],
    predictions: Iterable[Any],
    probabilities: Optional[Iterable[float]] = None,
    model_name: str = "baseline",
) -> pd.DataFrame:
    """
    Create a structured prediction output table for Sprint 1.

    The output follows REQ-17 and contains one row per case with:
    case ID, true label, predicted class, predicted probability, and model name.

    Args:
        case_ids: Case identifiers belonging to the prediction dataset.
        y_true: True labels. If unavailable, pass None.
        predictions: Predicted classes.
        probabilities: Predicted probabilities. If unavailable, pass None.
        model_name: Name of the model used to generate the predictions.

    Returns:
        A validated pandas DataFrame with the Sprint 1 prediction output.
    """
    case_id_series = _to_series(case_ids, "case_id")
    prediction_series = _to_series(predictions, "prediction")

    if y_true is None:
        y_true_series = pd.Series([None] * len(prediction_series), name="y_true")
    else:
        y_true_series = _to_series(y_true, "y_true")

    if probabilities is None:
        probability_series = pd.Series([None] * len(prediction_series), name="probability")
    else:
        probability_series = _to_series(probabilities, "probability")

    model_series = pd.Series([model_name] * len(prediction_series), name="model")

    columns = {
        "case_id": case_id_series,
        "y_true": y_true_series,
        "prediction": prediction_series,
        "probability": probability_series,
        "model": model_series,
    }

    _validate_equal_lengths(columns)

    output_df = pd.DataFrame(columns)
    output_df = output_df[REQUIRED_OUTPUT_COLUMNS]

    _validate_prediction_output(output_df)

    return output_df


def save_prediction_output(
    case_ids: Iterable[Any],
    y_true: Optional[Iterable[Any]],
    predictions: Iterable[Any],
    probabilities: Optional[Iterable[float]] = None,
    model_name: str = "baseline",
    output_path: Union[str, Path] = "reports/predictions_sprint1.csv",
) -> pd.DataFrame:
    """
    Create, validate, and save the Sprint 1 prediction output as CSV.

    Args:
        case_ids: Case identifiers belonging to the prediction dataset.
        y_true: True labels. If unavailable, pass None.
        predictions: Predicted classes.
        probabilities: Predicted probabilities. If unavailable, pass None.
        model_name: Name of the model used to generate the predictions.
        output_path: Destination path for the CSV file.

    Returns:
        The saved prediction output DataFrame.
    """
    output_df = create_prediction_output(
        case_ids=case_ids,
        y_true=y_true,
        predictions=predictions,
        probabilities=probabilities,
        model_name=model_name,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    return output_df


def load_prediction_output(input_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load and validate a saved prediction output CSV.

    Args:
        input_path: Path to a saved prediction output CSV.

    Returns:
        A validated prediction output DataFrame.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Prediction output file not found: {input_path}")

    output_df = pd.read_csv(input_path)
    _validate_prediction_output(output_df)

    return output_df


if __name__ == "__main__":
    demo_output = save_prediction_output(
        case_ids=["case_1", "case_2", "case_3"],
        y_true=[0, 1, 1],
        predictions=[0, 1, 0],
        probabilities=[0.10, 0.85, 0.40],
        model_name="demo_baseline",
        output_path="reports/predictions_sprint1_demo.csv",
    )

    print(demo_output.head())