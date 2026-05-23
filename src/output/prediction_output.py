from pathlib import Path
from typing import Any, Iterable, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd

"""
This module defines structured prediction outputs and basic visualizations.

The default visualization labels follow the project convention that class 1 is
the positive outcome and class 0 is the negative outcome. If the final outcome
definition uses more specific names, pass them via the class_labels argument.

The __main__ block uses artificial demo data only. For real Sprint 2 results,
call save_prediction_output() or save_model_prediction_output() from the
pipeline after the model has created predictions.
"""

"""For futute analysis: 
for sprint 2 my task was mostly to
create a more structured output format and basic visualizations.
I did not implement any new evaluation metrics or advanced visualizations, but I designed the output format to be flexible and extensible for future analysis. 
The output includes metadata columns for model name, model type, dataset split, threshold, and sprint, which will allow us to easily filter and compare different models and experiments in the future.
"""
REQUIRED_OUTPUT_COLUMNS = [
    "case_id",
    "y_true",
    "prediction",
    "probability",
    "model",
    "model_type",
    "dataset_split",
    "threshold",
    "sprint",
]

ALLOWED_BINARY_CLASSES = {0, 1}

DEFAULT_CLASS_LABELS = {
    0: "Class 0: negative outcome",
    1: "Class 1: positive outcome",
    "0": "Class 0: negative outcome",
    "1": "Class 1: positive outcome",
}

"""Convert an iterable to a pandas Series with a stable name."""
def _to_series(values: Iterable[Any], name: str) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.reset_index(drop=True).rename(name)
    return pd.Series(list(values), name=name)



"""Validate that all output columns have the same number of rows."""
def _validate_equal_lengths(columns: dict[str, pd.Series]) -> None:
   
    lengths = {column_name: len(values) for column_name, values in columns.items()}
    unique_lengths = set(lengths.values())

    if len(unique_lengths) != 1:
        raise ValueError(
            "Prediction output columns must have the same length. "
            f"Received lengths: {lengths}"
        )

"""Validate that a column contains only binary 0/1 values, ignoring missing values."""
def _validate_binary_values(series: pd.Series, column_name: str) -> None:
    non_missing_values = series.dropna()

    if non_missing_values.empty:
        return

    actual_classes = set(non_missing_values.unique())

    if not actual_classes.issubset(ALLOWED_BINARY_CLASSES):
        raise ValueError(
            f"{column_name} must contain only binary 0/1 values. "
            f"Found: {actual_classes}"
        )

"""Validate that the prediction output is complete and usable for inspection."""
def _validate_prediction_output(output_df: pd.DataFrame) -> None:
    missing_columns = [
        column for column in REQUIRED_OUTPUT_COLUMNS
        if column not in output_df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required output columns: {missing_columns}")

    if output_df.empty:
        raise ValueError("Prediction output is empty.")

    if output_df["case_id"].isna().any():
        raise ValueError("Prediction output contains missing case_id values.")

    if output_df["prediction"].isna().any():
        raise ValueError("Prediction output contains missing prediction values.")

    if output_df["model"].isna().any() or (
        output_df["model"].astype(str).str.strip() == ""
    ).any():
        raise ValueError("Prediction output contains missing or empty model names.")

    if output_df["model_type"].isna().any() or (
        output_df["model_type"].astype(str).str.strip() == ""
    ).any():
        raise ValueError("Prediction output contains missing or empty model_type values.")

    if output_df["dataset_split"].isna().any() or (
        output_df["dataset_split"].astype(str).str.strip() == ""
    ).any():
        raise ValueError("Prediction output contains missing or empty dataset_split values.")

    if output_df["sprint"].isna().any() or (
        output_df["sprint"].astype(str).str.strip() == ""
    ).any():
        raise ValueError("Prediction output contains missing or empty sprint values.")

    valid_probabilities = pd.to_numeric(output_df["probability"], errors="coerce").dropna()
    if not valid_probabilities.between(0, 1).all():
        raise ValueError("Probabilities must be between 0 and 1.")

    valid_thresholds = pd.to_numeric(output_df["threshold"], errors="coerce").dropna()
    if not valid_thresholds.between(0, 1).all():
        raise ValueError("Threshold values must be between 0 and 1.")

    _validate_binary_values(output_df["prediction"], "prediction")
    _validate_binary_values(output_df["y_true"], "y_true")


def create_prediction_output(
    case_ids: Iterable[Any],
    y_true: Optional[Iterable[Any]],
    predictions: Iterable[Any],
    probabilities: Optional[Iterable[float]] = None,
    model_name: str = "baseline",
    model_type: str = "baseline",
    dataset_split: str = "test",
    threshold: float = 0.5,
    sprint: str = "sprint2",
) -> pd.DataFrame:
    """
    Create a structured prediction output table.

    The output contains one row per case with case ID, true label, predicted class,
    predicted probability, model name, and Sprint 2 metadata.
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
    model_type_series = pd.Series([model_type] * len(prediction_series), name="model_type")
    dataset_split_series = pd.Series(
        [dataset_split] * len(prediction_series),
        name="dataset_split",
    )
    threshold_series = pd.Series([threshold] * len(prediction_series), name="threshold")
    sprint_series = pd.Series([sprint] * len(prediction_series), name="sprint")

    columns = {
        "case_id": case_id_series,
        "y_true": y_true_series,
        "prediction": prediction_series,
        "probability": probability_series,
        "model": model_series,
        "model_type": model_type_series, """sprint2 metadata columns for future analysis and filtering"""
        "dataset_split": dataset_split_series,"""sprint2 metadata columns for future analysis and filtering"""
        "threshold": threshold_series,"""sprint2 metadata columns for future analysis and filtering"""
        "sprint": sprint_series, 
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
    model_type: str = "baseline",
    dataset_split: str = "test",
    threshold: float = 0.5,
    sprint: str = "sprint2",
    output_path: Union[str, Path] = "reports/predictions_sprint2.csv",
) -> pd.DataFrame:
    """Create, validate, and save the prediction output as CSV."""
    output_df = create_prediction_output(
        case_ids=case_ids,
        y_true=y_true,
        predictions=predictions,
        probabilities=probabilities,
        model_name=model_name,
        model_type=model_type,
        dataset_split=dataset_split,
        threshold=threshold,
        sprint=sprint,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    return output_df


def load_prediction_output(input_path: Union[str, Path]) -> pd.DataFrame:
    """Load and validate a saved prediction output CSV."""
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Prediction output file not found: {input_path}")

    output_df = pd.read_csv(input_path)
    _validate_prediction_output(output_df)

    return output_df


def save_model_prediction_output(
    model: Any,
    features: Any,
    case_ids: Iterable[Any],
    y_true: Optional[Iterable[Any]],
    model_name: str,
    model_type: str = "baseline",
    dataset_split: str = "test",
    threshold: float = 0.5,
    sprint: str = "sprint2",
    output_path: Union[str, Path] = "reports/predictions_sprint2.csv",
) -> pd.DataFrame:
    """
    Generate predictions from a trained model and save the real prediction output.

    This function connects the output module to the actual model pipeline. It does
    not train the model itself. It expects that features, case_ids, and y_true are
    already aligned row by row.
    """
    if not hasattr(model, "predict"):
        raise TypeError("Model must provide a predict method.")

    predictions = model.predict(features)

    probabilities = None
    if hasattr(model, "predict_proba"):
        prediction_probabilities = model.predict_proba(features)
        if prediction_probabilities.shape[1] > 1:
            probabilities = prediction_probabilities[:, 1]
        else:
            probabilities = prediction_probabilities[:, 0]

    return save_prediction_output(
        case_ids=case_ids,
        y_true=y_true,
        predictions=predictions,
        probabilities=probabilities,
        model_name=model_name,
        model_type=model_type,
        dataset_split=dataset_split,
        threshold=threshold,
        sprint=sprint,
        output_path=output_path,
    )


def _prepare_distribution_table(
    series: pd.Series,
    value_column_name: str,
    class_labels: Optional[dict[Any, str]] = None,
) -> pd.DataFrame:
    """Create a count and percentage table for a categorical distribution."""
    counts = series.value_counts(dropna=False).sort_index()
    total = counts.sum()

    distribution_df = counts.reset_index()
    distribution_df.columns = [value_column_name, "count"]
    distribution_df["percentage"] = distribution_df["count"] / total * 100

    if class_labels is not None:
        label_column = f"{value_column_name}_label"
        distribution_df[label_column] = distribution_df[value_column_name].map(class_labels)
        distribution_df[label_column] = distribution_df[label_column].fillna(
            distribution_df[value_column_name].astype(str)
        )

    return distribution_df


def _save_horizontal_distribution_chart(
    distribution_df: pd.DataFrame,
    category_column: str,
    title: str,
    xlabel: str,
    output_path: Union[str, Path],
) -> Path:
    """Save a clean horizontal bar chart with count and percentage labels."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = distribution_df[category_column].astype(str)
    counts = distribution_df["count"]
    percentages = distribution_df["percentage"]

    fig_height = max(3.5, 0.6 * len(distribution_df) + 1.5)
    fig, ax = plt.subplots(figsize=(8, fig_height))

    bars = ax.barh(labels, counts)
    ax.bar_label(
        bars,
        labels=[
            f"{count} ({percentage:.1f}%)"
            for count, percentage in zip(counts, percentages)
        ],
        padding=4,
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    max_count = counts.max()
    ax.set_xlim(0, max_count * 1.25 if max_count > 0 else 1)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_prediction_distribution(
    prediction_output: Union[pd.DataFrame, str, Path],
    output_path: Union[str, Path] = "figures/prediction_distribution_sprint2.png",
    class_labels: Optional[dict[Any, str]] = None,
) -> Path:
    """Create and save a horizontal bar chart of predicted classes."""
    if isinstance(prediction_output, (str, Path)):
        output_df = load_prediction_output(prediction_output)
    else:
        output_df = prediction_output.copy()
        _validate_prediction_output(output_df)

    if class_labels is None:
        class_labels = DEFAULT_CLASS_LABELS

    distribution_df = _prepare_distribution_table(
        output_df["prediction"],
        value_column_name="prediction",
        class_labels=class_labels,
    )

    return _save_horizontal_distribution_chart(
        distribution_df=distribution_df,
        category_column="prediction_label",
        title="Prediction Distribution by Outcome Class",
        xlabel="Number of Cases",
        output_path=output_path,
    )


def plot_label_distribution(
    prediction_output: Union[pd.DataFrame, str, Path],
    output_path: Union[str, Path] = "figures/label_distribution_sprint2.png",
    class_labels: Optional[dict[Any, str]] = None,
) -> Path:
    """Create and save a horizontal bar chart of true labels."""
    if isinstance(prediction_output, (str, Path)):
        output_df = load_prediction_output(prediction_output)
    else:
        output_df = prediction_output.copy()
        _validate_prediction_output(output_df)

    if output_df["y_true"].isna().all():
        raise ValueError("Cannot plot label distribution because y_true is not available.")

    if class_labels is None:
        class_labels = DEFAULT_CLASS_LABELS

    distribution_df = _prepare_distribution_table(
        output_df["y_true"],
        value_column_name="true_label",
        class_labels=class_labels,
    )

    return _save_horizontal_distribution_chart(
        distribution_df=distribution_df,
        category_column="true_label_label",
        title="True Outcome Distribution",
        xlabel="Number of Cases",
        output_path=output_path,
    )


def create_prediction_visualizations(
    prediction_output: Union[pd.DataFrame, str, Path] = "reports/predictions_sprint2.csv",
    figures_dir: Union[str, Path] = "figures",
    class_labels: Optional[dict[Any, str]] = None,
) -> dict[str, Path]:
    """Create basic prediction visualizations from prediction output."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = {
        "prediction_distribution": plot_prediction_distribution(
            prediction_output=prediction_output,
            output_path=figures_dir / "prediction_distribution_sprint2.png",
            class_labels=class_labels,
        )
    }

    try:
        saved_paths["label_distribution"] = plot_label_distribution(
            prediction_output=prediction_output,
            output_path=figures_dir / "label_distribution_sprint2.png",
            class_labels=class_labels,
        )
    except ValueError:
        pass

    return saved_paths


def create_sprint1_visualizations(
    prediction_output: Union[pd.DataFrame, str, Path] = "reports/predictions_sprint1.csv",
    figures_dir: Union[str, Path] = "figures",
    class_labels: Optional[dict[Any, str]] = None,
) -> dict[str, Path]:
    """Backward-compatible wrapper for Sprint 1 notebooks."""
    return create_prediction_visualizations(
        prediction_output=prediction_output,
        figures_dir=figures_dir,
        class_labels=class_labels,
    )


if __name__ == "__main__":
    # Demo data only: these values are manually created to test this module.
    # They are not derived from the BPI 2013 event log or from the final model.
    demo_output = save_prediction_output(
        case_ids=["case_1", "case_2", "case_3"],
        y_true=[0, 1, 1],
        predictions=[0, 1, 0],
        probabilities=[0.10, 0.85, 0.40],
        model_name="demo_baseline",
        model_type="baseline",
        dataset_split="test",
        threshold=0.5,
        sprint="sprint2",
        output_path="reports/predictions_sprint2_demo.csv",
    )

    print("Output created:")
    print(demo_output.head())

    saved_figures = create_prediction_visualizations(
        prediction_output=demo_output,
        figures_dir="figures/demo",
    )
    print("Demo figures saved:", saved_figures)