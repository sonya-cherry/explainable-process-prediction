"""Reusable training pipeline for prefix-level process outcome prediction.

This module contains the complete workflow from raw event-log loading to
trained classification models.

The pipeline uses:

- a final-activity-based binary outcome;
- temporal train/validation/test splitting by original case;
- prefix generation after splitting;
- aligned feature encoding;
- majority baseline, Logistic Regression, and Random Forest;
- validation-based Random Forest model selection.

The module does not create final figures or SHAP explanations. Those belong
to the evaluation pipeline.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from scipy.sparse import csr_matrix

from src.dataExtraction.extract import import_data, split
from src.featureEngineering.feature_encoding import Encode
from src.featureEngineering.prefix_generation import generate_prefix
from src.modeling.baseline import train_majority_baseline
from src.modeling.evaluation import build_evaluation_table, evaluate_model
from src.modeling.model_selection import select_best_random_forest
from src.modeling.models import train_logistic_regression


DEFAULT_POSITIVE_FINAL_ACTIVITIES = ("Closed", "Resolved")

DEFAULT_RANDOM_FOREST_GRID = (
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 100, "max_depth": 10},
    {"n_estimators": 200, "max_depth": 5},
)

DEFAULT_EXCLUDED_FEATURE_KEYWORDS = (
    "case_duration",
    "relative_age",
)


@dataclass
class TrainingPipelineResult:
    """All reusable outputs produced by the training pipeline."""

    models: dict[str, Any]
    best_random_forest_parameters: dict[str, Any]

    X_train: csr_matrix
    X_validation: csr_matrix
    X_test: csr_matrix

    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series

    case_ids_train: pd.Index
    case_ids_validation: pd.Index
    case_ids_test: pd.Index

    feature_columns: list[str]
    removed_feature_columns: list[str]

    random_forest_validation_results: pd.DataFrame
    validation_comparison: pd.DataFrame

    dataset_summary: dict[str, Any]


def construct_final_activity_outcome(
    event_log: pd.DataFrame,
    positive_final_activities: Sequence[str] = DEFAULT_POSITIVE_FINAL_ACTIVITIES,
) -> pd.DataFrame:
    """Create a binary case outcome from the final lifecycle transition.

    The BPI 2013 incident log contains:

    - ``concept:name``: original process activity;
    - ``lifecycle:transition``: lifecycle state such as Closed or Resolved.

    For compatibility with the existing feature encoder, the original activity
    is renamed to ``event_type`` and ``lifecycle:transition`` becomes
    ``concept:name``.

    A case receives outcome 1 when its final lifecycle transition is contained
    in ``positive_final_activities``. Otherwise, it receives outcome 0.

    Parameters
    ----------
    event_log:
        Raw event-level DataFrame.
    positive_final_activities:
        Lifecycle transitions interpreted as positive final outcomes.

    Returns
    -------
    pd.DataFrame
        Event log with an integer ``outcome`` column.
    """
    required_columns = {
        "case:concept:name",
        "concept:name",
        "lifecycle:transition",
        "time:timestamp",
    }

    missing_columns = required_columns - set(event_log.columns)
    if missing_columns:
        raise ValueError(
            "Cannot construct final-activity outcome. "
            f"Missing columns: {sorted(missing_columns)}"
        )

    labelled_log = event_log.copy()
    labelled_log["time:timestamp"] = pd.to_datetime(
        labelled_log["time:timestamp"]
    )

    labelled_log = labelled_log.rename(
        columns={
            "concept:name": "event_type",
            "lifecycle:transition": "concept:name",
        }
    )

    final_activity_per_case = (
        labelled_log
        .sort_values(["case:concept:name", "time:timestamp"])
        .groupby("case:concept:name")["concept:name"]
        .last()
    )

    case_outcomes = (
        final_activity_per_case
        .isin(positive_final_activities)
        .astype(int)
    )

    labelled_log["outcome"] = (
        labelled_log["case:concept:name"]
        .map(case_outcomes)
        .astype(int)
    )

    return labelled_log


def _generate_prefix_split(
    event_log: pd.DataFrame,
    split_name: str,
    min_prefix: int,
) -> pd.DataFrame:
    """Generate prefixes and provide a clear error for empty output."""
    prefix_log = generate_prefix(
        event_log,
        min_prefix=min_prefix,
    )

    if prefix_log.empty:
        raise ValueError(
            f"No prefixes were generated for the {split_name} split. "
            "Each included case must contain more events than min_prefix."
        )

    return prefix_log


def _align_encoded_output(
    X: csr_matrix,
    y: pd.Series,
    case_ids: pd.Index,
) -> tuple[csr_matrix, pd.Series, pd.Index]:
    """Ensure that feature rows, labels, and case IDs use the same order.

    ``Encode`` constructs the feature matrix and label Series separately.
    Pandas groupby operations may sort labels lexicographically, while feature
    rows can retain insertion order. This is particularly risky for prefix IDs,
    where ``prefix_10`` may sort before ``prefix_2``.

    This function explicitly reindexes the labels to the feature-row case IDs.
    """
    aligned_case_ids = pd.Index(case_ids)
    aligned_y = y.reindex(aligned_case_ids)

    if aligned_y.isna().any():
        missing_ids = aligned_y[aligned_y.isna()].index.tolist()
        raise ValueError(
            "Some encoded feature rows have no aligned outcome label. "
            f"Example missing IDs: {missing_ids[:5]}"
        )

    if X.shape[0] != len(aligned_y):
        raise ValueError(
            "Feature and label row counts differ after alignment: "
            f"X has {X.shape[0]} rows, y has {len(aligned_y)} rows."
        )

    return X, aligned_y.astype(int), aligned_case_ids


def encode_splits(
    train_prefix_log: pd.DataFrame,
    validation_prefix_log: pd.DataFrame,
    test_prefix_log: pd.DataFrame,
) -> tuple[
    csr_matrix,
    csr_matrix,
    csr_matrix,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Index,
    pd.Index,
    pd.Index,
    list[str],
]:
    """Encode all splits using the training feature space.

    The training split defines the complete feature-column list. Validation
    and test matrices are reindexed to those columns by the existing encoder.
    """
    X_train, y_train, case_ids_train, feature_columns = Encode(
        train_prefix_log
    )

    X_validation, y_validation, case_ids_validation, _ = Encode(
        validation_prefix_log,
        feature_columns=feature_columns,
    )

    X_test, y_test, case_ids_test, _ = Encode(
        test_prefix_log,
        feature_columns=feature_columns,
    )

    X_train, y_train, case_ids_train = _align_encoded_output(
        X_train,
        y_train,
        case_ids_train,
    )
    X_validation, y_validation, case_ids_validation = _align_encoded_output(
        X_validation,
        y_validation,
        case_ids_validation,
    )
    X_test, y_test, case_ids_test = _align_encoded_output(
        X_test,
        y_test,
        case_ids_test,
    )

    if not (
        X_train.shape[1]
        == X_validation.shape[1]
        == X_test.shape[1]
        == len(feature_columns)
    ):
        raise ValueError(
            "The train, validation, and test feature spaces are not aligned."
        )

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        case_ids_train,
        case_ids_validation,
        case_ids_test,
        feature_columns,
    )


def remove_excluded_features(
    X_train: csr_matrix,
    X_validation: csr_matrix,
    X_test: csr_matrix,
    feature_columns: list[str],
    excluded_keywords: Sequence[str] = DEFAULT_EXCLUDED_FEATURE_KEYWORDS,
) -> tuple[
    csr_matrix,
    csr_matrix,
    csr_matrix,
    list[str],
    list[str],
]:
    """Remove leakage-prone or inconsistent features from every split.

    Feature removal is based on case-insensitive keyword matching. The same
    column indices are applied to all three matrices.
    """
    normalized_keywords = tuple(
        keyword.lower() for keyword in excluded_keywords
    )

    removed_columns = [
        column
        for column in feature_columns
        if any(
            keyword in column.lower()
            for keyword in normalized_keywords
        )
    ]

    kept_indices = [
        index
        for index, column in enumerate(feature_columns)
        if column not in removed_columns
    ]

    kept_columns = [
        feature_columns[index]
        for index in kept_indices
    ]

    if not kept_columns:
        raise ValueError(
            "All feature columns were removed. "
            "Review excluded_feature_keywords."
        )

    return (
        X_train[:, kept_indices],
        X_validation[:, kept_indices],
        X_test[:, kept_indices],
        kept_columns,
        removed_columns,
    )


def build_dataset_summary(
    labelled_log: pd.DataFrame,
    train_log: pd.DataFrame,
    validation_log: pd.DataFrame,
    test_log: pd.DataFrame,
    train_prefix_log: pd.DataFrame,
    validation_prefix_log: pd.DataFrame,
    test_prefix_log: pd.DataFrame,
    number_of_features: int,
) -> dict[str, Any]:
    """Collect compact dataset information for the notebook and README."""
    case_column = "case:concept:name"

    case_outcomes = (
        labelled_log
        .groupby(case_column)["outcome"]
        .first()
    )

    return {
        "events": len(labelled_log),
        "original_cases": labelled_log[case_column].nunique(),
        "activities": labelled_log["event_type"].nunique(),
        "positive_original_cases": int((case_outcomes == 1).sum()),
        "negative_original_cases": int((case_outcomes == 0).sum()),
        "train_original_cases": train_log[case_column].nunique(),
        "validation_original_cases": validation_log[case_column].nunique(),
        "test_original_cases": test_log[case_column].nunique(),
        "train_prefixes": train_prefix_log[case_column].nunique(),
        "validation_prefixes": validation_prefix_log[case_column].nunique(),
        "test_prefixes": test_prefix_log[case_column].nunique(),
        "number_of_features": number_of_features,
    }


def run_training_pipeline(
    event_log_path: str | Path,
    *,
    drop_columns: Sequence[str] = ("impact", "org:role"),
    positive_final_activities: Sequence[str] = (
        DEFAULT_POSITIVE_FINAL_ACTIVITIES
    ),
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    min_prefix: int = 1,
    random_forest_param_grid: list[dict[str, Any]] | None = None,
    random_forest_scoring: str = "f1",
    excluded_feature_keywords: Sequence[str] = (
        DEFAULT_EXCLUDED_FEATURE_KEYWORDS
    ),
) -> TrainingPipelineResult:
    """Run the complete reusable training pipeline.

    Parameters
    ----------
    event_log_path:
        Path to the BPI event log in XES or CSV format.
    drop_columns:
        Optional event-log columns removed during loading.
    positive_final_activities:
        Final lifecycle transitions treated as positive outcomes.
    train_ratio, validation_ratio, test_ratio:
        Temporal split ratios based on original case start times.
    min_prefix:
        Minimum number of events retained in a generated prefix.
    random_forest_param_grid:
        Random Forest configurations compared on validation data.
    random_forest_scoring:
        Validation metric used to select the best Random Forest.
    excluded_feature_keywords:
        Feature-name keywords removed before model training.

    Returns
    -------
    TrainingPipelineResult
        Trained models, aligned datasets, validation results, and metadata.
    """
    event_log_path = Path(event_log_path)

    if not event_log_path.exists():
        raise FileNotFoundError(
            f"Event log not found: {event_log_path}"
        )

    raw_log = import_data(
        str(event_log_path),
        drop_columns=list(drop_columns),
    )

    labelled_log = construct_final_activity_outcome(
        raw_log,
        positive_final_activities=positive_final_activities,
    )

    train_log, validation_log, test_log = split(
        labelled_log,
        train_ratio=train_ratio,
        val_ratio=validation_ratio,
        test_ratio=test_ratio,
    )

    train_prefix_log = _generate_prefix_split(
        train_log,
        split_name="training",
        min_prefix=min_prefix,
    )
    validation_prefix_log = _generate_prefix_split(
        validation_log,
        split_name="validation",
        min_prefix=min_prefix,
    )
    test_prefix_log = _generate_prefix_split(
        test_log,
        split_name="test",
        min_prefix=min_prefix,
    )

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        case_ids_train,
        case_ids_validation,
        case_ids_test,
        feature_columns,
    ) = encode_splits(
        train_prefix_log,
        validation_prefix_log,
        test_prefix_log,
    )

    (
        X_train,
        X_validation,
        X_test,
        feature_columns,
        removed_feature_columns,
    ) = remove_excluded_features(
        X_train,
        X_validation,
        X_test,
        feature_columns,
        excluded_keywords=excluded_feature_keywords,
    )

    baseline_model = train_majority_baseline(
        X_train,
        y_train,
    )

    logistic_regression_model = train_logistic_regression(
        X_train,
        y_train,
    )

    if random_forest_param_grid is None:
        random_forest_param_grid = [
            dict(parameters)
            for parameters in DEFAULT_RANDOM_FOREST_GRID
        ]

    (
        random_forest_model,
        best_random_forest_parameters,
        random_forest_validation_results,
    ) = select_best_random_forest(
        X_train,
        y_train,
        X_validation,
        y_validation,
        scoring=random_forest_scoring,
        param_grid=random_forest_param_grid,
    )

    models = {
        "Majority baseline": baseline_model,
        "Logistic Regression": logistic_regression_model,
        "Random Forest": random_forest_model,
    }

    validation_results = [
        evaluate_model(
            model,
            X_validation,
            y_validation,
            model_name,
        )
        for model_name, model in models.items()
    ]

    validation_comparison = build_evaluation_table(
        validation_results
    )

    dataset_summary = build_dataset_summary(
        labelled_log=labelled_log,
        train_log=train_log,
        validation_log=validation_log,
        test_log=test_log,
        train_prefix_log=train_prefix_log,
        validation_prefix_log=validation_prefix_log,
        test_prefix_log=test_prefix_log,
        number_of_features=len(feature_columns),
    )

    return TrainingPipelineResult(
        models=models,
        best_random_forest_parameters=(
            best_random_forest_parameters
        ),
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
        case_ids_train=case_ids_train,
        case_ids_validation=case_ids_validation,
        case_ids_test=case_ids_test,
        feature_columns=feature_columns,
        removed_feature_columns=removed_feature_columns,
        random_forest_validation_results=(
            random_forest_validation_results
        ),
        validation_comparison=validation_comparison,
        dataset_summary=dataset_summary,
    )

