"""Reusable evaluation pipeline for process outcome prediction.

This module evaluates the trained models on the held-out test set and creates
the final project outputs:

- model comparison tables;
- structured prediction outputs;
- model comparison and ROC plots;
- Random Forest confusion matrix;
- global SHAP explanations;
- representative local SHAP explanations.

The module receives a TrainingPipelineResult. It does not load data, construct
features, or train models again.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.explainability.shap_explainer import (
    save_global_shap_importance_plot,
    save_global_shap_importance_table,
    save_global_shap_summary,
    save_local_shap_bar_plot,
    select_local_explanation_indices,
)
from src.modeling.evaluation import (
    save_confusion_matrix_plot,
    save_model_comparison_plot,
    save_roc_curves,
)
from src.output.prediction_output import create_prediction_output
from src.pipeline.training_pipeline import TrainingPipelineResult


DEFAULT_MODEL_TYPES = {
    "Majority baseline": "baseline",
    "Logistic Regression": "interpretable",
    "Random Forest": "black_box",
}


@dataclass
class EvaluationPipelineResult:
    """Outputs produced by the complete evaluation pipeline."""

    test_comparison: pd.DataFrame
    predictions: pd.DataFrame
    local_explanations: pd.DataFrame

    shap_importance: pd.DataFrame | None

    predictions_by_model: dict[str, np.ndarray]
    probabilities_by_model: dict[str, np.ndarray]

    report_paths: dict[str, Path]
    figure_paths: dict[str, Path]
    local_shap_paths: dict[str, Path]


def _positive_class_probability(
    model: Any,
    X: Any,
) -> np.ndarray:
    """Return probabilities for class 1.

    Most classifiers in this project expose ``predict_proba``. This helper
    also handles the unusual situation where a model was trained on only one
    class, which can otherwise cause an indexing error when using ``[:, 1]``.

    Parameters
    ----------
    model:
        Fitted sklearn-compatible classifier.
    X:
        Feature matrix.

    Returns
    -------
    np.ndarray
        One probability or score per input row.
    """
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(X))
        classes = list(model.classes_)

        if 1 in classes:
            positive_class_index = classes.index(1)
            return probabilities[:, positive_class_index]

        # A model trained only on class 0 assigns probability 0 to class 1.
        return np.zeros(X.shape[0], dtype=float)

    if hasattr(model, "decision_function"):
        decision_scores = np.asarray(model.decision_function(X))

        # Convert arbitrary decision scores to the interval [0, 1].
        return 1.0 / (1.0 + np.exp(-decision_scores))

    # Final fallback for simple classifiers without probabilities or scores.
    return np.asarray(model.predict(X), dtype=float)


def _safe_roc_auc(
    y_true: Any,
    y_score: Any,
) -> float:
    """Compute ROC AUC when both classes occur in the test labels."""
    unique_classes = np.unique(np.asarray(y_true))

    if len(unique_classes) < 2:
        return float("nan")

    return float(roc_auc_score(y_true, y_score))


def _safe_pr_auc(
    y_true: Any,
    y_score: Any,
) -> float:
    """Compute Precision-Recall AUC when both classes are available."""
    unique_classes = np.unique(np.asarray(y_true))

    if len(unique_classes) < 2:
        return float("nan")

    return float(average_precision_score(y_true, y_score))


def _evaluate_predictions(
    model_name: str,
    y_true: Any,
    y_pred: Any,
    y_score: Any,
) -> dict[str, float | str]:
    """Calculate the final test metrics for one model."""
    return {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "roc_auc": _safe_roc_auc(y_true, y_score),
        "pr_auc": _safe_pr_auc(y_true, y_score),
    }


def _validate_training_result(
    training_result: TrainingPipelineResult,
) -> None:
    """Check that the received training output is internally consistent."""
    number_of_test_rows = training_result.X_test.shape[0]

    if number_of_test_rows == 0:
        raise ValueError("The test feature matrix is empty.")

    if len(training_result.y_test) != number_of_test_rows:
        raise ValueError(
            "The test labels are not aligned with the test feature matrix: "
            f"X_test has {number_of_test_rows} rows, but y_test has "
            f"{len(training_result.y_test)} labels."
        )

    if len(training_result.case_ids_test) != number_of_test_rows:
        raise ValueError(
            "The test case IDs are not aligned with the test feature matrix: "
            f"X_test has {number_of_test_rows} rows, but there are "
            f"{len(training_result.case_ids_test)} case IDs."
        )

    if training_result.X_test.shape[1] != len(
        training_result.feature_columns
    ):
        raise ValueError(
            "The test feature matrix and feature-name list are inconsistent: "
            f"X_test has {training_result.X_test.shape[1]} columns, but "
            f"{len(training_result.feature_columns)} feature names were provided."
        )

    if not training_result.models:
        raise ValueError("The training result contains no fitted models.")

    if "Random Forest" not in training_result.models:
        raise ValueError(
            "The evaluation pipeline requires a model named "
            "'Random Forest' for SHAP explanations."
        )


def _create_all_prediction_outputs(
    training_result: TrainingPipelineResult,
    predictions_by_model: dict[str, np.ndarray],
    probabilities_by_model: dict[str, np.ndarray],
    model_types: dict[str, str],
    threshold: float,
    sprint: str,
) -> pd.DataFrame:
    """Create one combined prediction table for all evaluated models."""
    prediction_tables = []

    for model_name in training_result.models:
        model_type = model_types.get(
            model_name,
            "unknown",
        )

        model_output = create_prediction_output(
            case_ids=training_result.case_ids_test,
            y_true=training_result.y_test,
            predictions=predictions_by_model[model_name],
            probabilities=probabilities_by_model[model_name],
            model_name=model_name,
            model_type=model_type,
            dataset_split="test",
            threshold=threshold,
            sprint=sprint,
        )

        prediction_tables.append(model_output)

    combined_output = pd.concat(
        prediction_tables,
        ignore_index=True,
    )

    return combined_output


def _create_local_shap_explanations(
    model: Any,
    X_test: Any,
    feature_columns: list[str],
    case_ids: pd.Index,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    output_dir: Path,
    max_display: int,
) -> tuple[dict[str, Path], pd.DataFrame]:
    """Create local SHAP plots for representative test cases.

    We select:

    - one correctly predicted positive case;
    - one correctly predicted negative case;
    - one misclassified case.

    Only the selected feature row is passed to SHAP. This is significantly
    faster than recalculating SHAP values for the complete test set once for
    every local explanation.
    """
    selected_indices = select_local_explanation_indices(
        y_true=y_true,
        y_pred=y_pred,
    )

    saved_paths: dict[str, Path] = {}
    explanation_records = []

    y_true_array = np.asarray(y_true)
    case_id_array = np.asarray(case_ids)

    for explanation_type, row_index in selected_indices.items():
        if row_index is None:
            continue

        case_features = X_test[row_index]

        output_path = (
            output_dir
            / f"shap_local_{explanation_type}.png"
        )

        saved_path = save_local_shap_bar_plot(
            model=model,
            X=case_features,
            feature_columns=feature_columns,
            row_index=0,
            output_path=output_path,
            max_display=max_display,
        )

        saved_paths[explanation_type] = saved_path

        explanation_records.append(
            {
                "explanation_type": explanation_type,
                "row_index": row_index,
                "case_id": case_id_array[row_index],
                "y_true": int(y_true_array[row_index]),
                "prediction": int(y_pred[row_index]),
                "probability": float(y_score[row_index]),
                "figure_path": str(saved_path),
            }
        )

    local_explanations = pd.DataFrame(
        explanation_records,
        columns=[
            "explanation_type",
            "row_index",
            "case_id",
            "y_true",
            "prediction",
            "probability",
            "figure_path",
        ],
    )

    return saved_paths, local_explanations


def run_evaluation_pipeline(
    training_result: TrainingPipelineResult,
    output_dir: str | Path = "outputs",
    *,
    generate_shap: bool = True,
    shap_max_samples: int = 300,
    shap_top_n: int = 15,
    local_shap_max_display: int = 10,
    threshold: float = 0.5,
    sprint: str = "sprint3",
    model_types: dict[str, str] | None = None,
) -> EvaluationPipelineResult:
    """Run the complete final evaluation pipeline.

    Parameters
    ----------
    training_result:
        Output returned by ``run_training_pipeline``.
    output_dir:
        Root directory for generated reports and figures.
    generate_shap:
        Whether SHAP tables and figures should be generated. This can be set
        to False in fast automated smoke tests.
    shap_max_samples:
        Maximum number of test rows used for global SHAP calculations.
    shap_top_n:
        Number of features shown in the global SHAP importance bar plot.
    local_shap_max_display:
        Number of features shown in each local SHAP explanation.
    threshold:
        Classification threshold recorded in the prediction output.
        The current sklearn model predictions still use their standard
        prediction behavior, normally equivalent to threshold 0.5.
    sprint:
        Metadata value stored in prediction outputs.
    model_types:
        Optional mapping from model names to categories such as baseline,
        interpretable, and black_box.

    Returns
    -------
    EvaluationPipelineResult
        Test metrics, prediction outputs, explanation information, and paths
        to every generated report and figure.
    """
    _validate_training_result(training_result)

    output_dir = Path(output_dir)
    reports_dir = output_dir / "reports"
    figures_dir = output_dir / "figures"
    explanations_dir = figures_dir / "explanations"

    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    explanations_dir.mkdir(parents=True, exist_ok=True)

    if model_types is None:
        model_types = DEFAULT_MODEL_TYPES.copy()

    predictions_by_model: dict[str, np.ndarray] = {}
    probabilities_by_model: dict[str, np.ndarray] = {}
    metric_results = []

    # Evaluate each fitted model on exactly the same held-out test data.
    for model_name, model in training_result.models.items():
        y_pred = np.asarray(
            model.predict(training_result.X_test)
        )

        y_score = _positive_class_probability(
            model,
            training_result.X_test,
        )

        predictions_by_model[model_name] = y_pred
        probabilities_by_model[model_name] = y_score

        metric_results.append(
            _evaluate_predictions(
                model_name=model_name,
                y_true=training_result.y_test,
                y_pred=y_pred,
                y_score=y_score,
            )
        )

    test_comparison = pd.DataFrame(
        metric_results,
        columns=[
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
        ],
    )

    predictions = _create_all_prediction_outputs(
        training_result=training_result,
        predictions_by_model=predictions_by_model,
        probabilities_by_model=probabilities_by_model,
        model_types=model_types,
        threshold=threshold,
        sprint=sprint,
    )

    report_paths: dict[str, Path] = {}
    figure_paths: dict[str, Path] = {}
    local_shap_paths: dict[str, Path] = {}

    # Save tabular outputs.
    test_comparison_path = reports_dir / "test_model_comparison.csv"
    test_comparison.to_csv(
        test_comparison_path,
        index=False,
    )
    report_paths["test_model_comparison"] = test_comparison_path

    predictions_path = reports_dir / "test_predictions.csv"
    predictions.to_csv(
        predictions_path,
        index=False,
    )
    report_paths["test_predictions"] = predictions_path

    validation_comparison_path = (
        reports_dir / "validation_model_comparison.csv"
    )
    training_result.validation_comparison.to_csv(
        validation_comparison_path,
        index=False,
    )
    report_paths[
        "validation_model_comparison"
    ] = validation_comparison_path

    random_forest_selection_path = (
        reports_dir / "random_forest_validation_results.csv"
    )
    training_result.random_forest_validation_results.to_csv(
        random_forest_selection_path,
        index=False,
    )
    report_paths[
        "random_forest_validation_results"
    ] = random_forest_selection_path

    # Save general evaluation figures.
    comparison_plot_path = save_model_comparison_plot(
        evaluation_table=test_comparison,
        output_path=figures_dir / "test_model_comparison.png",
        metrics=["f1", "roc_auc", "pr_auc"],
        title="Test-Set Model Comparison",
    )
    figure_paths["model_comparison"] = comparison_plot_path

    roc_models = {
        model_name: probabilities_by_model[model_name]
        for model_name in [
            "Logistic Regression",
            "Random Forest",
        ]
        if model_name in probabilities_by_model
    }

    if roc_models:
        roc_curve_path = save_roc_curves(
            model_scores=roc_models,
            y_true=training_result.y_test,
            output_path=figures_dir / "test_roc_curves.png",
        )
        figure_paths["roc_curves"] = roc_curve_path

    random_forest_predictions = predictions_by_model["Random Forest"]
    random_forest_probabilities = probabilities_by_model["Random Forest"]

    confusion_matrix_path = save_confusion_matrix_plot(
        y_true=training_result.y_test,
        y_pred=random_forest_predictions,
        model_name="Random Forest",
        output_path=(
            figures_dir
            / "random_forest_confusion_matrix.png"
        ),
    )
    figure_paths[
        "random_forest_confusion_matrix"
    ] = confusion_matrix_path

    shap_importance: pd.DataFrame | None = None
    local_explanations = pd.DataFrame(
        columns=[
            "explanation_type",
            "row_index",
            "case_id",
            "y_true",
            "prediction",
            "probability",
            "figure_path",
        ]
    )

    if generate_shap:
        random_forest_model = training_result.models[
            "Random Forest"
        ]

        shap_summary_path = save_global_shap_summary(
            model=random_forest_model,
            X=training_result.X_test,
            feature_columns=training_result.feature_columns,
            output_path=(
                explanations_dir
                / "shap_summary.png"
            ),
            max_samples=shap_max_samples,
        )
        figure_paths["shap_summary"] = shap_summary_path

        shap_importance_plot_path = (
            save_global_shap_importance_plot(
                model=random_forest_model,
                X=training_result.X_test,
                feature_columns=(
                    training_result.feature_columns
                ),
                output_path=(
                    explanations_dir
                    / "shap_importance.png"
                ),
                max_samples=shap_max_samples,
                top_n=shap_top_n,
            )
        )
        figure_paths[
            "shap_importance"
        ] = shap_importance_plot_path

        shap_importance_path = (
            reports_dir / "shap_feature_importance.csv"
        )
        shap_importance = save_global_shap_importance_table(
            model=random_forest_model,
            X=training_result.X_test,
            feature_columns=training_result.feature_columns,
            output_path=shap_importance_path,
            max_samples=shap_max_samples,
        )
        report_paths[
            "shap_feature_importance"
        ] = shap_importance_path

        (
            local_shap_paths,
            local_explanations,
        ) = _create_local_shap_explanations(
            model=random_forest_model,
            X_test=training_result.X_test,
            feature_columns=training_result.feature_columns,
            case_ids=training_result.case_ids_test,
            y_true=training_result.y_test,
            y_pred=random_forest_predictions,
            y_score=random_forest_probabilities,
            output_dir=explanations_dir,
            max_display=local_shap_max_display,
        )

        local_explanations_path = (
            reports_dir / "local_explanation_cases.csv"
        )
        local_explanations.to_csv(
            local_explanations_path,
            index=False,
        )
        report_paths[
            "local_explanation_cases"
        ] = local_explanations_path

        for explanation_type, path in (
            local_shap_paths.items()
        ):
            figure_paths[
                f"shap_local_{explanation_type}"
            ] = path

    return EvaluationPipelineResult(
        test_comparison=test_comparison,
        predictions=predictions,
        local_explanations=local_explanations,
        shap_importance=shap_importance,
        predictions_by_model=predictions_by_model,
        probabilities_by_model=probabilities_by_model,
        report_paths=report_paths,
        figure_paths=figure_paths,
        local_shap_paths=local_shap_paths,
    )

