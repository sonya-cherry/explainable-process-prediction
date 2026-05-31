"""
Model selection utilities for Sprint 2.

This module compares several Random Forest configurations on a validation set
and selects the best one according to a chosen metric.
"""

from typing import Any

import pandas as pd

from src.modeling.evaluation import evaluate_model
from src.modeling.models import train_random_forest


# Small grid for Sprint 2. The goal is not exhaustive tuning, but a clear
# validation-based comparison of several reasonable configurations.
RANDOM_FOREST_PARAM_GRID = [
    {"n_estimators": 100, "max_depth": None},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 100, "max_depth": 10},
    {"n_estimators": 200, "max_depth": None},
    {"n_estimators": 200, "max_depth": 5},
    {"n_estimators": 200, "max_depth": 10},
]


def select_best_random_forest(
    X_train,
    y_train,
    X_val,
    y_val,
    scoring: str = "f1",
    param_grid: list[dict[str, Any]] | None = None,
) -> tuple[Any, dict[str, Any], pd.DataFrame]:
    """
    Train and compare Random Forest configurations on a validation set.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        X_val: Validation feature matrix.
        y_val: Validation labels.
        scoring: Metric used to select the best configuration.
            Expected values are columns returned by evaluate_model,
            for example "f1", "roc_auc", "accuracy", "precision", or "recall".
        param_grid: Optional list of Random Forest hyperparameter dictionaries.
            If None, RANDOM_FOREST_PARAM_GRID is used.

    Returns:
        Tuple containing:
        - best fitted Random Forest model
        - best hyperparameter dictionary
        - DataFrame with validation metrics for all tested configurations
    """
    if param_grid is None:
        param_grid = RANDOM_FOREST_PARAM_GRID

    validation_results = []
    fitted_models = []

    for params in param_grid:
        model = train_random_forest(X_train, y_train, **params)

        model_name = f"Random Forest {params}"
        metrics = evaluate_model(model, X_val, y_val, model_name)

        # Store parameters explicitly so the selected configuration can be
        # documented in the report or README.
        metrics["params"] = params

        validation_results.append(metrics)
        fitted_models.append(model)

    results_table = pd.DataFrame(validation_results)

    if scoring not in results_table.columns:
        raise ValueError(
            f"Unknown scoring metric '{scoring}'. "
            f"Available metrics are: {list(results_table.columns)}"
        )

    best_index = results_table[scoring].idxmax()
    best_model = fitted_models[best_index]
    best_params = results_table.loc[best_index, "params"]

    return best_model, best_params, results_table


