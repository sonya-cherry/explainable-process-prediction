import numpy as np

from src.modeling.model_selection import select_best_random_forest


X_TRAIN = np.array(
    [
        [0.0, 1.0, 2.0],
        [1.0, 0.0, 1.0],
        [2.0, 1.0, 0.0],
        [3.0, 1.0, 1.0],
        [4.0, 0.0, 2.0],
        [5.0, 1.0, 3.0],
    ]
)
Y_TRAIN = np.array([0, 0, 0, 1, 1, 1])

X_VAL = np.array(
    [
        [0.5, 1.0, 2.0],
        [1.5, 0.0, 1.0],
        [4.5, 0.0, 2.0],
        [5.5, 1.0, 3.0],
    ]
)
Y_VAL = np.array([0, 0, 1, 1])


def test_select_best_random_forest_returns_model_params_and_results_table():
    param_grid = [
        {"n_estimators": 10, "max_depth": 2},
        {"n_estimators": 20, "max_depth": 3},
    ]

    best_model, best_params, results_table = select_best_random_forest(
        X_TRAIN,
        Y_TRAIN,
        X_VAL,
        Y_VAL,
        param_grid=param_grid,
    )

    assert hasattr(best_model, "predict")
    assert hasattr(best_model, "predict_proba")
    assert isinstance(best_params, dict)
    assert len(results_table) == len(param_grid)

    expected_columns = {
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "params",
    }
    assert expected_columns.issubset(results_table.columns)
