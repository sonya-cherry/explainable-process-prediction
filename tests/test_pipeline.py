

"""Integration test for the reusable training and evaluation pipelines.

The existing test suite checks individual modules separately. This file adds
one lightweight smoke test that verifies that the modules can also work
together as a complete pipeline.

SHAP is deliberately disabled here because it has separate tests and would
make this integration test unnecessarily slow.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.pipeline.training_pipeline as training_pipeline_module
from src.pipeline import run_evaluation_pipeline, run_training_pipeline


def _build_synthetic_event_log(number_of_cases: int = 20) -> pd.DataFrame:
    """Create a small event log with ordered cases and both outcome classes.

    Every case contains four events. Cases with an even numeric index end in
    ``Closed`` and therefore receive the positive outcome. Odd-indexed cases
    end in ``Cancelled`` and receive the negative outcome.

    Case start times increase by one day. The temporal 70/15/15 split therefore
    produces 14 training cases, 3 validation cases, and 3 test cases without
    mixing events from the same original case.
    """
    rows: list[dict] = []
    base_time = pd.Timestamp("2024-01-01 08:00:00", tz="UTC")

    activities = [
        "Create Incident",
        "Assign Incident",
        "Work on Incident",
        "Complete Incident",
    ]

    for case_index in range(number_of_cases):
        case_id = f"case_{case_index:02d}"
        case_start = base_time + pd.Timedelta(days=case_index)
        positive_outcome = case_index % 2 == 0

        lifecycle_transitions = [
            "Open",
            "In Progress",
            "In Progress",
            "Closed" if positive_outcome else "Cancelled",
        ]

        for event_index, (activity, lifecycle_transition) in enumerate(
            zip(activities, lifecycle_transitions)
        ):
            rows.append(
                {
                    "case:concept:name": case_id,
                    "concept:name": activity,
                    "lifecycle:transition": lifecycle_transition,
                    "time:timestamp": (
                        case_start + pd.Timedelta(hours=event_index)
                    ),
                    "org:resource": (
                        "support_team_a"
                        if case_index % 2 == 0
                        else "support_team_b"
                    ),
                    "priority": (
                        "high" if case_index % 3 == 0 else "normal"
                    ),
                }
            )

    return pd.DataFrame(rows)


def test_complete_pipeline_runs_on_synthetic_log(
    tmp_path,
    monkeypatch,
):
    """Training and evaluation should run together and create valid outputs."""
    synthetic_log = _build_synthetic_event_log()

    # ``run_training_pipeline`` checks that the supplied path exists before it
    # calls the loader. The loader itself is patched so this integration test
    # does not need to store or parse a real XES file.
    placeholder_log_path = tmp_path / "synthetic_event_log.xes"
    placeholder_log_path.write_text("synthetic test log", encoding="utf-8")

    def fake_import_data(file_path, drop_columns=None, **kwargs):
        assert Path(file_path) == placeholder_log_path
        return synthetic_log.copy()

    monkeypatch.setattr(
        training_pipeline_module,
        "import_data",
        fake_import_data,
    )

    training_result = run_training_pipeline(
        event_log_path=placeholder_log_path,
        min_prefix=1,
        random_forest_param_grid=[
            {"n_estimators": 5, "max_depth": 3}
        ],
    )

    # Check that all expected models were trained.
    assert set(training_result.models) == {
        "Majority baseline",
        "Logistic Regression",
        "Random Forest",
    }
    assert training_result.best_random_forest_parameters == {
        "n_estimators": 5,
        "max_depth": 3,
    }

    # Twenty chronologically ordered cases should be split as 14/3/3.
    assert training_result.dataset_summary["train_original_cases"] == 14
    assert training_result.dataset_summary["validation_original_cases"] == 3
    assert training_result.dataset_summary["test_original_cases"] == 3

    # Four-event traces produce prefixes of lengths 1, 2, and 3.
    assert training_result.dataset_summary["train_prefixes"] == 42
    assert training_result.dataset_summary["validation_prefixes"] == 9
    assert training_result.dataset_summary["test_prefixes"] == 9

    # Features, labels, and IDs must remain aligned in every split.
    for X, y, case_ids in [
        (
            training_result.X_train,
            training_result.y_train,
            training_result.case_ids_train,
        ),
        (
            training_result.X_validation,
            training_result.y_validation,
            training_result.case_ids_validation,
        ),
        (
            training_result.X_test,
            training_result.y_test,
            training_result.case_ids_test,
        ),
    ]:
        assert X.shape[0] == len(y) == len(case_ids)
        assert X.shape[1] == len(training_result.feature_columns)
        assert not np.isnan(X.data).any()
        assert not y.isna().any()
        assert all("_prefix_" in str(case_id) for case_id in case_ids)

    output_dir = tmp_path / "pipeline_outputs"

    evaluation_result = run_evaluation_pipeline(
        training_result=training_result,
        output_dir=output_dir,
        generate_shap=False,
    )

    # The final comparison must contain one row per trained model.
    assert set(evaluation_result.test_comparison["model"]) == set(
        training_result.models
    )
    assert {
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    }.issubset(evaluation_result.test_comparison.columns)

    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]
    metric_values = evaluation_result.test_comparison[metric_columns]
    assert metric_values.notna().all().all()
    assert ((metric_values >= 0) & (metric_values <= 1)).all().all()

    # Each test prefix should have one prediction row for every model.
    expected_prediction_rows = (
        training_result.X_test.shape[0]
        * len(training_result.models)
    )
    assert len(evaluation_result.predictions) == expected_prediction_rows
    assert evaluation_result.predictions["case_id"].notna().all()
    assert evaluation_result.predictions["y_true"].notna().all()
    assert evaluation_result.predictions["prediction"].notna().all()
    assert evaluation_result.predictions["probability"].between(0, 1).all()
    assert set(evaluation_result.predictions["dataset_split"]) == {"test"}
    assert set(evaluation_result.predictions["sprint"]) == {"sprint3"}

    # Evaluation should create all non-SHAP reports and figures.
    assert evaluation_result.report_paths
    assert evaluation_result.figure_paths
    assert all(path.exists() for path in evaluation_result.report_paths.values())
    assert all(path.exists() for path in evaluation_result.figure_paths.values())

    # SHAP was intentionally disabled for this fast smoke test.
    assert evaluation_result.shap_importance is None
    assert evaluation_result.local_explanations.empty
    assert evaluation_result.local_shap_paths == {}
