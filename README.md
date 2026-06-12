# Explainable Process Prediction

Outcome prediction and explainability for event log data, using the BPI
Challenge 2013 incidents log as the project dataset.

The project predicts binary process outcomes from event log data. The current
main workflow uses prefix-level case representations: original cases are split
into train, validation, and test sets, prefixes are generated inside each split,
models are trained on aligned encoded prefix features, and Random Forest
predictions are explained with SHAP.

## Project Structure

```text
src/dataExtraction/        Event log loading helpers
src/featureEngineering/    Outcome labels, feature encoding, and prefix generation
src/modeling/              Baseline, model training, selection, and evaluation
src/output/                Structured prediction CSVs and output plots
src/explainability/        SHAP global and local explanation utilities
src/prototype/             Minimal prediction-and-explanation prototype
tests/                     Automated tests for Sprint 3 components
notebooks/                 Exploration, training, and explanation notebooks
```

## Setup

Create and activate a Python environment, then install the project
dependencies used by the notebooks and tests:

```bash
python -m pip install -r requirements.txt
```

Place the BPI Challenge 2013 incidents event log under:

```text
data/raw/BPI_Challenge_2013_incidents/
```

The repository currently contains generated reports, figures, and model
artifacts from earlier sprint runs. Regenerate outputs from the notebooks or
modules instead of editing those files manually.

## Running Tests

Run the full automated test suite from the repository root:

```bash
.venv/bin/pytest tests/ -v
```

Or run a single file:

```bash
.venv/bin/pytest tests/test_evaluation.py -v
.venv/bin/pytest tests/test_explainability.py -v
```

The suite contains 23 tests across 7 files:

| File | What is covered |
|---|---|
| `test_baseline.py` | Majority baseline always predicts the most frequent class |
| `test_evaluation.py` | `compute_basic_metrics`, `compute_metrics_with_roc_auc`, `evaluate_model` (direct and without `predict_proba`), `save_confusion_matrix_plot`, `save_roc_curve`, `save_roc_curves` (multi-model), `build_evaluation_table`, `save_model_comparison_plot` |
| `test_explainability.py` | `prepare_shap_dataframe` (feature names, sampling), `save_global_shap_summary`, `save_global_shap_importance_table`, `save_global_shap_importance_plot`, `save_local_shap_bar_plot` (normal and out-of-bounds index), `select_local_explanation_indices` (mixed predictions and all-correct edge case) |
| `test_model_selection.py` | `select_best_random_forest` returns the best model, its parameters, and a results table |
| `test_pipeline.py` | Full training + evaluation pipeline on a synthetic log (SHAP disabled for speed) |
| `test_prediction_output.py` | `create_prediction_output` metadata, probability validation, `save_model_prediction_output`, `load_prediction_output` round-trip |
| `test_prototype_predict_explain.py` | `predict_and_explain_case` returns prediction, probability, and top features; rejects multi-row input |

## Current Pipeline

The final reusable pipeline is implemented in `src/pipeline/` and demonstrated in:

```text
notebooks/7-sprint3FinalPipeline.ipynb
```

It performs the following steps:

1. Load the BPI Challenge 2013 incidents log.
2. Define a binary outcome from the final lifecycle transition (`Closed`/`Resolved` → positive).
3. Split original cases temporally into train (70 %), validation (15 %), and test (15 %) sets.
4. Generate prefixes separately inside each split to prevent case leakage.
5. Encode prefix logs into aligned feature matrices (train defines the feature space).
6. Remove leakage-prone features (`relative_age_*`).
7. Train majority baseline, Logistic Regression, and Random Forest; select RF hyperparameters on the validation set.
8. Evaluate all models once on the held-out test set.
9. Export structured prediction CSVs, model comparison and ROC plots, confusion matrix, and global + local SHAP explanations.
10. Demonstrate the prototype single-case prediction and explanation interface.

The same pipeline can also be run from the command line:

```bash
python scripts/run_pipeline.py \
    --data data/raw/BPI_Challenge_2013_incidents/BPI_Challenge_2013_incidents.xes \
    --output-dir outputs
```

Generated outputs land under `outputs/reports/` and `outputs/figures/`.

## Training And Evaluation

Model training helpers are in `src/modeling/models.py`.

```python
from src.modeling.models import train_logistic_regression, train_random_forest

log_reg = train_logistic_regression(X_train, y_train)
rf = train_random_forest(X_train, y_train, n_estimators=200, max_depth=10)
```

Evaluation helpers are in `src/modeling/evaluation.py`.

```python
from src.modeling.evaluation import (
    build_evaluation_table,
    evaluate_model,
    save_confusion_matrix_plot,
    save_model_comparison_plot,
    save_roc_curve,
    save_roc_curves,
)

rf_metrics = evaluate_model(rf, X_test, y_test, "Random Forest")
comparison = build_evaluation_table([rf_metrics])
save_model_comparison_plot(comparison, "figures/model_comparison_sprint3.png")

y_pred = rf.predict(X_test)
y_score = rf.predict_proba(X_test)[:, 1]
save_roc_curve(y_test, y_score, "Random Forest", "figures/roc_curve_sprint3.png")
save_confusion_matrix_plot(
    y_test,
    y_pred,
    "Random Forest",
    "figures/confusion_matrix_sprint3.png",
)

save_roc_curves(
    {
        "Logistic Regression": log_reg.predict_proba(X_test)[:, 1],
        "Random Forest": rf.predict_proba(X_test)[:, 1],
    },
    y_test,
    "figures/combined_roc_curves.png",
)
```

## Prediction Outputs

Structured prediction outputs are created by `src/output/prediction_output.py`.
Use `save_model_prediction_output` after a model has been trained:

```python
from src.output.prediction_output import save_model_prediction_output

prediction_output = save_model_prediction_output(
    model=rf,
    features=X_test,
    case_ids=test_case_ids,
    y_true=y_test,
    model_name="Random Forest",
    model_type="black_box",
    dataset_split="test",
    sprint="sprint3",
    output_path="reports/predictions_sprint3.csv",
)
```

The output includes `case_id`, `y_true`, `prediction`, `probability`, `model`,
`model_type`, `dataset_split`, `threshold`, and `sprint`.

## Prefix-Level Prediction

Prefix generation is implemented in `src/featureEngineering/prefix_generation.py`.
The function removes the full-length trace from the generated prefix set to
reduce direct target leakage.

```python
from src.featureEngineering.prefix_generation import generate_prefix

train_prefix_df = generate_prefix(train_df)
val_prefix_df = generate_prefix(val_df)
test_prefix_df = generate_prefix(test_df)
```

Important: split original cases before calling `generate_prefix(...)`. This
keeps all prefixes from an original case inside one split. The generated
`case:concept:name` values are rewritten as prefix IDs such as
`original_case_prefix_3`, and `prefix_length` is added as a feature.

## SHAP Explanations

SHAP utilities are in `src/explainability/shap_explainer.py`. For the current
prefix workflow, use the selected Random Forest together with the aligned
prefix-level test matrix and feature names.

```python
from src.explainability.shap_explainer import (
    save_global_shap_importance_plot,
    save_global_shap_importance_table,
    save_global_shap_summary,
    save_selected_local_explanations,
)

save_global_shap_summary(
    rf,
    X_test,
    feature_columns,
    "figures/explanations/shap_summary.png",
    max_samples=500,
)
save_global_shap_importance_table(
    rf,
    X_test,
    feature_columns,
    "reports/shap_feature_importance.csv",
)
save_global_shap_importance_plot(
    rf,
    X_test,
    feature_columns,
    "figures/explanations/shap_importance_bar.png",
)
save_selected_local_explanations(
    rf,
    X_test,
    feature_columns,
    y_test,
    rf.predict(X_test),
    "figures/explanations",
)
```

## Prototype Prediction And Explanation

The minimal Sprint 3 prototype is available in
`src/prototype/predict_explain.py`.

```python
from src.prototype.predict_explain import predict_and_explain_case

result = predict_and_explain_case(
    model=rf,
    case_features=X_test[0],
    feature_columns=feature_columns,
    case_id=test_case_ids[0],
    top_n=5,
    local_plot_path="figures/explanations/prototype_case_0.png",
)
```

The function returns the predicted class, positive-class probability, top local
SHAP contributors, and the optional local plot path.

## Notebook Workflow

| Notebook | Purpose |
|---|---|
| `1-dataExploration.ipynb` | Initial data exploration |
| `2-baselinePipeline.ipynb` | Majority baseline |
| `3-training.ipynb` | Early model training experiments |
| `4-ShapExplanationPipeline.ipynb` | Early SHAP experiments |
| `5-trainingWithPrefixes.ipynb` | Prefix generation and prefix-level model training |
| `6-sprint2FinalPipeline.ipynb` | Sprint 2 final demo |
| `7-sprint3FinalPipeline.ipynb` | **Sprint 3 final pipeline — start here for the complete end-to-end demo** |

All notebooks call reusable functions from `src/` and display the generated tables, plots, and explanations.

## Notes

Prefix-level prediction is harder than completed-case prediction. The majority
baseline can look strong on accuracy and F1 when prefix labels are imbalanced,
so ROC AUC and PR AUC should be the primary comparison metrics.

Future-looking fields and full-trace information must be excluded from training,
evaluation, and SHAP inputs. Prefixes must be generated after splitting original
cases to prevent case leakage across splits.
