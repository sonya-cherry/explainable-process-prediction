# Explainable Process Prediction

Outcome prediction and explainability for event log data, using the BPI
Challenge 2013 incidents log as the project dataset.

The project predicts a binary process outcome from encoded case features,
evaluates baseline and machine-learning models, saves structured prediction
outputs, and explains Random Forest predictions with SHAP.

## Project Structure

```text
src/dataExtraction/        Event log loading helpers
src/featureEngineering/    Outcome labels, feature encoding, and prefixes
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

The repository currently contains local generated reports, figures, and model
artifacts from earlier sprints. For a clean run, regenerate outputs from the
notebooks or modules instead of editing those files manually.

## Running Tests

Run the automated test suite from the repository root:

```bash
pytest
```

Sprint 3 adds coverage for:

- prediction output metadata and validation,
- ROC AUC computation,
- confusion matrix and ROC figure creation,
- model comparison table and plot creation,
- SHAP explanation generation on a small Random Forest,
- the prototype prediction-and-explanation wrapper.

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

## SHAP Explanations

SHAP utilities are in `src/explainability/shap_explainer.py`. For Sprint 3,
use the final no-leakage Random Forest model together with the aligned
no-leakage feature matrix and feature names.

```python
from src.explainability.shap_explainer import (
    save_global_shap_importance_plot,
    save_global_shap_importance_table,
    save_global_shap_summary,
    save_selected_local_explanations,
)

save_global_shap_summary(
    rf,
    X_test_no_leakage,
    feature_columns_no_leakage,
    "figures/explanations/shap_summary_sprint3.png",
    max_samples=500,
)
save_global_shap_importance_table(
    rf,
    X_test_no_leakage,
    feature_columns_no_leakage,
    "reports/shap_feature_importance_sprint3.csv",
)
save_global_shap_importance_plot(
    rf,
    X_test_no_leakage,
    feature_columns_no_leakage,
    "figures/explanations/shap_importance_bar_sprint3.png",
)
save_selected_local_explanations(
    rf,
    X_test_no_leakage,
    feature_columns_no_leakage,
    y_test,
    rf.predict(X_test_no_leakage),
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
    case_features=X_test_no_leakage.iloc[[0]],
    feature_columns=feature_columns_no_leakage,
    case_id=test_case_ids[0],
    top_n=5,
    local_plot_path="figures/explanations/prototype_case_0.png",
)
```

The function returns the predicted class, positive-class probability, top local
SHAP contributors, and the optional local plot path.

## Notebook Workflow

Suggested notebook order:

1. `notebooks/1-dataExploration.ipynb`
2. `notebooks/2-baselinePipeline.ipynb`
3. `notebooks/3-training.ipynb`
4. `notebooks/4-ShapExplanationPipeline.ipynb`
5. `notebooks/5-sprint3Demo.ipynb`

Notebooks should call reusable functions from `src/` and mainly display the
generated tables, plots, and explanations.

## Known Limitations

The current Sprint 3 code path is best described as completed-case outcome
classification with no-leakage feature variants. Full prefix-level prediction
is not yet finalized. If prefix rows are used, all prefixes from the same
original case must stay in the same train, validation, or test split to avoid
case leakage.

Duration-derived and future-looking fields should be excluded from the final
no-leakage training, evaluation, prediction output, and SHAP explanation inputs.
