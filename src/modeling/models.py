from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


RANDOM_STATE = 42


def train_logistic_regression(X_train, y_train, **params):
    """
    Train an interpretable logistic regression model.

    Logistic regression is used as the Sprint 2 white-box model. The model
    returns class probabilities, which are needed for ROC AUC computation and
    structured prediction output.

    Parameters
    ----------
    X_train : array-like
        Training feature matrix.
    y_train : array-like
        Binary training labels.
    **params
        Optional hyperparameters passed to sklearn's LogisticRegression.

    Returns
    -------
    LogisticRegression
        Fitted logistic regression model.
    """
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        **params,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, **params):
    """
    Train a random forest classifier.

    Random forest is used as the Sprint 2 black-box model. Hyperparameters are
    passed through **params so that model selection can compare different
    configurations on the validation set.

    Parameters
    ----------
    X_train : array-like
        Training feature matrix.
    y_train : array-like
        Binary training labels.
    **params
        Optional hyperparameters passed to sklearn's RandomForestClassifier.

    Returns
    -------
    RandomForestClassifier
        Fitted random forest model.
    """
    model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
        **params,
    )
    model.fit(X_train, y_train)
    return model
