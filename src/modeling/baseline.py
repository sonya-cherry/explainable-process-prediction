from sklearn.dummy import DummyClassifier
"""
Baseline models for initial process outcome prediction experiments.
"""


def train_majority_baseline(X_train, y_train) -> DummyClassifier:
    """
    Train a majority-class baseline classifier.

    The classifier always predicts the most frequent outcome class
    observed in the training data.
    """
    model = DummyClassifier(strategy="most_frequent", random_state=42)
    model.fit(X=X_train, y=y_train)
    
    return model

