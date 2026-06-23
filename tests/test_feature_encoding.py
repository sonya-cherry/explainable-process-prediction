import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from src.feature_engineering.feature_encoding import Encode


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "case:concept:name": [
            "C1", "C1", "C1",
            "C2", "C2"
        ],
        "concept:name": [
            "A", "B", "C",
            "A", "A"
        ],
        "org:resource": [
            "R1", "R2", "R3",
            "R1", "R1"
        ],
        "amount": [
            10, 20, 30,
            5, 15
        ],
        "time:timestamp": pd.to_datetime([
            "2024-01-01 10:00:00",
            "2024-01-01 11:00:00",
            "2024-01-01 12:00:00",
            "2024-01-02 10:00:00",
            "2024-01-02 11:00:00"
        ]),
        "outcome": [
            True, True, True,
            False, False
        ]
    })


# ============================================================================
# VALIDATION
# ============================================================================

def test_missing_case_column_raises(sample_df):
    df = sample_df.drop(columns=["case:concept:name"])

    with pytest.raises(ValueError, match="Missing required columns"):
        Encode(df)


def test_missing_outcome_column_raises(sample_df):
    df = sample_df.drop(columns=["outcome"])

    with pytest.raises(ValueError, match="Missing required columns"):
        Encode(df)


# ============================================================================
# OUTPUT STRUCTURE
# ============================================================================

def test_returns_sparse_matrix(sample_df):
    X, y, case_ids, feature_columns = Encode(sample_df)

    assert isinstance(X, csr_matrix)


def test_one_row_per_case(sample_df):
    X, y, case_ids, feature_columns = Encode(sample_df)

    assert X.shape[0] == 2


def test_case_id_alignment(sample_df):
    X, y, case_ids, feature_columns = Encode(sample_df)

    assert list(case_ids) == ["C1", "C2"]


def test_outcome_alignment(sample_df):
    X, y, case_ids, feature_columns = Encode(sample_df)

    assert list(y) == [True, False]


# ============================================================================
# NUMERIC FEATURES
# ============================================================================

def test_numeric_aggregations(sample_df):
    X, y, case_ids, feature_columns = Encode(sample_df)

    encoded = pd.DataFrame(
        X.toarray(),
        index=case_ids,
        columns=feature_columns
    )

    assert encoded.loc["C1", "amount_sum"] == 60
    assert encoded.loc["C1", "amount_mean"] == 20
    assert encoded.loc["C1", "amount_min"] == 10
    assert encoded.loc["C1", "amount_max"] == 30
    assert encoded.loc["C1", "amount_count"] == 3

    assert encoded.loc["C2", "amount_sum"] == 20
    assert encoded.loc["C2", "amount_count"] == 2


# ============================================================================
# TEMPORAL FEATURES
# ============================================================================

def test_temporal_features_created(sample_df):
    X, y, case_ids, feature_columns = Encode(sample_df)

    expected = [
        "elapsed_time_sum",
        "time_since_last_sum",
        "hour_mean",
        "month_mean",
        "relative_age_max",
    ]

    for col in expected:
        assert col in feature_columns


def test_no_timestamp_column(sample_df):
    df = sample_df.drop(columns=["time:timestamp"])

    X, y, case_ids, feature_columns = Encode(df)

    assert X.shape[0] == 2


# ============================================================================
# FIRST / LAST ACTIVITY
# ============================================================================

def test_first_activity_feature_exists(sample_df):
    X, y, case_ids, feature_columns = Encode(sample_df)

    assert "first_activity=A" in feature_columns


def test_last_activity_feature_exists(sample_df):
    X, y, case_ids, feature_columns = Encode(sample_df)

    assert "last_activity=C" in feature_columns


# ============================================================================
# TRANSITION FEATURES
# ============================================================================

def test_transition_encoding_concept_name(sample_df):
    X, y, case_ids, feature_columns = Encode(sample_df)

    assert "concept:name_transition=A->B" in feature_columns
    assert "concept:name_transition=B->C" in feature_columns


def test_transition_encoding_org_resource(sample_df):
    X, y, case_ids, feature_columns = Encode(sample_df)

    assert "org:resource_transition=R1->R2" in feature_columns
    assert "org:resource_transition=R2->R3" in feature_columns


# ============================================================================
# HIGH CARDINALITY PROTECTION
# ============================================================================

def test_high_cardinality_categorical_skipped():
    rows = []

    for i in range(200):
        rows.append({
            "case:concept:name": f"C{i}",
            "concept:name": f"ACT_{i}",
            "high_cardinality": f"VAL_{i}",
            "outcome": True
        })

    df = pd.DataFrame(rows)

    X, y, case_ids, feature_columns = Encode(df)

    high_cardinality_features = [
        col
        for col in feature_columns
        if col.startswith("high_cardinality=")
    ]

    assert len(high_cardinality_features) == 0


# ============================================================================
# FEATURE COLUMN REUSE
# ============================================================================

def test_feature_column_reuse():
    train = pd.DataFrame({
        "case:concept:name": ["C1", "C1"],
        "concept:name": ["A", "B"],
        "amount": [10, 20],
        "outcome": [True, True]
    })

    test = pd.DataFrame({
        "case:concept:name": ["C2", "C2"],
        "concept:name": ["A", "A"],
        "amount": [5, 5],
        "outcome": [False, False]
    })

    _, _, _, train_columns = Encode(train)

    X_test, _, _, returned_columns = Encode(
        test,
        feature_columns=train_columns
    )

    assert returned_columns == train_columns
    assert X_test.shape[1] == len(train_columns)


def test_missing_feature_columns_filled_with_zero():
    train = pd.DataFrame({
        "case:concept:name": ["C1", "C1"],
        "concept:name": ["A", "D"],
        "outcome": [True, True]
    })

    test = pd.DataFrame({
        "case:concept:name": ["C2", "C2"],
        "concept:name": ["A", "A"],
        "outcome": [False, False]
    })

    _, _, _, train_columns = Encode(train)

    X_test, _, case_ids, _ = Encode(
        test,
        feature_columns=train_columns
    )

    encoded = pd.DataFrame(
        X_test.toarray(),
        index=case_ids,
        columns=train_columns
    )

    if "concept:name=D" in encoded.columns:
        assert encoded.loc["C2", "concept:name=D"] == 0


# ============================================================================
# EDGE CASES
# ============================================================================

def test_single_event_trace():
    df = pd.DataFrame({
        "case:concept:name": ["C1"],
        "concept:name": ["A"],
        "outcome": [True]
    })

    X, y, case_ids, feature_columns = Encode(df)

    assert X.shape[0] == 1
    assert list(case_ids) == ["C1"]
    assert list(y) == [True]