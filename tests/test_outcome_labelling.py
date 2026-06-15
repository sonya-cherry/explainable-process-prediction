"""
tests/test_outcome.py

Tests for outcome.py — the outcome() labelling function.

Run all:   pytest test_outcome.py -v
"""

import pandas as pd
import pytest

from src.featureEngineering.outcome_labelling import outcome


# ===========================================================================
# HELPERS & FIXTURES
# ===========================================================================

def _make_df(cases: dict, extra_cols: dict = None) -> pd.DataFrame:
    """
    Build a minimal event-log DataFrame from {case_id: [activity, ...]}
    and optional flat extra columns.

        _make_df({"C1": ["A", "B"], "C2": ["A", "A"]}, {"amount": [10, 20, 5, 5]})
    """
    rows = []
    for case_id, activities in cases.items():
        for act in activities:
            rows.append({"case:concept:name": case_id, "concept:name": act})
    df = pd.DataFrame(rows)
    if extra_cols:
        for col, vals in extra_cols.items():
            df[col] = vals
    return df


def _case_outcomes(result: pd.DataFrame) -> dict:
    """
    Collapse a result DataFrame to {case_id: outcome_bool} using the
    first row of each case (all rows in a case must share the same value).
    """
    return result.groupby("case:concept:name")["outcome"].first().to_dict()


@pytest.fixture()
def sample_df():
    """
    C1: activities A → B → C,  amounts 10, 20, 30  (varied)
    C2: activities A → A → A,  amounts  5,  5,  5  (homogeneous)

    Designed so that for virtually every operator one case passes and
    the other fails, making each test a genuine discriminating assertion.
    """
    return pd.DataFrame({
        "case:concept:name": ["C1", "C1", "C1", "C2", "C2", "C2"],
        "concept:name":      ["A",  "B",  "C",  "A",  "A",  "A"],
        "amount":            [10,   20,   30,    5,    5,    5],
    })


# ===========================================================================
# VALIDATION
# ===========================================================================

class TestValidation:

    def test_unknown_operator_raises(self, sample_df):
        with pytest.raises(ValueError, match="not recognised"):
            outcome(sample_df, [{"feature": "amount", "operator": "banana", "value": 10}])

    def test_unknown_feature_raises(self, sample_df):
        with pytest.raises(ValueError, match="not recognised"):
            outcome(sample_df, [{"feature": "does_not_exist", "operator": "ever_gt", "value": 10}])

    def test_empty_rules_all_true(self, sample_df):
        """No rules → every row gets outcome=True."""
        result = outcome(sample_df, [])
        assert result["outcome"].all()

    def test_output_contains_outcome_column(self, sample_df):
        result = outcome(sample_df, [])
        assert "outcome" in result.columns

    def test_outcome_column_is_boolean(self, sample_df):
        result = outcome(sample_df, [{"feature": "amount", "operator": "ever_gt", "value": 5}])
        assert result["outcome"].dtype == bool

    def test_original_dataframe_not_modified(self, sample_df):
        original = sample_df.copy(deep=True)
        outcome(sample_df, [])
        pd.testing.assert_frame_equal(sample_df, original)

    def test_returns_dataframe(self, sample_df):
        result = outcome(sample_df, [{"feature": "amount", "operator": "ever_gt", "value": 5}])
        assert isinstance(result, pd.DataFrame)


# ===========================================================================
# ALWAYS operators  (condition must hold for every event in the case)
# ===========================================================================

class TestAlwaysOperators:

    def test_always_gt(self, sample_df):
        result = outcome(sample_df, [{"feature": "amount", "operator": "always_gt", "value": 5}])
        # C1=[10,20,30]: all > 5 ✓   C2=[5,5,5]: 5 is not > 5 ✗
        assert _case_outcomes(result) == {"C1": True, "C2": False}

    def test_always_ge(self, sample_df):
        result = outcome(sample_df, [{"feature": "amount", "operator": "always_ge", "value": 5}])
        # C1=[10,20,30]: all ≥ 5 ✓   C2=[5,5,5]: all ≥ 5 ✓
        assert _case_outcomes(result) == {"C1": True, "C2": True}

    def test_always_lt(self, sample_df):
        result = outcome(sample_df, [{"feature": "amount", "operator": "always_lt", "value": 25}])
        # C1=[10,20,30]: 30 is not < 25 ✗   C2=[5,5,5]: all < 25 ✓
        assert _case_outcomes(result) == {"C1": False, "C2": True}

    def test_always_le(self, sample_df):
        result = outcome(sample_df, [{"feature": "amount", "operator": "always_le", "value": 30}])
        # C1=[10,20,30]: all ≤ 30 ✓   C2=[5,5,5]: all ≤ 30 ✓
        assert _case_outcomes(result) == {"C1": True, "C2": True}

    def test_always_eq(self, sample_df):
        result = outcome(sample_df, [{"feature": "amount", "operator": "always_eq", "value": 5}])
        # C1=[10,20,30]: none == 5 ✗   C2=[5,5,5]: all == 5 ✓
        assert _case_outcomes(result) == {"C1": False, "C2": True}

    def test_always_ne(self, sample_df):
        result = outcome(sample_df, [{"feature": "amount", "operator": "always_ne", "value": 5}])
        # C1=[10,20,30]: all ≠ 5 ✓   C2=[5,5,5]: none ≠ 5 ✗
        assert _case_outcomes(result) == {"C1": True, "C2": False}


# ===========================================================================
# EVER operators  (condition must hold for at least one event)
# ===========================================================================

class TestEverOperators:

    def test_ever_gt(self, sample_df):
        result = outcome(sample_df, [{"feature": "amount", "operator": "ever_gt", "value": 25}])
        # C1 has 30 > 25 ✓   C2 max=5, never > 25 ✗
        assert _case_outcomes(result) == {"C1": True, "C2": False}

    def test_ever_lt(self, sample_df):
        result = outcome(sample_df, [{"feature": "amount", "operator": "ever_lt", "value": 6}])
        # C1 min=10, never < 6 ✗   C2 has 5 < 6 ✓
        assert _case_outcomes(result) == {"C1": False, "C2": True}

    def test_ever_ge(self, sample_df):
        result = outcome(sample_df, [{"feature": "amount", "operator": "ever_ge", "value": 30}])
        # C1 has 30 ≥ 30 ✓   C2 max=5, never ≥ 30 ✗
        assert _case_outcomes(result) == {"C1": True, "C2": False}

    def test_ever_le(self, sample_df):
        result = outcome(sample_df, [{"feature": "amount", "operator": "ever_le", "value": 5}])
        # C1 min=10, never ≤ 5 ✗   C2 has 5 ≤ 5 ✓
        assert _case_outcomes(result) == {"C1": False, "C2": True}

    def test_ever_eq(self, sample_df):
        result = outcome(sample_df, [{"feature": "amount", "operator": "ever_eq", "value": 20}])
        # C1 has 20 ✓   C2 never has 20 ✗
        assert _case_outcomes(result) == {"C1": True, "C2": False}

    def test_ever_ne(self, sample_df):
        result = outcome(sample_df, [{"feature": "amount", "operator": "ever_ne", "value": 5}])
        # C1 has values ≠ 5 ✓   C2 is always 5, never ≠ 5 ✗
        assert _case_outcomes(result) == {"C1": True, "C2": False}


# ===========================================================================
# CONTAINS operators
# ===========================================================================

class TestContainsOperators:

    def test_always_contains(self, sample_df):
        result = outcome(sample_df, [{"feature": "concept:name", "operator": "always_contains", "value": ["A"]}])
        # C1=[A,B,C]: B and C not in ["A"] ✗   C2=[A,A,A]: all in ["A"] ✓
        assert _case_outcomes(result) == {"C1": False, "C2": True}

    def test_ever_contains(self, sample_df):
        result = outcome(sample_df, [{"feature": "concept:name", "operator": "ever_contains", "value": ["B"]}])
        # C1 has B ✓   C2 never has B ✗
        assert _case_outcomes(result) == {"C1": True, "C2": False}

    def test_never_contains(self, sample_df):
        result = outcome(sample_df, [{"feature": "concept:name", "operator": "never_contains", "value": ["B"]}])
        # C1 has B ✗   C2 never has B ✓
        assert _case_outcomes(result) == {"C1": False, "C2": True}

    def test_ever_contains_multi_value_list(self):
        """Value list with multiple entries: match on any of them."""
        df = _make_df({"C1": ["A", "B", "C"], "C2": ["X", "Y", "Z"]})
        result = outcome(df, [{"feature": "concept:name", "operator": "ever_contains", "value": ["A", "Z"]}])
        # C1 has A ✓   C2 has Z ✓   both pass
        assert _case_outcomes(result) == {"C1": True, "C2": True}

    def test_never_contains_no_match_at_all(self):
        """never_contains with a value absent from all cases → all True."""
        df = _make_df({"C1": ["A", "B"], "C2": ["C", "D"]})
        result = outcome(df, [{"feature": "concept:name", "operator": "never_contains", "value": ["Q"]}])
        assert result["outcome"].all()


# ===========================================================================
# TRACE POSITION operators
# ===========================================================================

class TestTracePositionOperators:

    def test_starts_with(self, sample_df):
        result = outcome(sample_df, [{"feature": "concept:name", "operator": "starts_with", "value": "A"}])
        # both C1 and C2 start with A ✓
        assert result["outcome"].all()

    def test_ends_with(self, sample_df):
        result = outcome(sample_df, [{"feature": "concept:name", "operator": "ends_with", "value": "C"}])
        # C1 ends with C ✓   C2 ends with A ✗
        assert _case_outcomes(result) == {"C1": True, "C2": False}

    def test_not_starts_with(self, sample_df):
        result = outcome(sample_df, [{"feature": "concept:name", "operator": "not_starts_with", "value": "A"}])
        # both start with A, so not_starts_with A → both False
        assert not result["outcome"].any()

    def test_not_ends_with(self, sample_df):
        result = outcome(sample_df, [{"feature": "concept:name", "operator": "not_ends_with", "value": "C"}])
        # C1 ends with C ✗   C2 ends with A ✓
        assert _case_outcomes(result) == {"C1": False, "C2": True}


# ===========================================================================
# STRUCTURE operators
# ===========================================================================

class TestStructureOperators:

    def test_all_identical(self, sample_df):
        result = outcome(sample_df, [{"feature": "concept:name", "operator": "all_identical"}])
        # C1=[A,B,C]: 3 unique values ✗   C2=[A,A,A]: 1 unique value ✓
        assert _case_outcomes(result) == {"C1": False, "C2": True}

    def test_all_distinct(self, sample_df):
        result = outcome(sample_df, [{"feature": "concept:name", "operator": "all_distinct"}])
        # C1=[A,B,C]: all unique ✓   C2=[A,A,A]: A repeated ✗
        assert _case_outcomes(result) == {"C1": True, "C2": False}

    def test_all_identical_single_event_is_true(self):
        """A single-event trace is trivially all-identical."""
        df = _make_df({"C1": ["A"]})
        result = outcome(df, [{"feature": "concept:name", "operator": "all_identical"}])
        assert result["outcome"].all()

    def test_all_distinct_single_event_is_true(self):
        """A single-event trace is trivially all-distinct."""
        df = _make_df({"C1": ["A"]})
        result = outcome(df, [{"feature": "concept:name", "operator": "all_distinct"}])
        assert result["outcome"].all()


# ===========================================================================
# RULE COMBINATION  (AND semantics across multiple rules)
# ===========================================================================

class TestRuleCombination:

    def test_multiple_rules_combined_with_and(self, sample_df):
        result = outcome(sample_df, [
            {"feature": "amount",       "operator": "ever_gt",  "value": 25},
            {"feature": "concept:name", "operator": "ends_with", "value": "C"},
        ])
        # C1: ever_gt 25 ✓ AND ends_with C ✓ → True
        # C2: ever_gt 25 ✗                   → False
        assert _case_outcomes(result) == {"C1": True, "C2": False}

    def test_multiple_rules_can_reject_all_cases(self, sample_df):
        result = outcome(sample_df, [
            {"feature": "amount",       "operator": "always_gt", "value": 100},
            {"feature": "concept:name", "operator": "ends_with", "value": "C"},
        ])
        # always_gt 100 fails for every case
        assert not result["outcome"].any()

    def test_failing_first_rule_overrides_passing_second(self):
        """AND semantics: a False from rule 1 cannot be rescued by rule 2."""
        df = _make_df({"C1": ["A", "B"]}, {"amount": [1, 2]})
        result = outcome(df, [
            {"feature": "amount", "operator": "always_gt", "value": 5},  # fails
            {"feature": "amount", "operator": "ever_gt",   "value": 0},  # would pass
        ])
        assert not result["outcome"].any()

    def test_rules_across_different_features(self):
        """Rules referencing different columns are ANDed correctly."""
        df = _make_df(
            {"C1": ["Start", "End"], "C2": ["Start", "Other"]},
            {"amount": [10, 20, 10, 5]},
        )
        result = outcome(df, [
            {"feature": "concept:name", "operator": "ends_with",  "value": "End"},
            {"feature": "amount",       "operator": "always_ge",  "value": 10},
        ])
        # C1: ends_with End ✓, all amounts ≥ 10 ✓ → True
        # C2: ends_with Other ✗                   → False
        assert _case_outcomes(result) == {"C1": True, "C2": False}

    def test_three_rules_all_must_pass(self, sample_df):
        result = outcome(sample_df, [
            {"feature": "concept:name", "operator": "all_identical"},
            {"feature": "concept:name", "operator": "starts_with", "value": "A"},
            {"feature": "amount",       "operator": "always_eq",   "value": 5},
        ])
        # C1: all_identical ✗ → False
        # C2: all_identical ✓, starts_with A ✓, always_eq 5 ✓ → True
        assert _case_outcomes(result) == {"C1": False, "C2": True}

    def test_duplicate_rules_are_idempotent(self, sample_df):
        """Applying the same rule twice must give the same result as once."""
        rule = {"feature": "amount", "operator": "ever_gt", "value": 25}
        once  = outcome(sample_df, [rule])
        twice = outcome(sample_df, [rule, rule])
        pd.testing.assert_series_equal(
            once["outcome"].reset_index(drop=True),
            twice["outcome"].reset_index(drop=True),
        )


# ===========================================================================
# OUTCOME UNIFORMITY INVARIANT
#
# The core contract: outcome is a case-level label. Every row belonging
# to the same case must carry the same boolean. Tested for all 21 operators.
# ===========================================================================

class TestOutcomeUniformity:

    ALL_OPS = [
        ("always_gt",       "amount",        10),
        ("always_lt",       "amount",        50),
        ("always_ge",       "amount",        10),
        ("always_le",       "amount",        50),
        ("always_eq",       "amount",        20),
        ("always_ne",       "amount",        99),
        ("ever_gt",         "amount",        25),
        ("ever_lt",         "amount",        10),
        ("ever_ge",         "amount",        30),
        ("ever_le",         "amount",         5),
        ("ever_eq",         "amount",        20),
        ("ever_ne",         "amount",        10),
        ("ever_contains",   "concept:name", ["A"]),
        ("always_contains", "concept:name", ["A"]),
        ("never_contains",  "concept:name", ["Q"]),
        ("starts_with",     "concept:name", "A"),
        ("ends_with",       "concept:name", "C"),
        ("not_starts_with", "concept:name", "X"),
        ("not_ends_with",   "concept:name", "X"),
        ("all_identical",   "concept:name",  None),
        ("all_distinct",    "concept:name",  None),
    ]

    @pytest.mark.parametrize("op,feature,value", ALL_OPS)
    def test_outcome_uniform_within_case(self, op, feature, value, sample_df):
        rule = {"feature": feature, "operator": op}
        if value is not None:
            rule["value"] = value
        result = outcome(sample_df, [rule])
        for case_id in result["case:concept:name"].unique():
            case_outcomes = result[result["case:concept:name"] == case_id]["outcome"]
            assert case_outcomes.nunique() == 1, (
                f"Operator '{op}' produced mixed per-row outcomes within case '{case_id}'"
            )