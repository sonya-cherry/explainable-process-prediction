"""
tests/test_prefix.py

Tests for prefix.py — the generate_prefix() function.

Run all:   pytest test_prefix.py -v
"""

import pandas as pd
import pytest

from src.featureEngineering.prefix_generation import generate_prefix


# ===========================================================================
# HELPERS & FIXTURES
# ===========================================================================

def _make_df(cases: dict) -> pd.DataFrame:
    """
    Build a minimal event-log DataFrame from {case_id: [activity, ...]}.
    Timestamps are assigned in order (T+0s, T+1s, ...) so sort order is
    deterministic without needing real datetimes.

        _make_df({"C1": ["A", "B", "C"], "C2": ["A", "B"]})
    """
    rows = []
    for case_id, activities in cases.items():
        for i, act in enumerate(activities):
            rows.append({
                "case:concept:name": case_id,
                "concept:name":      act,
                "time:timestamp":    pd.Timestamp("2024-01-01") + pd.Timedelta(seconds=i),
            })
    return pd.DataFrame(rows)


def _prefix_lengths_for(result: pd.DataFrame, original_case_id: str) -> list[int]:
    """Return sorted list of distinct prefix_length values for a given original case."""
    mask = result["case:concept:name"].str.startswith(f"{original_case_id}_prefix_")
    return sorted(result[mask]["prefix_length"].unique().tolist())


def _rows_for_prefix(result: pd.DataFrame, original_case_id: str, k: int) -> pd.DataFrame:
    """Return the rows belonging to <case_id>_prefix_<k>."""
    label = f"{original_case_id}_prefix_{k}"
    return result[result["case:concept:name"] == label].reset_index(drop=True)


@pytest.fixture()
def three_event_df():
    """Single case C1 with 3 events: A → B → C."""
    return _make_df({"C1": ["A", "B", "C"]})


@pytest.fixture()
def two_case_df():
    """
    C1: A → B → C  (3 events → prefixes of length 1, 2)
    C2: X → Y      (2 events → prefix  of length 1)
    """
    return _make_df({"C1": ["A", "B", "C"], "C2": ["X", "Y"]})


# ===========================================================================
# BASIC STRUCTURE
# ===========================================================================

class TestBasicStructure:

    def test_returns_dataframe(self, three_event_df):
        result = generate_prefix(three_event_df)
        assert isinstance(result, pd.DataFrame)

    def test_original_df_not_mutated(self, three_event_df):
        original = three_event_df.copy(deep=True)
        generate_prefix(three_event_df)
        pd.testing.assert_frame_equal(three_event_df, original)

    def test_prefix_length_column_added(self, three_event_df):
        result = generate_prefix(three_event_df)
        assert "prefix_length" in result.columns

    def test_output_retains_original_columns(self, three_event_df):
        result = generate_prefix(three_event_df)
        for col in three_event_df.columns:
            assert col in result.columns


# ===========================================================================
# FULL-TRACE EXCLUSION  (the core leakage-prevention contract)
# ===========================================================================

class TestFullTraceExclusion:

    def test_full_trace_not_present(self, three_event_df):
        """
        A 3-event case must never appear as a prefix of length 3.
        The function comment says full-length is removed to avoid leakage.
        """
        result = generate_prefix(three_event_df)
        assert 3 not in result["prefix_length"].values

    def test_max_prefix_length_is_n_minus_one(self, three_event_df):
        """Longest prefix is always one event shorter than the full trace."""
        result = generate_prefix(three_event_df)
        assert result["prefix_length"].max() == 2

    def test_full_trace_excluded_for_every_case(self, two_case_df):
        """Leakage prevention applies to all cases, not just the first."""
        result = generate_prefix(two_case_df)
        # C1 has 3 events → prefix_length 3 must not appear for C1
        c1_lengths = _prefix_lengths_for(result, "C1")
        assert 3 not in c1_lengths
        # C2 has 2 events → prefix_length 2 must not appear for C2
        c2_lengths = _prefix_lengths_for(result, "C2")
        assert 2 not in c2_lengths


# ===========================================================================
# PREFIX COUNT & LENGTHS
# ===========================================================================

class TestPrefixCounts:

    def test_three_event_case_produces_two_prefixes(self, three_event_df):
        """3 events → prefixes of length 1 and 2 (full trace excluded)."""
        result = generate_prefix(three_event_df)
        assert _prefix_lengths_for(result, "C1") == [1, 2]

    def test_two_event_case_produces_one_prefix(self):
        """2 events → only prefix of length 1."""
        df = _make_df({"C1": ["A", "B"]})
        result = generate_prefix(df)
        assert _prefix_lengths_for(result, "C1") == [1]

    def test_four_event_case_produces_three_prefixes(self):
        """4 events → prefixes of length 1, 2, 3."""
        df = _make_df({"C1": ["A", "B", "C", "D"]})
        result = generate_prefix(df)
        assert _prefix_lengths_for(result, "C1") == [1, 2, 3]

    def test_total_row_count_is_sum_of_prefix_lengths(self, two_case_df):
        """
        For a case with N events the row contribution is 1+2+...+(N-1) = N*(N-1)/2.
        C1 (N=3): 1+2 = 3 rows
        C2 (N=2): 1   = 1 row
        Total: 4 rows.
        """
        result = generate_prefix(two_case_df)
        assert len(result) == 4

    def test_each_prefix_has_correct_row_count(self, three_event_df):
        """prefix_k must contain exactly k rows."""
        result = generate_prefix(three_event_df)
        for k in [1, 2]:
            rows = _rows_for_prefix(result, "C1", k)
            assert len(rows) == k, f"prefix_{k} should have {k} rows, got {len(rows)}"


# ===========================================================================
# CASE ID RENAMING
# ===========================================================================

class TestCaseIdRenaming:

    def test_prefix_case_ids_follow_naming_convention(self, three_event_df):
        """Generated case IDs must follow the pattern <original>_prefix_<k>."""
        result = generate_prefix(three_event_df)
        expected = {"C1_prefix_1", "C1_prefix_2"}
        assert set(result["case:concept:name"].unique()) == expected

    def test_original_case_id_not_present_in_output(self, three_event_df):
        """The raw case ID 'C1' must not appear — only prefixed variants."""
        result = generate_prefix(three_event_df)
        assert "C1" not in result["case:concept:name"].values

    def test_case_ids_are_independent_across_cases(self, two_case_df):
        result = generate_prefix(two_case_df)
        ids = set(result["case:concept:name"].unique())
        assert "C1_prefix_1" in ids
        assert "C1_prefix_2" in ids
        assert "C2_prefix_1" in ids


# ===========================================================================
# CONTENT CORRECTNESS
# ===========================================================================

class TestContentCorrectness:

    def test_prefix_1_contains_only_first_event(self, three_event_df):
        result = generate_prefix(three_event_df)
        rows = _rows_for_prefix(result, "C1", 1)
        assert list(rows["concept:name"]) == ["A"]

    def test_prefix_2_contains_first_two_events(self, three_event_df):
        result = generate_prefix(three_event_df)
        rows = _rows_for_prefix(result, "C1", 2)
        assert list(rows["concept:name"]) == ["A", "B"]

    def test_prefix_events_are_in_chronological_order(self):
        """Events within each prefix must respect the original timestamp order."""
        df = _make_df({"C1": ["A", "B", "C", "D"]})
        result = generate_prefix(df)
        rows = _rows_for_prefix(result, "C1", 3)
        timestamps = list(rows["time:timestamp"])
        assert timestamps == sorted(timestamps)

    def test_prefix_does_not_contain_future_events(self, three_event_df):
        """prefix_1 must not contain event B or C — only what precedes position k."""
        result = generate_prefix(three_event_df)
        rows = _rows_for_prefix(result, "C1", 1)
        assert "B" not in rows["concept:name"].values
        assert "C" not in rows["concept:name"].values

    def test_prefix_length_column_value_matches_row_count(self, three_event_df):
        """The prefix_length value stored in each row must equal the actual row count."""
        result = generate_prefix(three_event_df)
        for case_id in result["case:concept:name"].unique():
            rows = result[result["case:concept:name"] == case_id]
            stored_length = rows["prefix_length"].iloc[0]
            assert len(rows) == stored_length

    def test_prefix_length_column_is_uniform_within_prefix(self, three_event_df):
        """Every row in a given prefix must carry the same prefix_length value."""
        result = generate_prefix(three_event_df)
        for case_id in result["case:concept:name"].unique():
            rows = result[result["case:concept:name"] == case_id]
            assert rows["prefix_length"].nunique() == 1


# ===========================================================================
# SORTING  (input order must not affect output)
# ===========================================================================

class TestSortingRobustness:

    def test_reversed_input_order_gives_same_result(self):
        """generate_prefix must sort internally; reversed input is identical output."""
        df = _make_df({"C1": ["A", "B", "C"]})
        df_reversed = df.iloc[::-1].reset_index(drop=True)
        normal   = generate_prefix(df).reset_index(drop=True)
        reversed_ = generate_prefix(df_reversed).reset_index(drop=True)
        pd.testing.assert_frame_equal(
            normal.sort_values(["case:concept:name", "time:timestamp"]).reset_index(drop=True),
            reversed_.sort_values(["case:concept:name", "time:timestamp"]).reset_index(drop=True),
        )

    def test_shuffled_multi_case_input_gives_correct_prefixes(self):
        """Cases interleaved in input rows must still produce correct per-case prefixes."""
        df = _make_df({"C1": ["A", "B", "C"], "C2": ["X", "Y", "Z"]})
        shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        result = generate_prefix(shuffled)
        # C1 prefix_1 should still be just activity A
        rows = _rows_for_prefix(result, "C1", 1)
        assert list(rows["concept:name"]) == ["A"]


# ===========================================================================
# min_prefix PARAMETER
# ===========================================================================

class TestMinPrefix:

    def test_default_min_prefix_is_one(self, three_event_df):
        """Default behaviour: shortest prefix has length 1."""
        result = generate_prefix(three_event_df)
        assert 1 in result["prefix_length"].values

    def test_min_prefix_two_excludes_length_one(self, three_event_df):
        result = generate_prefix(three_event_df, min_prefix=2)
        assert 1 not in result["prefix_length"].values
        assert 2 in result["prefix_length"].values

    def test_min_prefix_respects_upper_bound(self):
        """min_prefix=2 on a 4-event case → lengths 2 and 3 only."""
        df = _make_df({"C1": ["A", "B", "C", "D"]})
        result = generate_prefix(df, min_prefix=2)
        assert _prefix_lengths_for(result, "C1") == [2, 3]

    def test_min_prefix_equal_to_trace_length_returns_empty(self):
        """
        If min_prefix >= len(trace), range(min_prefix, len(group)) is empty
        and no prefixes are generated for that case.
        """
        df = _make_df({"C1": ["A", "B"]})
        result = generate_prefix(df, min_prefix=2)
        assert len(result) == 0


# ===========================================================================
# EDGE CASES
# ===========================================================================

class TestEdgeCases:

    def test_single_event_case_produces_no_prefixes(self):
        """
        A 1-event trace has no valid prefixes: range(1, 1) is empty.
        Full trace is excluded and there is nothing shorter.
        """
        df = _make_df({"C1": ["A"]})
        result = generate_prefix(df)
        assert len(result) == 0

    def test_empty_dataframe_returns_empty(self):
        df = pd.DataFrame(columns=["case:concept:name", "concept:name", "time:timestamp"])
        result = generate_prefix(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_empty_result_has_correct_columns(self):
        """Empty output must still carry the expected column schema."""
        df = pd.DataFrame(columns=["case:concept:name", "concept:name", "time:timestamp"])
        result = generate_prefix(df)
        for col in ["case:concept:name", "concept:name", "time:timestamp"]:
            assert col in result.columns

    def test_many_cases_all_produce_independent_prefixes(self):
        """Prefixes from different cases must not bleed into each other."""
        cases = {f"C{i}": ["A", "B", "C"] for i in range(10)}
        df = _make_df(cases)
        result = generate_prefix(df)
        for i in range(10):
            lengths = _prefix_lengths_for(result, f"C{i}")
            assert lengths == [1, 2], f"C{i} produced unexpected lengths {lengths}"

    def test_no_duplicate_prefix_case_ids(self, two_case_df):
        """Each (case_id, prefix_length) combination must be unique."""
        result = generate_prefix(two_case_df)
        assert result["case:concept:name"].nunique() == len(result["case:concept:name"].unique())