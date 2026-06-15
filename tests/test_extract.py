"""
tests/test_extract.py

Run all:          pytest test_extract.py -v
Skip integration: pytest test_extract.py -v -m "not integration"
"""

from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd
import pytest

from src.data_extraction import extract
from src.data_extraction.extract import (
    _import_csv,
    _import_xes,
    _postprocess,
    import_data,
    split,
)

# ===========================================================================
# CONSTANTS & SHARED HELPERS
# ===========================================================================

REQUIRED_COLS = {"case:concept:name", "concept:name", "time:timestamp"}


def _ts(year=2024, month=1, day=1, hour=0):
    """Timezone-aware datetime, keeps fixtures concise."""
    return datetime(year, month, day, hour, tzinfo=timezone.utc)


def _write_csv(path, rows, sep=";"):
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, sep=sep)
    return str(path)


# ===========================================================================
# SHARED FIXTURES
# ===========================================================================

@pytest.fixture()
def minimal_df():
    """Smallest valid DataFrame: 2 cases, 3 events."""
    return pd.DataFrame({
        "case:concept:name": ["C1", "C1", "C2"],
        "concept:name":      ["A",  "B",  "A"],
        "time:timestamp": [_ts(2024, 1, 1), _ts(2024, 1, 2), _ts(2024, 1, 3)],
    })


@pytest.fixture()
def four_case_df():
    """
    Four cases (A–D) with clean monthly separation.
    Designed so exact partition membership can be asserted.
    Shuffled so row order != chronological order.
    """
    rows = [
        # case A  — January
        {"case:concept:name": "A", "concept:name": "start", "time:timestamp": _ts(2024, 1, 1)},
        {"case:concept:name": "A", "concept:name": "end",   "time:timestamp": _ts(2024, 1, 2)},
        # case B  — February
        {"case:concept:name": "B", "concept:name": "start", "time:timestamp": _ts(2024, 2, 1)},
        {"case:concept:name": "B", "concept:name": "end",   "time:timestamp": _ts(2024, 2, 2)},
        # case C  — March
        {"case:concept:name": "C", "concept:name": "start", "time:timestamp": _ts(2024, 3, 1)},
        {"case:concept:name": "C", "concept:name": "end",   "time:timestamp": _ts(2024, 3, 2)},
        # case D  — April
        {"case:concept:name": "D", "concept:name": "start", "time:timestamp": _ts(2024, 4, 1)},
        {"case:concept:name": "D", "concept:name": "end",   "time:timestamp": _ts(2024, 4, 2)},
    ]
    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


@pytest.fixture()
def hundred_case_df():
    """100 single-event cases for ratio arithmetic tests."""
    return pd.DataFrame({
        "case:concept:name": [f"C{i:03d}" for i in range(100)],
        "concept:name":      ["A"] * 100,
        "time:timestamp":    [_ts(2024, 1, 1, hour=i % 24) for i in range(100)],
    })


# ===========================================================================
# import_data  —  routing & validation
# ===========================================================================

class TestImportData:

    def test_unsupported_extension_raises(self, tmp_path):
        p = tmp_path / "data.txt"
        p.write_text("x")
        with pytest.raises(ValueError, match="Unsupported"):
            import_data(str(p))

    def test_no_extension_raises(self, tmp_path):
        p = tmp_path / "datafile"
        p.write_text("x")
        with pytest.raises(ValueError, match="Unsupported"):
            import_data(str(p))

    def test_xes_delegates_to_import_xes(self, tmp_path, minimal_df):
        p = tmp_path / "log.xes"
        p.write_text("<dummy/>")
        with patch.object(extract, "_import_xes", return_value=minimal_df) as mock_xes, \
             patch.object(extract, "_postprocess", return_value=minimal_df):
            import_data(str(p))
        mock_xes.assert_called_once_with(str(p))

    def test_csv_delegates_to_import_csv(self, tmp_path, minimal_df):
        p = tmp_path / "log.csv"
        p.write_text("x")
        with patch.object(extract, "_import_csv", return_value=minimal_df) as mock_csv, \
             patch.object(extract, "_postprocess", return_value=minimal_df):
            import_data(str(p), str_caseID="cid", str_activity="act", str_timestamp="ts")
        mock_csv.assert_called_once_with(str(p), "cid", "act", "ts")

    def test_csv_missing_all_column_args_raises(self, tmp_path):
        p = tmp_path / "log.csv"
        p.write_text("x")
        with pytest.raises(ValueError, match="Missing required"):
            import_data(str(p))

    @pytest.mark.parametrize("kwargs", [
        dict(str_activity="act", str_timestamp="ts"),   # missing caseID
        dict(str_caseID="cid",   str_timestamp="ts"),   # missing activity
        dict(str_caseID="cid",   str_activity="act"),   # missing timestamp
    ])
    def test_csv_missing_single_column_arg_raises(self, tmp_path, kwargs):
        p = tmp_path / "log.csv"
        p.write_text("x")
        with pytest.raises(ValueError, match="Missing required"):
            import_data(str(p), **kwargs)

    def test_drop_columns_forwarded_to_postprocess(self, tmp_path, minimal_df):
        p = tmp_path / "log.xes"
        p.write_text("<dummy/>")
        with patch.object(extract, "_import_xes", return_value=minimal_df), \
             patch.object(extract, "_postprocess", return_value=minimal_df) as mock_pp:
            import_data(str(p), drop_columns=["extra"])
        assert mock_pp.call_args[0][1] == ["extra"]

    def test_returns_dataframe(self, tmp_path, minimal_df):
        p = tmp_path / "log.xes"
        p.write_text("<dummy/>")
        with patch.object(extract, "_import_xes", return_value=minimal_df), \
             patch.object(extract, "_postprocess", return_value=minimal_df):
            result = import_data(str(p))
        assert isinstance(result, pd.DataFrame)


# ===========================================================================
# _import_csv
# ===========================================================================

class TestImportCsv:

    _ROWS = [
        {"case_id": "C1", "activity": "A", "timestamp": "2024-01-01 00:00:00"},
        {"case_id": "C1", "activity": "B", "timestamp": "2024-01-02 00:00:00"},
        {"case_id": "C2", "activity": "A", "timestamp": "2024-01-03 00:00:00"},
    ]

    def test_reads_semicolon_delimited(self, tmp_path):
        path = _write_csv(tmp_path / "log.csv", self._ROWS)
        df = _import_csv(path, "case_id", "activity", "timestamp")
        assert len(df) == 3

    def test_renames_columns_to_pm4py_standard(self, tmp_path):
        path = _write_csv(tmp_path / "log.csv", self._ROWS)
        df = _import_csv(path, "case_id", "activity", "timestamp")
        assert REQUIRED_COLS.issubset(set(df.columns))

    def test_timestamp_column_is_datetime(self, tmp_path):
        path = _write_csv(tmp_path / "log.csv", self._ROWS)
        df = _import_csv(path, "case_id", "activity", "timestamp")
        assert pd.api.types.is_datetime64_any_dtype(df["time:timestamp"])

    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(Exception):
            _import_csv(str(tmp_path / "missing.csv"), "cid", "act", "ts")


# ===========================================================================
# _import_xes
# ===========================================================================

class TestImportXes:

    def test_returns_dataframe(self, minimal_df):
        with patch("pm4py.read_xes", return_value=minimal_df):
            result = _import_xes("dummy.xes")
        assert isinstance(result, pd.DataFrame)

    def test_called_with_legacy_false(self, minimal_df):
        with patch("pm4py.read_xes", return_value=minimal_df) as mock_read:
            _import_xes("dummy.xes")
        mock_read.assert_called_once_with("dummy.xes", return_legacy_log_object=False)

    def test_propagates_file_not_found(self):
        with patch("pm4py.read_xes", side_effect=FileNotFoundError("not found")):
            with pytest.raises(FileNotFoundError):
                _import_xes("ghost.xes")


# ===========================================================================
# _postprocess
# ===========================================================================

class TestPostprocess:

    # --- happy path ---

    def test_valid_df_with_extra_column_returned(self):
        df = pd.DataFrame({
            "case:concept:name": ["A"],
            "concept:name":      ["Start"],
            "time:timestamp":    [pd.Timestamp("2024-01-01")],
            "resource":          ["R1"],
        })
        result = _postprocess(df, [])
        assert len(result) == 1
        assert set(result.columns) == set(df.columns)

    # --- missing required columns ---

    @pytest.mark.parametrize("drop_col", [
        "case:concept:name",
        "concept:name",
        "time:timestamp",
    ])
    def test_missing_required_column_raises(self, minimal_df, drop_col):
        df = minimal_df.drop(columns=[drop_col])
        with pytest.raises(ValueError, match="Missing required columns"):
            _postprocess(df, [])

    def test_missing_multiple_required_columns_all_named_in_error(self):
        df = pd.DataFrame({"irrelevant": [1]})
        with pytest.raises(ValueError) as exc_info:
            _postprocess(df, [])
        msg = str(exc_info.value)
        assert "case:concept:name" in msg
        assert "concept:name" in msg
        assert "time:timestamp" in msg

    # --- drop_columns ---

    def test_drops_requested_column(self, minimal_df):
        df = minimal_df.copy()
        df["resource"] = "R1"
        result = _postprocess(df, ["resource"])
        assert "resource" not in result.columns

    def test_drops_multiple_columns_leaves_others(self, minimal_df):
        df = minimal_df.copy()
        df["col_a"] = 1
        df["col_b"] = 2
        df["cost"]  = 10
        result = _postprocess(df, ["col_a", "col_b"])
        assert "col_a" not in result.columns
        assert "col_b" not in result.columns
        assert "cost" in result.columns         # untargeted column survives

    def test_unknown_drop_column_raises(self, minimal_df):
        with pytest.raises(ValueError, match="Columns not found"):
            _postprocess(minimal_df.copy(), ["does_not_exist"])

    def test_empty_drop_list_leaves_all_columns(self, minimal_df):
        df = minimal_df.copy()
        df["extra"] = 5
        result = _postprocess(df, [])
        assert "extra" in result.columns

    @pytest.mark.parametrize("protected", [
        "case:concept:name",
        "concept:name",
        "time:timestamp",
    ])
    def test_protected_columns_not_dropped(self, minimal_df, protected):
        """Protected columns in drop_columns are silently skipped."""
        result = _postprocess(minimal_df.copy(), [protected])
        assert protected in result.columns

    def test_original_df_not_mutated(self, minimal_df):
        df = minimal_df.copy()
        df["extra"] = 7
        original_cols = list(df.columns)
        _postprocess(df, ["extra"])
        assert list(df.columns) == original_cols


# ===========================================================================
# split
# ===========================================================================

class TestSplit:

    # --- ratio validation ---

    def test_ratios_not_summing_to_one_raises(self, four_case_df):
        with pytest.raises(ValueError, match="Ratios must sum to 1"):
            split(four_case_df, 0.5, 0.3, 0.3)

    def test_floating_point_near_one_accepted(self, four_case_df):
        # e.g. 0.1 + 0.1 + 0.8 has fp representation error; np.isclose should pass it
        train, val, test = split(four_case_df, 0.1, 0.1, 0.8)
        assert isinstance(train, pd.DataFrame)

    # --- return type ---

    def test_returns_three_dataframes(self, four_case_df):
        result = split(four_case_df)
        assert len(result) == 3
        assert all(isinstance(r, pd.DataFrame) for r in result)

    # --- temporal ordering (exact membership) ---

    def test_temporal_ordering_exact_membership(self, four_case_df):
        """
        4 cases (A=Jan, B=Feb, C=Mar, D=Apr), split 50/25/25.
        Verifies chronological assignment precisely, not just min/max.
        """
        train, val, test = split(four_case_df, 0.5, 0.25, 0.25)
        assert set(train["case:concept:name"].unique()) == {"A", "B"}
        assert set(val["case:concept:name"].unique())   == {"C"}
        assert set(test["case:concept:name"].unique())  == {"D"}

    # --- partition integrity ---

    def test_no_case_overlap_across_splits(self, four_case_df):
        train, val, test = split(four_case_df, 0.5, 0.25, 0.25)
        train_cases = set(train["case:concept:name"].unique())
        val_cases   = set(val["case:concept:name"].unique())
        test_cases  = set(test["case:concept:name"].unique())
        assert train_cases.isdisjoint(val_cases)
        assert train_cases.isdisjoint(test_cases)
        assert val_cases.isdisjoint(test_cases)

    def test_all_cases_covered(self, four_case_df):
        train, val, test = split(four_case_df, 0.5, 0.25, 0.25)
        original = set(four_case_df["case:concept:name"].unique())
        recovered = (
            set(train["case:concept:name"].unique())
            | set(val["case:concept:name"].unique())
            | set(test["case:concept:name"].unique())
        )
        assert recovered == original

    def test_each_case_appears_in_exactly_one_split(self, four_case_df):
        partitions = split(four_case_df, 0.5, 0.25, 0.25)
        for case_id in four_case_df["case:concept:name"].unique():
            appearances = sum(
                case_id in set(p["case:concept:name"].unique())
                for p in partitions
            )
            assert appearances == 1

    def test_row_counts_sum_to_total(self, four_case_df):
        train, val, test = split(four_case_df, 0.5, 0.25, 0.25)
        assert len(train) + len(val) + len(test) == len(four_case_df)

    def test_all_events_of_a_case_stay_together(self, four_case_df):
        """No case should be split across partitions — all its rows land together."""
        train, val, test = split(four_case_df, 0.5, 0.25, 0.25)
        for part in (train, val, test):
            for case_id in part["case:concept:name"].unique():
                expected = (four_case_df["case:concept:name"] == case_id).sum()
                actual   = (part["case:concept:name"] == case_id).sum()
                assert actual == expected

    # --- ratio arithmetic ---

    def test_default_ratios_70_15_15(self, hundred_case_df):
        train, val, test = split(hundred_case_df)
        assert len(train) == 70
        assert len(val)   == 15
        assert len(test)  == 15

    def test_custom_ratios_80_10_10(self, hundred_case_df):
        train, val, test = split(hundred_case_df, 0.8, 0.1, 0.1)
        assert len(train) == 80
        assert len(val)   == 10
        assert len(test)  == 10

    # --- edge cases ---

    def test_single_case_rows_all_in_one_split(self, minimal_df):
        single = minimal_df[minimal_df["case:concept:name"] == "C1"].copy()
        train, val, test = split(single)
        assert len(train) + len(val) + len(test) == len(single)

    def test_zero_val_ratio_gives_empty_val(self, four_case_df):
        _, val, _ = split(four_case_df, 0.75, 0.0, 0.25)
        assert len(val) == 0

    def test_zero_test_ratio_gives_empty_test(self, four_case_df):
        _, _, test = split(four_case_df, 0.75, 0.25, 0.0)
        assert len(test) == 0

    # --- immutability ---

    def test_original_df_not_mutated(self, four_case_df):
        original_len  = len(four_case_df)
        original_cols = list(four_case_df.columns)
        split(four_case_df, 0.5, 0.25, 0.25)
        assert len(four_case_df)      == original_len
        assert list(four_case_df.columns) == original_cols

    def test_splits_are_independent_copies(self, four_case_df):
        train, val, _ = split(four_case_df, 0.5, 0.25, 0.25)
        train["_sentinel"] = 99
        assert "_sentinel" not in val.columns


# ===========================================================================
# Integration
# ===========================================================================

@pytest.mark.integration
class TestIntegration:

    def _write_event_log(self, path):
        rows = [
            {"cid": "C1", "act": "Start", "ts": "2024-01-01 08:00:00"},
            {"cid": "C1", "act": "End",   "ts": "2024-01-01 09:00:00"},
            {"cid": "C2", "act": "Start", "ts": "2024-01-02 08:00:00"},
            {"cid": "C2", "act": "End",   "ts": "2024-01-02 09:00:00"},
            {"cid": "C3", "act": "Start", "ts": "2024-01-03 08:00:00"},
            {"cid": "C3", "act": "End",   "ts": "2024-01-03 09:00:00"},
        ]
        pd.DataFrame(rows).to_csv(path, index=False, sep=";")

    def test_csv_import_then_split_runs_without_error(self, tmp_path):
        p = str(tmp_path / "log.csv")
        self._write_event_log(p)
        df = import_data(p, str_caseID="cid", str_activity="act", str_timestamp="ts")
        train, val, test = split(df)
        assert all(isinstance(r, pd.DataFrame) for r in (train, val, test))

    def test_required_columns_present_in_every_split(self, tmp_path):
        p = str(tmp_path / "log.csv")
        self._write_event_log(p)
        df = import_data(p, str_caseID="cid", str_activity="act", str_timestamp="ts")
        for part in split(df):
            assert REQUIRED_COLS.issubset(set(part.columns))