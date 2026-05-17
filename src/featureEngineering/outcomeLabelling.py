
import pandas as pd
import pm4py


'''
FIRST IDEA: take as input a List of Dictionaries, that defines outcome relevant values for each column/attribute. User Layer code could then decide how to define outcomes,
            act as a safety layer etc. NOT OPTIMIZED, but maybe good? 

INPUT:1)Dataframe from extract.py
      2)List of Dicts. Format:
                        rules = [
                            { 
                                "feature": "numericalFeatureName",
                                "operator": "< or > or <= or >= or ==",
                                "value": Value
                            }
                            {
                                "feature": "categoricalFeatureName"
                                "operator": "contains"
                                "value": "featureValue"

                            }
                        ]
'''                                  
def outcome(df: pd.DataFrame, outcome: list, prefix:bool):
    pass




# SPRINT 1 IMPLEMENTATION: duration-based binary outcome labeling.
def compute_duration_outcomes(
    df: pd.DataFrame,
    case_col: str = "case:concept:name",
    timestamp_col: str = "time:timestamp",
) -> pd.DataFrame:
    """
    Compute binary case outcomes based on case duration.

    Positive outcome: case duration is less than or equal to the median case duration.
    Negative outcome: case duration is greater than the median case duration.

    Returns a case-level DataFrame with one row per case and columns:
    case_start, case_end, duration, outcome.
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    case_df = (
        df.groupby(case_col)[timestamp_col]
        .agg(
            case_start="min",
            case_end="max",
        )
        .reset_index()
    )

    case_df["duration"] = case_df["case_end"] - case_df["case_start"]
    outcome_threshold = case_df["duration"].median()
    case_df["outcome"] = (case_df["duration"] <= outcome_threshold).astype(int)
    
    return case_df

'''
Temporary outcome labeller for compute_duration_outcomes()
input: the original df, and the df from compute_duration_outcomes(), to add a label to each  
'''
def tempFormat(df: pd.DataFrame, dur: pd.DataFrame):
    df = df.merge(dur[["case:concept:name","outcome"]], on="case:concept:name", how="left")
    return df


