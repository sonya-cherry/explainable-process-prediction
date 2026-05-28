
import pandas as pd
import pm4py
import operator


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


                    Relevant operators: <,>,<=,>=,==,!=,contains, endsWith, startsWith, distinct, identical 


OUTPUT: Dataframe with outcome column. True for accepting/positive outcome, False for rejecting/negative outcome
'''                                  




def outcome(df: pd.DataFrame, outcome: list):
    df = df.copy()
    case_col = "case:concept:name"
    def always(op):
        return lambda s, v: (
            op(s, v)
            .groupby(df[case_col])
            .transform("all")
        )

    def ever(op):
        return lambda s, v: (
            op(s, v)
            .groupby(df[case_col])
            .transform("any")
        )

    ops = {
        # -------------------
        # ALWAYS (all events)
        # -------------------
        "always_gt": always(operator.gt),
        "always_lt": always(operator.lt),
        "always_ge": always(operator.ge),
        "always_le": always(operator.le),
        "always_eq": always(operator.eq),
        "always_ne": always(operator.ne),

        # -------------------
        # EVER (at least one)
        # -------------------
        "ever_gt": ever(operator.gt),
        "ever_lt": ever(operator.lt),
        "ever_ge": ever(operator.ge),
        "ever_le": ever(operator.le),
        "ever_eq": ever(operator.eq),
        "ever_ne": ever(operator.ne),

        
        # -------------------
        # CONTAINS
        # -------------------
        "always_contains": lambda s, v: (
            s.isin(v)
             .groupby(df[case_col])
             .transform("all")
        ),

        "ever_contains": lambda s, v: (
            s.isin(v)
             .groupby(df[case_col])
             .transform("any")
        ),

        "never_contains": lambda s, v: (
            ~s.isin(v)
             .groupby(df[case_col])
             .transform("all")
        ),

        # -------------------
        # TRACE POSITION
        # -------------------
        "starts_with": lambda s, v: (
            s.groupby(df[case_col])
             .transform("first")
             .eq(v)
        ),

        "ends_with": lambda s, v: (
            s.groupby(df[case_col])
             .transform("last")
             .eq(v)
        ),

        "not_starts_with": lambda s, v: (
            s.groupby(df[case_col])
             .transform("first")
             .ne(v)
        ),

        "not_ends_with": lambda s, v: (
            s.groupby(df[case_col])
             .transform("last")
             .ne(v)
        ),

        # -------------------
        # STRUCTURE
        # -------------------
        "all_identical": lambda s, v=None: (
            s.groupby(df[case_col])
             .transform("nunique")
             .eq(1)
        ),

        "all_distinct": lambda s, v=None: (
            s.groupby(df[case_col])
             .transform("size")
             .eq(
                 s.groupby(df[case_col])
                  .transform("nunique")
             )
        ),
    }
    
    df["outcome"] = True
    used_features = set()
    for rule in outcome:
        if(rule["operator"] not in ops):
            raise ValueError(f"Operator '{rule['operator']}' not recognised")
        
        if(rule["feature"] not in df.columns):
            raise ValueError(f"Feature '{rule['feature']}' not recognised")
        ##calc by ops
        condition = ops[rule["operator"]](
            df[rule["feature"]],
            rule.get("value", None)
        )
        df["outcome"] &= condition

    return df;


    





#--------------------------------------------------------------------
' REDUNTANT CODE '

#-------------------------------------------------------------------

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


