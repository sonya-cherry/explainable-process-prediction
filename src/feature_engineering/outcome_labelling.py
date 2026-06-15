
import pandas as pd
import pm4py
import operator


'''
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


                    Relevant operators: <,>,<=,>=,==,!=,contains, endsWith, startsWith


OUTPUT: Dataframe with outcome column. True for accepting/positive outcome, False for rejecting/negative outcome

WARNING: ANY OUTCOME LABELLING ACCORDING TO GENERATED RULES IS ALLOWED AND WILL NOT LEAD TO ERRORS. HOWEVER, TARGET FEATURE LEAKAGE CANNOT BE UNIFORMLY PREVENTED WITHOUT KNOWLEDGE OF DATASET RELATIONS AND OUTCOME LABELS. 
BE ADVISED THAT CERTAIN RULES CAN LEAD TO LEAKAGE BASED OVERFITTING AND ERRONEOUS RESULTS. DO NOT TAKE THE ABILITY TO CREATE ANY OUTCOME CLASSIFIER AS THE ABILITY TO CLASSIFY ANY OUTCOME WELL.
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


    




