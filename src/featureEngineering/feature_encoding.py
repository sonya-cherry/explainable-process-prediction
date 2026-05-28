import pm4py
import pandas as pd
from scipy.sparse import csr_matrix


'''
Feature engineering step to encode the dataframe into a feature matrix. For the sake of the tracer bullet being created in sprint 1, this version will be implemented through a variation of one hot encoding.
WHAT THIS VERSION CAN DO:
simplify each trace into a single row
assert whether any categorical trait is contained in the trace
assert for each numerical trait certain values (sum, mean, min, max, std, count)

WHAT THIS VERSION CANT DO:
recognise duplicate activities
recognise activity ordering
recognise looping structures / causalities
temporal features left out for the tracer bullet, since the outcome definition is wholly based on time.


INPUT: as input, a dataframe is taken with an extra 'outcome' column. A boolean input for prefix generation will be added, but the prefix generation itself not implemented for this tracer further than a 
docking spot in the code. 

OUTPUT: Sparse matrix, outcome label series, and caseId series, as a tuple. 


'''




"""
VERSION 2:
SHOULD SOLVE:
1) ACTIVITY DIRECTLY FOLLOWS ORDERING
2) ACTIVITY DUPLICATES
3) RECOGNISE LOOPS
4) TEMPORAL FEATURES
5) PREVENTS CATEGORICAL EXPLOSION BY CHECKING NUMBER OF DISTINCT VALUES
"""
def Encode(df: pd.DataFrame) -> tuple[csr_matrix, pd.Series, pd.Series]:
    df = df.copy()
    required = {"case:concept:name", "outcome"}
    missing = required - set(df.columns)
    transition_columns = {"concept:name","lifecycle:transition","org:resource"}


    '''SAFETY CHECK'''

    if missing: 
        raise ValueError(f"Missing required columns: {missing}")
    featureMatrix = pd.DataFrame(index=df["case:concept:name"].unique())


    ''' TIME BASED FEATURES'''


    if("time:timestamp" in df.columns):
        df["elapsed_time"] = (df["time:timestamp"]- df.groupby("case:concept:name")["time:timestamp"].transform("first")).dt.total_seconds()
        df["time_since_last"] = (df.groupby("case:concept:name")["time:timestamp"].diff().dt.total_seconds().fillna(0))
        df["hour"] = df["time:timestamp"].dt.hour
        df["day_of_week"] = df["time:timestamp"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)
        df["month"] = df["time:timestamp"].dt.month
        typical_duration = (df.groupby("case:concept:name")["elapsed_time"].max().median())
        df["relative_age"] = (df["elapsed_time"] / typical_duration)

    '''NUMERICAL AND CATEGORICAL FEATURE ENCODING'''
    df["first_activity"] = (df.groupby("case:concept:name")["concept:name"].transform("first"))

    df["last_activity"] = (df.groupby("case:concept:name")["concept:name"].transform("last"))
    
    original_columns = list(df.columns)
    for col in original_columns:
        if col in {"case:concept:name", "outcome", "time:timestamp"}:
            continue

        if ((pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col])) and df[col].nunique() < 100):
            # -------------------------
            # STANDARD COUNT ENCODING
            # -------------------------
            temp = pd.crosstab(df["case:concept:name"], df[col]).add_prefix(f"{col}=")
            # -------------------------
            # TRANSITION ENCODING
            # ------------------------
            if col in transition_columns:
                next_values = (df.groupby("case:concept:name")[col].shift(-1))
                valid = next_values.notna()
                transitions = (df[col].astype(str) + "->" + next_values.astype(str))
                transition_temp = pd.crosstab(df.loc[valid, "case:concept:name"],transitions[valid]).add_prefix(f"{col}_transition=")
                temp = temp.join(transition_temp)
        elif pd.api.types.is_numeric_dtype(df[col]):
            temp = df.groupby("case:concept:name")[col].agg(["sum", "mean", "min", "max", "std","count"])
            temp = temp.add_prefix(f"{col}_")
        else:
            continue
        featureMatrix = featureMatrix.join(temp)
        
    
    y = df.groupby("case:concept:name")["outcome"].first() ##outcome as series
    case_ids = featureMatrix.index ##caseIds as series
    X_sparse = csr_matrix(featureMatrix.fillna(0).values) #feature matrix as sparse matrix

    return X_sparse,y,case_ids



    
    
    

