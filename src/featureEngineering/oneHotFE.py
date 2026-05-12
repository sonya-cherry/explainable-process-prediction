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


def Encode(df: pd.DataFrame, prefix: bool) -> tuple[csr_matrix, pd.Series, pd.Series]:
    


    
    #Validate for id, activity, outcome
    required = {"case:concept:name", "concept:name", "outcome"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}") 


    #OH encode each categorical column
    featureMatrix = pd.DataFrame(index=df["case:concept:name"].unique())
    for col in df.columns:
        if col in {"case:concept:name", "outcome", "time:timestamp"}:
            continue
        if (pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col])):
            temp = pd.crosstab(df["case:concept:name"],df[col]).clip(upper=1)
            temp = temp.add_prefix(f"{col}=")

        elif pd.api.types.is_numeric_dtype(df[col]):
            temp = df.groupby("case:concept:name")[col].agg(["sum", "mean", "min", "max", "std","count"])
            temp = temp.add_prefix(f"{col}_")
        
        featureMatrix = featureMatrix.join(temp)
    
    y = df.groupby("case:concept:name")["outcome"].first() ##Outcome as time series
    case_ids = featureMatrix.index
    X_sparse = csr_matrix(featureMatrix.fillna(0).values)

    return X_sparse,y,case_ids

    
    
    

