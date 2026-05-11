import pm4py
import pandas as pd
from scipy.sparse import csr_matrix


'''
Feature engineering step to encode the dataframe into a feature matrix. For the sake of the tracer bullet being created in sprint 1, this version will be implemented through a variation of one hot encoding.
WHAT THIS VERSION CAN DO:
simplify each trace into a single row
assert whether an activity is contained in the trace
assert simple metrics to each row (trace length, trace lifecycle etc.)

WHAT THIS VERSION CANT DO:
recognise duplicate activities
recognise activity ordering
recognise looping structures / causalities



INPUT: as input, a dataframe is taken with an extra 'outcome' column. A boolean input for prefix generation will be added, but the prefix generation itself not implemented for this tracer further than a 
docking spot in the code. 

OUTPUT: Sparse matrix, and a time series, as a tuple. 


'''


def oneHotEncode(df: pd.DataFrame, prefix: bool) -> tuple[csr_matrix, pd.Series]:
    
    

    if(prefix):
        #missing prefix generation
        raise NotImplementedError("prefix generation left out for sprint 1 tracer bullet")
    

