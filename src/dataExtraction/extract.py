import pm4py
import pandas as pd
#from pm4py.objects.log.util import func as functools ### Utility scripts see PM4PY tutorials #4 minute 9

'''


MAIN LINE:
1) LOAD XES FROM FOLDER
2) CONVERT TO DATAFRAME
3) SELECT STANDARDISED CASE_ID (TRACE LEVEL CONCEPT:NAME), ACTIVITY (EVENT LEVEL CONCEPT:NAME), TIMESTAMP (TIME.TIMESTAMP)
    2.5) SELECT EXTRA ATTRIBUTES TO BE CHECKED / TO BE IGNORED
4) CLEAN REST (((NO COMPLEX CLEANING YET, JUST REMOVING COLUMNS)))
5) SORT AND GROUP BY TIMESTAMP, CASE_ID
6) 


'''


       

## IMPORTS DATA IN EITHER XES OR CSV.
## INPUT: str:file_path with format ___.xes or ___.csv, list:drop_columns containing what columns to remove, column names relevant to csv files
## OUTPUT: pandas Dataframe sorted by CASEID, and secondarily TIMESTAMP
def import_data(file_path: str, str_caseID: str = None, str_activity: str = None, str_timestamp:str = None, drop_columns=None,):

    drop_columns = drop_columns or []

    if(file_path.endswith(".xes")):
        df = _import_xes(file_path)
    elif(file_path.endswith(".csv")):
        if None in (str_caseID, str_activity, str_timestamp):
            raise ValueError("Missing required columns names for csv files")
        df = _import_csv(file_path, str_caseID, str_activity, str_timestamp)
    else:
        raise ValueError("Unsupported File Type. Must be .xes or .csv")
    
    df = _postprocess(df,drop_columns)
    print(df)
    return df




def _import_xes(file_path:str):
    return pm4py.read_xes(file_path, return_legacy_log_object=False) #returns dataframe



def _import_csv(file_path:str, str_caseID: str, str_activity:str, str_timestamp:str):
    df = pd.read_csv(file_path, sep=';')
    return pm4py.format_dataframe(
        df,
        case_id=str_caseID,
        activity_key=str_activity,
        timestamp_key=str_timestamp
        )


def _postprocess(df,drop_columns):
    required = {"case:concept:name", "concept:name", "time:timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if drop_columns:
        protected = {"case:concept:name", "concept:name", "time:timestamp"}
        invalid = [c for c in drop_columns if c not in df.columns]
        if invalid:
            raise ValueError(f"Columns not found: {invalid}")

        df = df.drop(columns=[c for c in drop_columns if c not in protected])
    return df.sort_values(by=["case:concept:name", "time:timestamp"]) #added for redundancy

        
import_data("data/raw/BPI_Challenge_2013_incidents/BPI_Challenge_2013_incidents.xes", ["impact","org:role"])

    

