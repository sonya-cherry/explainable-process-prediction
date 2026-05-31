import pandas as pd



"""
GENERATES PREFIXES, AUTOMATICALLY REMOVE FULL LENGTH VERSION TO AVOID GENERAL LEAKAGE

"""
def generate_prefix(df: pd.DataFrame, min_prefix: int = 1) -> pd.DataFrame:

    case_col = "case:concept:name"
    time_col = "time:timestamp"

    df = df.sort_values([case_col, time_col])

    prefixes = []

    for case_id, group in df.groupby(case_col):

        group = group.reset_index(drop=True)

        # avoid full trace leakage
        for k in range(min_prefix, len(group)):

            prefix = group.iloc[:k].copy()

            prefix[case_col] = f"{case_id}_prefix_{k}"
            prefix["prefix_length"] = k

            prefixes.append(prefix)

    return pd.concat(prefixes, ignore_index=True)