
import pandas
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
def outcome(df: pandas.DataFrame, outcome: list, prefix:bool):
    pass



