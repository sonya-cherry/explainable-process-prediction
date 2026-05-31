"""
Example rule lists for outcome() input.
Based on actual columns in uploaded BPI 2013 event log.
This list is only there for syntax support. It is not guaranteed that all these rules will create a functional classifier due to data leakage risks
"""

# -------------------------
# NUMERICAL / ORDERABLE
# -------------------------

# Every event happens after this timestamp
after_april_2010 = [
    {
        "feature": "time:timestamp",
        "operator": "always_gt",
        "value": "2010-04-01T00:00:00+02:00"
    }
]

# At least one event happens before this timestamp
has_early_event = [
    {
        "feature": "time:timestamp",
        "operator": "ever_lt",
        "value": "2010-04-01T00:00:00+02:00"
    }
]

# Every event on or after date
all_after_cutoff = [
    {
        "feature": "time:timestamp",
        "operator": "always_ge",
        "value": "2010-03-31T00:00:00+02:00"
    }
]

# If impact is numeric in your data
high_numeric_impact = [
    {
        "feature": "impact",
        "operator": "ever_gt",
        "value": 2
    }
]


# -------------------------
# ACTIVITY RULES
# -------------------------

contains_completed = [
    {
        "feature": "concept:name",
        "operator": "ever_contains",
        "value": ["Completed"]
    }
]

never_cancelled = [
    {
        "feature": "concept:name",
        "operator": "never_contains",
        "value": ["Cancelled"]
    }
]

starts_with_accepted = [
    {
        "feature": "concept:name",
        "operator": "starts_with",
        "value": "Accepted"
    }
]

ends_with_completed = [
    {
        "feature": "concept:name",
        "operator": "ends_with",
        "value": "Completed"
    }
]


# -------------------------
# IMPACT RULES
# -------------------------

always_medium_impact = [
    {
        "feature": "impact",
        "operator": "always_eq",
        "value": "Medium"
    }
]

ever_high_impact = [
    {
        "feature": "impact",
        "operator": "ever_eq",
        "value": "High"
    }
]


# -------------------------
# STRUCTURAL RULES
# -------------------------

same_resource_entire_trace = [
    {
        "feature": "org:resource",
        "operator": "all_identical"
    }
]

same_role_entire_trace = [
    {
        "feature": "org:role",
        "operator": "all_identical"
    }
]

all_activities_unique = [
    {
        "feature": "concept:name",
        "operator": "all_distinct"
    }
]


# -------------------------
# COMBINED
# -------------------------

successful_case = [
    {
        "feature": "concept:name",
        "operator": "starts_with",
        "value": "Accepted"
    },
    {
        "feature": "concept:name",
        "operator": "ever_contains",
        "value": ["Completed"]
    },
    {
        "feature": "concept:name",
        "operator": "never_contains",
        "value": ["Cancelled"]
    }
]