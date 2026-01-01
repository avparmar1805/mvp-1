import pandas as pd
from typing import Dict, Any, List

def classify_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Classify columns into Temporal, Numeric, Categorical, and Text.
    """
    classification = {
        "temporal": [],
        "numeric": [],
        "categorical": [],
        "text": []
    }
    
    for col in df.columns:
        # Check for numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            classification["numeric"].append(col)
        # Check for datetime
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            classification["temporal"].append(col)
        else:
            # Check cardinality for categorical vs text
            # Heuristic: < 20 unique values is categorical
            if df[col].nunique() < 20:
                classification["categorical"].append(col)
            else:
                classification["text"].append(col)
                
    return classification

def recommend_chart(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Recommend the best chart type based on the dataframe structure.
    Returns a config dictionary.
    """
    cols = classify_columns(df)
    
    config = {
        "type": "table",
        "primary_metric": None,
        "dimensions": [],
        "title": "Data Preview"
    }

    # Helper: Find first metric
    metric = cols["numeric"][0] if cols["numeric"] else None
    
    if not metric:
        return config

    # Rule 1: Time Series (Line Chart)
    # Condition: 1+ Temporal + 1+ Numeric
    if cols["temporal"]:
        config["type"] = "line"
        config["primary_metric"] = metric
        config["dimensions"] = [cols["temporal"][0]]
        config["title"] = f"{metric} over Time"
        return config

    # Rule 2: Category Comparison (Bar Chart)
    # Condition: 1+ Categorical + 1+ Numeric
    if cols["categorical"]:
        config["type"] = "bar"
        config["primary_metric"] = metric
        config["dimensions"] = [cols["categorical"][0]]
        config["title"] = f"{metric} by {cols['categorical'][0]}"
        return config

    # Rule 3: Correlation (Scatter Plot)
    # Condition: 2+ Numeric and NO Temporal
    if len(cols["numeric"]) >= 2 and not cols["temporal"]:
        config["type"] = "scatter"
        config["primary_metric"] = cols["numeric"][1] # Y axis
        config["dimensions"] = [cols["numeric"][0]]   # X axis
        config["title"] = f"{cols['numeric'][1]} vs {cols['numeric'][0]}"
        return config
        
    return config
