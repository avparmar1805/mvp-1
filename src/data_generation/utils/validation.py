"""Validation utilities for generated datasets."""

from typing import Any, Dict, List, Optional, Set

import pandas as pd
from loguru import logger


def validate_dataset(
    df: pd.DataFrame,
    validation_rules: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Run validation checks on a generated dataset.
    
    Args:
        df: DataFrame to validate
        validation_rules: List of validation rule dictionaries
        
    Returns:
        List of validation results
    """
    results = []
    
    for rule in validation_rules:
        rule_type = rule.get("type")
        
        if rule_type == "row_count":
            result = _validate_row_count(df, rule)
        elif rule_type == "null_percentage":
            result = _validate_null_percentage(df, rule)
        elif rule_type == "foreign_key":
            result = _validate_foreign_key(df, rule)
        elif rule_type == "unique":
            result = _validate_unique(df, rule)
        elif rule_type == "range":
            result = _validate_range(df, rule)
        elif rule_type == "not_null":
            result = _validate_not_null(df, rule)
        else:
            result = {
                "rule": f"Unknown rule type: {rule_type}",
                "passed": False,
                "error": "Unknown rule type",
            }
        
        results.append(result)
        
        # Log validation result
        status = "✓" if result["passed"] else "✗"
        logger.debug(f"  {status} {result['rule']}")
    
    return results


def _validate_row_count(df: pd.DataFrame, rule: Dict) -> Dict:
    """Validate row count is within tolerance of expected."""
    expected = rule["expected"]
    tolerance = rule.get("tolerance", 0.05)
    actual = len(df)
    
    diff_pct = abs(actual - expected) / expected if expected > 0 else 0
    passed = diff_pct <= tolerance
    
    return {
        "rule": f"Row count ~{expected:,} (±{tolerance:.0%})",
        "passed": passed,
        "expected": expected,
        "actual": actual,
        "difference_pct": diff_pct,
    }


def _validate_null_percentage(df: pd.DataFrame, rule: Dict) -> Dict:
    """Validate null percentage is within allowed limit."""
    column = rule["column"]
    max_pct = rule["max_percentage"]
    
    if column not in df.columns:
        return {
            "rule": f"{column} null % <= {max_pct:.1%}",
            "passed": False,
            "error": f"Column {column} not found",
        }
    
    null_count = df[column].isnull().sum()
    null_pct = null_count / len(df) if len(df) > 0 else 0
    passed = null_pct <= max_pct
    
    return {
        "rule": f"{column} null % <= {max_pct:.1%}",
        "passed": passed,
        "actual_pct": null_pct,
        "null_count": null_count,
    }


def _validate_foreign_key(df: pd.DataFrame, rule: Dict) -> Dict:
    """Validate foreign key references are valid."""
    fk_column = rule["fk_column"]
    ref_values = rule["ref_values"]
    max_invalid = rule.get("max_invalid", 0)
    
    if fk_column not in df.columns:
        return {
            "rule": f"{fk_column} FK integrity",
            "passed": False,
            "error": f"Column {fk_column} not found",
        }
    
    # Convert ref_values to set for faster lookup
    if isinstance(ref_values, pd.Series):
        ref_set = set(ref_values)
    else:
        ref_set = set(ref_values)
    
    invalid_mask = ~df[fk_column].isin(ref_set) & df[fk_column].notna()
    invalid_count = invalid_mask.sum()
    passed = invalid_count <= max_invalid
    
    return {
        "rule": f"{fk_column} FK integrity (max {max_invalid} invalid)",
        "passed": passed,
        "invalid_count": invalid_count,
    }


def _validate_unique(df: pd.DataFrame, rule: Dict) -> Dict:
    """Validate column values are unique."""
    column = rule["column"]
    
    if column not in df.columns:
        return {
            "rule": f"{column} uniqueness",
            "passed": False,
            "error": f"Column {column} not found",
        }
    
    total = len(df)
    unique = df[column].nunique()
    duplicate_count = total - unique
    passed = duplicate_count == 0
    
    return {
        "rule": f"{column} uniqueness",
        "passed": passed,
        "duplicate_count": duplicate_count,
    }


def _validate_range(df: pd.DataFrame, rule: Dict) -> Dict:
    """Validate numeric column is within expected range."""
    column = rule["column"]
    min_val = rule.get("min")
    max_val = rule.get("max")
    
    if column not in df.columns:
        return {
            "rule": f"{column} range [{min_val}, {max_val}]",
            "passed": False,
            "error": f"Column {column} not found",
        }
    
    violations = 0
    if min_val is not None:
        violations += (df[column] < min_val).sum()
    if max_val is not None:
        violations += (df[column] > max_val).sum()
    
    passed = violations == 0
    
    return {
        "rule": f"{column} range [{min_val}, {max_val}]",
        "passed": passed,
        "violations": violations,
    }


def _validate_not_null(df: pd.DataFrame, rule: Dict) -> Dict:
    """Validate column has no null values."""
    column = rule["column"]
    
    if column not in df.columns:
        return {
            "rule": f"{column} not null",
            "passed": False,
            "error": f"Column {column} not found",
        }
    
    null_count = df[column].isnull().sum()
    passed = null_count == 0
    
    return {
        "rule": f"{column} not null",
        "passed": passed,
        "null_count": null_count,
    }


def validate_cross_dataset(datasets: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
    """
    Validate relationships across multiple datasets.
    
    Args:
        datasets: Dictionary mapping dataset names to DataFrames
        
    Returns:
        List of cross-dataset validation results
    """
    results = []
    
    # Check orders -> customers FK
    if "orders" in datasets and "customers" in datasets:
        valid_customers = set(datasets["customers"]["customer_id"])
        invalid = ~datasets["orders"]["customer_id"].isin(valid_customers)
        results.append({
            "rule": "orders.customer_id -> customers.customer_id",
            "passed": invalid.sum() == 0,
            "invalid_count": invalid.sum(),
        })
    
    # Check orders -> products FK
    if "orders" in datasets and "products" in datasets:
        valid_products = set(datasets["products"]["product_id"])
        invalid = ~datasets["orders"]["product_id"].isin(valid_products)
        results.append({
            "rule": "orders.product_id -> products.product_id",
            "passed": invalid.sum() == 0,
            "invalid_count": invalid.sum(),
        })
    
    # Check marketing_events -> campaigns FK
    if "marketing_events" in datasets and "marketing_campaigns" in datasets:
        valid_campaigns = set(datasets["marketing_campaigns"]["campaign_id"])
        invalid = ~datasets["marketing_events"]["campaign_id"].isin(valid_campaigns)
        # Allow 1% invalid (intentional quality issue)
        max_invalid = int(len(datasets["marketing_events"]) * 0.02)
        results.append({
            "rule": "marketing_events.campaign_id -> campaigns.campaign_id",
            "passed": invalid.sum() <= max_invalid,
            "invalid_count": invalid.sum(),
            "max_allowed": max_invalid,
        })
    
    # Check support_tickets -> customers FK
    if "support_tickets" in datasets and "customers" in datasets:
        valid_customers = set(datasets["customers"]["customer_id"])
        invalid = ~datasets["support_tickets"]["customer_id"].isin(valid_customers)
        results.append({
            "rule": "support_tickets.customer_id -> customers.customer_id",
            "passed": invalid.sum() == 0,
            "invalid_count": invalid.sum(),
        })
    
    return results

