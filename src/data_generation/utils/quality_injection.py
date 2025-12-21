"""Utilities for injecting intentional data quality issues."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


def inject_quality_issues(
    df: pd.DataFrame,
    config: Dict[str, Dict[str, Any]],
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Inject intentional data quality issues into a DataFrame.
    
    Args:
        df: DataFrame to modify
        config: Dictionary mapping column names to issue configurations
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with injected quality issues
        
    Example config:
        {
            'discount_amount': {'type': 'null', 'percentage': 0.02},
            'order_id': {'type': 'duplicate', 'percentage': 0.01},
            'total_amount': {'type': 'invalid', 'percentage': 0.005, 'invalid_value': -100},
            'email': {'type': 'invalid_format', 'percentage': 0.02, 'invalid_value': 'invalid-email'}
        }
    """
    if seed is not None:
        np.random.seed(seed)
    
    df_copy = df.copy()
    issues_injected = []
    
    for column, issue_config in config.items():
        issue_type = issue_config.get("type")
        
        if column not in df_copy.columns and issue_type != "duplicate":
            logger.warning(f"Column {column} not found, skipping quality injection")
            continue
        
        if issue_type == "null":
            df_copy, count = _inject_nulls(df_copy, column, issue_config)
            issues_injected.append(f"{count} nulls in {column}")
            
        elif issue_type == "duplicate":
            df_copy, count = _inject_duplicates(df_copy, column, issue_config)
            issues_injected.append(f"{count} duplicate rows")
            
        elif issue_type == "invalid":
            df_copy, count = _inject_invalid_values(df_copy, column, issue_config)
            issues_injected.append(f"{count} invalid values in {column}")
            
        elif issue_type == "invalid_format":
            df_copy, count = _inject_invalid_format(df_copy, column, issue_config)
            issues_injected.append(f"{count} invalid format in {column}")
            
        elif issue_type == "mismatch":
            df_copy, count = _inject_mismatch(df_copy, column, issue_config)
            issues_injected.append(f"{count} mismatched values in {column}")
    
    if issues_injected:
        logger.debug(f"Injected quality issues: {', '.join(issues_injected)}")
    
    return df_copy


def _inject_nulls(
    df: pd.DataFrame,
    column: str,
    config: Dict,
) -> tuple[pd.DataFrame, int]:
    """Inject null values into a column."""
    percentage = config.get("percentage", 0.01)
    count = int(len(df) * percentage)
    
    if count == 0:
        return df, 0
    
    indices = np.random.choice(df.index, size=count, replace=False)
    df.loc[indices, column] = None
    
    return df, count


def _inject_duplicates(
    df: pd.DataFrame,
    column: str,
    config: Dict,
) -> tuple[pd.DataFrame, int]:
    """Inject duplicate rows into the DataFrame."""
    percentage = config.get("percentage", 0.01)
    count = int(len(df) * percentage)
    
    if count == 0:
        return df, 0
    
    # Select random rows to duplicate
    dup_indices = np.random.choice(df.index, size=count, replace=True)
    duplicates = df.loc[dup_indices].copy()
    
    # Concatenate duplicates
    df = pd.concat([df, duplicates], ignore_index=True)
    
    return df, count


def _inject_invalid_values(
    df: pd.DataFrame,
    column: str,
    config: Dict,
) -> tuple[pd.DataFrame, int]:
    """Inject invalid values into a column."""
    percentage = config.get("percentage", 0.01)
    invalid_value = config.get("invalid_value")
    count = int(len(df) * percentage)
    
    if count == 0 or invalid_value is None:
        return df, 0
    
    indices = np.random.choice(df.index, size=count, replace=False)
    df.loc[indices, column] = invalid_value
    
    return df, count


def _inject_invalid_format(
    df: pd.DataFrame,
    column: str,
    config: Dict,
) -> tuple[pd.DataFrame, int]:
    """Inject values with invalid format (e.g., invalid emails)."""
    percentage = config.get("percentage", 0.01)
    invalid_value = config.get("invalid_value", "INVALID")
    count = int(len(df) * percentage)
    
    if count == 0:
        return df, 0
    
    indices = np.random.choice(df.index, size=count, replace=False)
    df.loc[indices, column] = invalid_value
    
    return df, count


def _inject_mismatch(
    df: pd.DataFrame,
    column: str,
    config: Dict,
) -> tuple[pd.DataFrame, int]:
    """Inject mismatched values (e.g., category doesn't match product)."""
    percentage = config.get("percentage", 0.01)
    mismatch_value = config.get("mismatch_value", "MISMATCH")
    count = int(len(df) * percentage)
    
    if count == 0:
        return df, 0
    
    indices = np.random.choice(df.index, size=count, replace=False)
    df.loc[indices, column] = mismatch_value
    
    return df, count


def get_quality_issue_summary(
    original_df: pd.DataFrame,
    modified_df: pd.DataFrame,
    config: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Generate a summary of injected quality issues.
    
    Args:
        original_df: Original DataFrame before injection
        modified_df: Modified DataFrame after injection
        config: Quality issue configuration used
        
    Returns:
        List of quality issue descriptions
    """
    issues = []
    
    for column, issue_config in config.items():
        issue_type = issue_config.get("type")
        percentage = issue_config.get("percentage", 0)
        
        if issue_type == "null":
            if column in modified_df.columns:
                actual_nulls = modified_df[column].isnull().sum()
                issues.append({
                    "column": column,
                    "type": "null_values",
                    "expected_percentage": percentage,
                    "actual_count": actual_nulls,
                    "description": f"{percentage:.1%} null values in {column}",
                })
        
        elif issue_type == "duplicate":
            row_diff = len(modified_df) - len(original_df)
            issues.append({
                "column": column,
                "type": "duplicate_rows",
                "expected_percentage": percentage,
                "actual_count": row_diff,
                "description": f"{percentage:.1%} duplicate rows added",
            })
        
        elif issue_type in ("invalid", "invalid_format", "mismatch"):
            issues.append({
                "column": column,
                "type": issue_type,
                "expected_percentage": percentage,
                "description": f"{percentage:.1%} {issue_type} values in {column}",
            })
    
    return issues

