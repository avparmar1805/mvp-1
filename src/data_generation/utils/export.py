"""Export utilities for saving DataFrames to Parquet with metadata."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from loguru import logger


def export_to_parquet(
    df: pd.DataFrame,
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Export DataFrame to Parquet format with metadata JSON.
    
    Args:
        df: DataFrame to export
        output_path: Path for the Parquet file
        metadata: Optional metadata dictionary with 'name' key
        
    Returns:
        Dictionary with export statistics
    """
    output_path = Path(output_path)
    
    # Create parent directories if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write Parquet file
    df.to_parquet(
        output_path,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )
    
    file_size = output_path.stat().st_size
    
    # Generate metadata
    metadata_dict = _generate_metadata(df, output_path, metadata)
    
    # Write metadata JSON
    metadata_path = output_path.with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump(metadata_dict, f, indent=2, default=str)
    
    logger.info(
        f"Exported {len(df):,} rows to {output_path.name} "
        f"({file_size / 1024 / 1024:.2f} MB)"
    )
    
    return {
        "path": str(output_path),
        "row_count": len(df),
        "size_bytes": file_size,
        "metadata_path": str(metadata_path),
    }


def _generate_metadata(
    df: pd.DataFrame,
    output_path: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate metadata dictionary for the dataset."""
    metadata = metadata or {}
    
    # Build schema information
    schema = []
    for col in df.columns:
        col_info = {
            "name": col,
            "type": str(df[col].dtype),
            "nullable": bool(df[col].isnull().any()),
            "null_count": int(df[col].isnull().sum()),
            "unique_count": int(df[col].nunique()),
        }
        
        # Add statistics for numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info["statistics"] = {
                "min": float(df[col].min()) if not df[col].isnull().all() else None,
                "max": float(df[col].max()) if not df[col].isnull().all() else None,
                "mean": float(df[col].mean()) if not df[col].isnull().all() else None,
                "std": float(df[col].std()) if not df[col].isnull().all() else None,
            }
        
        # Add value counts for categorical columns with few unique values
        if df[col].nunique() <= 20 and df[col].dtype == "object":
            value_counts = df[col].value_counts().head(10).to_dict()
            col_info["top_values"] = {str(k): int(v) for k, v in value_counts.items()}
        
        schema.append(col_info)
    
    return {
        "dataset_name": metadata.get("name", output_path.stem),
        "description": metadata.get("description", ""),
        "created_at": pd.Timestamp.now().isoformat(),
        "row_count": len(df),
        "column_count": len(df.columns),
        "size_bytes": output_path.stat().st_size if output_path.exists() else 0,
        "schema": schema,
        "quality_issues": metadata.get("quality_issues", []),
    }

