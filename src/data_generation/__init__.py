"""Data generation module for Bronze layer datasets."""

from src.data_generation.generators import (
    generate_customers,
    generate_products,
    generate_orders,
    generate_campaigns,
    generate_events,
    generate_interactions,
    generate_tickets,
)

from src.data_generation.utils import (
    export_to_parquet,
    validate_dataset,
    validate_cross_dataset,
    inject_quality_issues,
)

__all__ = [
    # Generators
    "generate_customers",
    "generate_products",
    "generate_orders",
    "generate_campaigns",
    "generate_events",
    "generate_interactions",
    "generate_tickets",
    # Utils
    "export_to_parquet",
    "validate_dataset",
    "validate_cross_dataset",
    "inject_quality_issues",
]

