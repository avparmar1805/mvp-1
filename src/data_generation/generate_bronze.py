#!/usr/bin/env python3
"""
Bronze Layer Data Generation Script

Generates synthetic e-commerce data for the Agentic Data Product Builder.
Creates 7 datasets with realistic distributions and intentional quality issues.

Usage:
    python -m src.data_generation.generate_bronze --output-dir data/bronze
    python -m src.data_generation.generate_bronze --output-dir data/bronze --seed 42
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from faker import Faker
from loguru import logger

from src.data_generation.generators.customers import (
    generate_customers,
    inject_customer_quality_issues,
)
from src.data_generation.generators.products import (
    generate_products,
    inject_product_quality_issues,
)
from src.data_generation.generators.orders import (
    generate_orders,
    inject_order_quality_issues,
)
from src.data_generation.generators.campaigns import generate_campaigns
from src.data_generation.generators.events import (
    generate_events,
    inject_event_quality_issues,
)
from src.data_generation.generators.interactions import (
    generate_interactions,
    inject_interaction_quality_issues,
)
from src.data_generation.generators.tickets import generate_tickets
from src.data_generation.utils.export import export_to_parquet
from src.data_generation.utils.validation import validate_cross_dataset


def main(
    output_dir: str = "data/bronze",
    seed: int = 42,
    inject_issues: bool = True,
) -> dict:
    """
    Generate all bronze layer datasets.
    
    Args:
        output_dir: Output directory for Parquet files
        seed: Random seed for reproducibility
        inject_issues: Whether to inject quality issues
        
    Returns:
        Dictionary with generation statistics
    """
    start_time = time.time()
    
    # Set random seeds
    np.random.seed(seed)
    Faker.seed(seed)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Bronze Layer Data Generation")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_path.absolute()}")
    logger.info(f"Random seed: {seed}")
    logger.info(f"Inject quality issues: {inject_issues}")
    logger.info("=" * 60)
    
    stats = {}
    datasets = {}
    
    # =========================================================================
    # Step 1: Generate Master Data (Customers, Products)
    # =========================================================================
    logger.info("\n[1/7] Generating customers...")
    customers = generate_customers(n=10000, seed=seed)
    if inject_issues:
        customers = inject_customer_quality_issues(customers, seed=seed)
    
    result = export_to_parquet(
        customers,
        output_path / "customers" / "customers.parquet",
        metadata={
            "name": "bronze.customers",
            "description": "Customer master data with profile and loyalty information",
            "quality_issues": [
                "5% null phone numbers",
                "2% invalid email formats",
                "1% duplicate emails",
            ] if inject_issues else [],
        }
    )
    stats["customers"] = result
    datasets["customers"] = customers
    
    logger.info("\n[2/7] Generating products...")
    products = generate_products(n=1000, seed=seed)
    if inject_issues:
        products = inject_product_quality_issues(products, seed=seed)
    
    result = export_to_parquet(
        products,
        output_path / "products" / "products.parquet",
        metadata={
            "name": "bronze.products",
            "description": "Product catalog with pricing and category information",
            "quality_issues": [
                "3% null rating_avg",
                "1% negative margin_pct",
            ] if inject_issues else [],
        }
    )
    stats["products"] = result
    datasets["products"] = products
    
    # =========================================================================
    # Step 2: Generate Transactional Data
    # =========================================================================
    logger.info("\n[3/7] Generating orders...")
    orders = generate_orders(customers, products, n=100000, seed=seed)
    if inject_issues:
        orders = inject_order_quality_issues(orders, seed=seed)
    
    result = export_to_parquet(
        orders,
        output_path / "orders" / "orders.parquet",
        metadata={
            "name": "bronze.orders",
            "description": "Order transactions with temporal patterns and seasonal variations",
            "quality_issues": [
                "2% null discount_amount",
                "1% duplicate order_id",
                "0.5% negative total_amount",
                "3% mismatched category",
            ] if inject_issues else [],
        }
    )
    stats["orders"] = result
    datasets["orders"] = orders
    
    # =========================================================================
    # Step 3: Generate Marketing Data
    # =========================================================================
    logger.info("\n[4/7] Generating marketing campaigns...")
    campaigns = generate_campaigns(n=50, seed=seed)
    
    result = export_to_parquet(
        campaigns,
        output_path / "marketing_campaigns" / "campaigns.parquet",
        metadata={
            "name": "bronze.marketing_campaigns",
            "description": "Marketing campaign master data with budget and targeting",
        }
    )
    stats["marketing_campaigns"] = result
    datasets["marketing_campaigns"] = campaigns
    
    logger.info("\n[5/7] Generating marketing events...")
    events = generate_events(campaigns, n=500000, seed=seed)
    if inject_issues:
        valid_campaign_ids = set(campaigns["campaign_id"])
        events = inject_event_quality_issues(events, valid_campaign_ids, seed=seed)
    
    result = export_to_parquet(
        events,
        output_path / "marketing_events" / "events.parquet",
        metadata={
            "name": "bronze.marketing_events",
            "description": "Marketing events with funnel metrics (impressions, clicks, conversions)",
            "quality_issues": [
                "1% invalid campaign_id",
                "2% null location",
            ] if inject_issues else [],
        }
    )
    stats["marketing_events"] = result
    datasets["marketing_events"] = events
    
    # =========================================================================
    # Step 4: Generate User Interaction Data
    # =========================================================================
    logger.info("\n[6/7] Generating user interactions...")
    interactions = generate_interactions(customers, products, n=200000, seed=seed)
    if inject_issues:
        valid_product_ids = set(products["product_id"])
        interactions = inject_interaction_quality_issues(interactions, valid_product_ids, seed=seed)
    
    result = export_to_parquet(
        interactions,
        output_path / "user_interactions" / "interactions.parquet",
        metadata={
            "name": "bronze.user_interactions",
            "description": "User-product interactions for recommendation features",
            "quality_issues": [
                "5% invalid product_id",
                "3% null session_id",
            ] if inject_issues else [],
        }
    )
    stats["user_interactions"] = result
    datasets["user_interactions"] = interactions
    
    # =========================================================================
    # Step 5: Generate Support Ticket Data
    # =========================================================================
    logger.info("\n[7/7] Generating support tickets...")
    tickets = generate_tickets(customers, n=5000, seed=seed)
    
    result = export_to_parquet(
        tickets,
        output_path / "support_tickets" / "tickets.parquet",
        metadata={
            "name": "bronze.support_tickets",
            "description": "Customer support tickets with resolution metrics",
        }
    )
    stats["support_tickets"] = result
    datasets["support_tickets"] = tickets
    
    # =========================================================================
    # Step 6: Cross-Dataset Validation
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Cross-Dataset Validation")
    logger.info("=" * 60)
    
    validation_results = validate_cross_dataset(datasets)
    for result in validation_results:
        status = "✓" if result["passed"] else "✗"
        logger.info(f"  {status} {result['rule']}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    elapsed_time = time.time() - start_time
    total_rows = sum(s["row_count"] for s in stats.values())
    total_size = sum(s["size_bytes"] for s in stats.values())
    
    logger.info("\n" + "=" * 60)
    logger.info("Generation Summary")
    logger.info("=" * 60)
    
    for name, s in stats.items():
        logger.info(f"  {name}: {s['row_count']:,} rows ({s['size_bytes'] / 1024 / 1024:.2f} MB)")
    
    logger.info("-" * 60)
    logger.info(f"  Total: {total_rows:,} rows ({total_size / 1024 / 1024:.2f} MB)")
    logger.info(f"  Time: {elapsed_time:.2f} seconds")
    logger.info("=" * 60)
    
    return {
        "stats": stats,
        "total_rows": total_rows,
        "total_size_bytes": total_size,
        "elapsed_time": elapsed_time,
        "validation_results": validation_results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate bronze layer synthetic data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/bronze",
        help="Output directory for Parquet files (default: data/bronze)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--no-quality-issues",
        action="store_true",
        help="Don't inject intentional quality issues"
    )
    
    args = parser.parse_args()
    
    try:
        result = main(
            output_dir=args.output_dir,
            seed=args.seed,
            inject_issues=not args.no_quality_issues,
        )
        
        logger.info("\n✅ Bronze layer generation complete!")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"\n❌ Generation failed: {e}")
        raise

