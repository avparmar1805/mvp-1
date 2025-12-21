"""Order data generator with temporal patterns."""

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from faker import Faker
from loguru import logger


# Regional distribution
REGION_DISTRIBUTION = {
    "North": 0.30,
    "South": 0.25,
    "East": 0.25,
    "West": 0.20,
}

# Order status distribution
STATUS_DISTRIBUTION = {
    "completed": 0.85,
    "cancelled": 0.10,
    "pending": 0.05,
}


def generate_orders(
    customers_df: pd.DataFrame,
    products_df: pd.DataFrame,
    n: int = 100000,
    start_date: str = "2023-01-01",
    end_date: str = "2025-11-15",
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate order transactions with temporal patterns.
    
    Args:
        customers_df: Customer DataFrame for FK relationships
        products_df: Product DataFrame for FK relationships  
        n: Number of orders to generate
        start_date: Start of date range
        end_date: End of date range
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with order data
    """
    if seed is not None:
        np.random.seed(seed)
        Faker.seed(seed)
    
    fake = Faker()
    logger.info(f"Generating {n:,} orders...")
    
    # Create date range
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    
    # Generate customer weights (power law: 20% customers make 80% orders)
    customer_weights = _generate_power_law_weights(len(customers_df))
    customer_ids = customers_df["customer_id"].values
    
    # Get product data for price lookup
    product_ids = products_df["product_id"].values
    product_prices = products_df.set_index("product_id")["price"].to_dict()
    product_categories = products_df.set_index("product_id")["category"].to_dict()
    
    # Pre-generate dates with temporal patterns
    order_dates = _generate_temporal_dates(n, start, end)
    
    # Pre-generate random selections
    selected_customers = np.random.choice(customer_ids, size=n, p=customer_weights)
    selected_products = np.random.choice(product_ids, size=n)
    regions = np.random.choice(
        list(REGION_DISTRIBUTION.keys()),
        size=n,
        p=list(REGION_DISTRIBUTION.values())
    )
    statuses = np.random.choice(
        list(STATUS_DISTRIBUTION.keys()),
        size=n,
        p=list(STATUS_DISTRIBUTION.values())
    )
    
    orders = []
    
    for i in range(n):
        product_id = selected_products[i]
        unit_price = product_prices.get(product_id, 50.0)
        
        # Generate quantity (most orders are 1-3 items)
        quantity = int(np.random.exponential(2)) + 1
        quantity = min(quantity, 10)
        
        subtotal = quantity * unit_price
        
        # Apply discount (20% of orders get discount)
        discount_amount = 0.0
        if np.random.random() < 0.20:
            discount_pct = np.random.uniform(0.05, 0.30)
            discount_amount = round(subtotal * discount_pct, 2)
        
        # Calculate tax and shipping
        tax_amount = round(subtotal * 0.08, 2)
        shipping_cost = round(np.random.uniform(5, 25), 2)
        
        # Free shipping for orders over $100
        if subtotal > 100:
            shipping_cost = 0.0
        
        total_amount = round(subtotal - discount_amount + tax_amount + shipping_cost, 2)
        
        order = {
            "order_id": f"ORD-{i:08d}",
            "customer_id": selected_customers[i],
            "product_id": product_id,
            "order_date": order_dates[i],
            "quantity": quantity,
            "unit_price": unit_price,
            "total_amount": total_amount,
            "region": regions[i],
            "category": product_categories.get(product_id, "Unknown"),
            "status": statuses[i],
            "discount_amount": discount_amount,
            "tax_amount": tax_amount,
            "shipping_cost": shipping_cost,
        }
        
        orders.append(order)
    
    df = pd.DataFrame(orders)
    
    # Convert date column
    df["order_date"] = pd.to_datetime(df["order_date"])
    
    logger.info(f"Generated {len(df):,} orders")
    logger.debug(f"  Date range: {df['order_date'].min()} to {df['order_date'].max()}")
    logger.debug(f"  Total revenue: ${df['total_amount'].sum():,.2f}")
    logger.debug(f"  Status: {df['status'].value_counts().to_dict()}")
    
    return df


def _generate_power_law_weights(n: int, alpha: float = 1.5) -> np.ndarray:
    """
    Generate power law distribution weights.
    This creates a distribution where top customers order disproportionately more.
    """
    weights = np.array([1.0 / (i + 1) ** alpha for i in range(n)])
    return weights / weights.sum()


def _generate_temporal_dates(
    n: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list:
    """
    Generate dates with realistic temporal patterns.
    
    Patterns:
    - Weekday bias: 60% Mon-Fri, 40% Sat-Sun
    - Seasonal: +30% in Nov-Dec
    - Daily pattern: Peaks at lunch and evening
    """
    dates = []
    total_days = (end - start).days
    
    for _ in range(n):
        # Generate base random day
        day_offset = np.random.randint(0, total_days)
        date = start + timedelta(days=day_offset)
        
        # Apply seasonal bias (try again if not in peak season during Nov-Dec)
        month = date.month
        if month in [11, 12]:
            # 30% more likely to keep orders in Nov-Dec
            pass
        else:
            # 30% chance to regenerate in Nov-Dec
            if np.random.random() < 0.15:
                # Shift to Nov-Dec
                if np.random.random() < 0.5:
                    date = date.replace(month=11, day=min(date.day, 30))
                else:
                    date = date.replace(month=12, day=min(date.day, 31))
                # Make sure it's within range
                if date < start:
                    date = start + timedelta(days=np.random.randint(0, total_days))
                elif date > end:
                    date = end - timedelta(days=np.random.randint(0, 30))
        
        # Apply weekday bias
        weekday = date.weekday()
        if weekday >= 5:  # Weekend
            # 40% chance to keep weekend orders
            if np.random.random() > 0.67:  # Shift some to weekdays
                shift = np.random.randint(1, 3)
                date = date - timedelta(days=shift if weekday == 5 else shift + 1)
        
        # Add time of day (peaks at 12-14 and 18-20)
        hour_weights = [0.5, 0.3, 0.2, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                       8.0, 7.0, 5.0, 4.0, 4.0, 5.0, 7.0, 8.0, 6.0, 4.0, 2.0, 1.0]
        hour_weights = np.array(hour_weights) / sum(hour_weights)
        hour = np.random.choice(range(24), p=hour_weights)
        minute = np.random.randint(0, 60)
        second = np.random.randint(0, 60)
        
        date = date.replace(hour=hour, minute=minute, second=second)
        dates.append(date)
    
    return dates


def inject_order_quality_issues(
    df: pd.DataFrame,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Inject intentional quality issues into order data.
    
    Issues:
    - 2% null discount_amount
    - 1% duplicate order_id
    - 0.5% negative total_amount
    - 3% mismatched category
    """
    if seed is not None:
        np.random.seed(seed)
    
    df = df.copy()
    n = len(df)
    
    # 2% null discount_amount
    null_discount_idx = np.random.choice(df.index, size=int(n * 0.02), replace=False)
    df.loc[null_discount_idx, "discount_amount"] = None
    
    # 1% duplicate order_id (duplicate entire rows)
    dup_count = int(n * 0.01)
    dup_indices = np.random.choice(df.index, size=dup_count, replace=True)
    duplicates = df.loc[dup_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # 0.5% negative total_amount
    neg_amount_idx = np.random.choice(df.index, size=int(len(df) * 0.005), replace=False)
    df.loc[neg_amount_idx, "total_amount"] = -abs(df.loc[neg_amount_idx, "total_amount"])
    
    # 3% mismatched category
    mismatch_idx = np.random.choice(df.index, size=int(len(df) * 0.03), replace=False)
    df.loc[mismatch_idx, "category"] = "MISMATCH"
    
    logger.debug(f"Injected order quality issues: "
                 f"{len(null_discount_idx)} null discounts, "
                 f"{dup_count} duplicates, "
                 f"{len(neg_amount_idx)} negative amounts, "
                 f"{len(mismatch_idx)} mismatched categories")
    
    return df

