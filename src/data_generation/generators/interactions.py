"""User interaction data generator for product recommendations."""

from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd
from faker import Faker
from loguru import logger


# Interaction type distribution (funnel)
INTERACTION_DISTRIBUTION = {
    "view": 0.70,
    "cart": 0.15,
    "purchase": 0.08,
    "rating": 0.05,
    "wishlist": 0.02,
}

# Rating distribution (skewed positive)
RATING_DISTRIBUTION = {
    5: 0.40,
    4: 0.30,
    3: 0.15,
    2: 0.10,
    1: 0.05,
}

# Device distribution
DEVICE_DISTRIBUTION = {
    "desktop": 0.35,
    "mobile": 0.55,
    "tablet": 0.10,
}

# Referrer sources
REFERRERS = [
    "direct",
    "google",
    "facebook",
    "instagram",
    "email_campaign",
    "affiliate",
    "twitter",
    "pinterest",
    "tiktok",
    "organic_search",
]


def generate_interactions(
    customers_df: pd.DataFrame,
    products_df: pd.DataFrame,
    n: int = 200000,
    start_date: str = "2024-01-01",
    end_date: str = "2025-11-15",
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate user-product interaction data.
    
    Args:
        customers_df: Customer DataFrame (users map to customers)
        products_df: Product DataFrame for FK relationships
        n: Number of interactions to generate
        start_date: Start of date range
        end_date: End of date range
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with interaction data
    """
    if seed is not None:
        np.random.seed(seed)
        Faker.seed(seed)
    
    fake = Faker()
    logger.info(f"Generating {n:,} user interactions...")
    
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    total_days = (end - start).days
    
    # Create user pool (mix of customers and anonymous users)
    # 60% are known customers, 40% are anonymous visitors
    customer_ids = customers_df["customer_id"].values
    n_anonymous = int(len(customer_ids) * 0.5)
    anonymous_ids = [f"ANON-{i:08d}" for i in range(n_anonymous)]
    
    all_user_ids = list(customer_ids) + anonymous_ids
    
    # Product weights (popular products get more interactions)
    product_ids = products_df["product_id"].values
    product_ratings = products_df["rating_avg"].fillna(3.0).values
    product_weights = product_ratings / product_ratings.sum()
    
    # Generate sessions (users have multiple interactions per session)
    session_counter = 0
    current_session = None
    current_user = None
    interactions_in_session = 0
    
    interactions = []
    
    for i in range(n):
        # Decide if this is a new session
        if current_session is None or interactions_in_session > np.random.geometric(0.3):
            session_counter += 1
            current_session = f"SESS-{session_counter:08d}"
            interactions_in_session = 0
            
            # 70% chance to be a returning user
            if np.random.random() < 0.7 and i > 0:
                current_user = np.random.choice(all_user_ids[:min(i, len(all_user_ids))])
            else:
                current_user = np.random.choice(all_user_ids)
        
        interactions_in_session += 1
        
        # Select product (weighted by popularity)
        product_id = np.random.choice(product_ids, p=product_weights)
        
        # Generate interaction type
        interaction_type = np.random.choice(
            list(INTERACTION_DISTRIBUTION.keys()),
            p=list(INTERACTION_DISTRIBUTION.values())
        )
        
        # Generate timestamp within session
        day_offset = np.random.randint(0, total_days)
        base_date = start + timedelta(days=day_offset)
        
        # Add time of day (e-commerce peaks in evening)
        hour_weights = [0.3, 0.2, 0.1, 0.1, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0,
                       6.0, 5.0, 4.0, 4.0, 5.0, 6.0, 7.0, 8.0, 7.0, 5.0, 3.0, 1.0]
        hour_weights = np.array(hour_weights) / sum(hour_weights)
        hour = np.random.choice(range(24), p=hour_weights)
        
        timestamp = base_date.replace(
            hour=hour,
            minute=np.random.randint(0, 60),
            second=np.random.randint(0, 60)
        )
        
        # Generate rating (only for rating interactions)
        rating = None
        if interaction_type == "rating":
            rating = np.random.choice(
                list(RATING_DISTRIBUTION.keys()),
                p=list(RATING_DISTRIBUTION.values())
            )
        
        # Generate duration (for views)
        duration_seconds = None
        if interaction_type == "view":
            # Log-normal distribution for view duration (10s to 30min)
            duration_seconds = int(np.clip(np.random.lognormal(4, 1), 10, 1800))
        
        # Generate device
        device_type = np.random.choice(
            list(DEVICE_DISTRIBUTION.keys()),
            p=list(DEVICE_DISTRIBUTION.values())
        )
        
        # Generate referrer
        referrer = np.random.choice(REFERRERS, p=[
            0.25,  # direct
            0.20,  # google
            0.12,  # facebook
            0.10,  # instagram
            0.10,  # email_campaign
            0.08,  # affiliate
            0.05,  # twitter
            0.04,  # pinterest
            0.03,  # tiktok
            0.03,  # organic_search
        ])
        
        interaction = {
            "interaction_id": f"INT-{i:08d}",
            "user_id": current_user,
            "product_id": product_id,
            "interaction_type": interaction_type,
            "timestamp": timestamp,
            "rating": rating,
            "session_id": current_session,
            "duration_seconds": duration_seconds,
            "referrer": referrer,
            "device_type": device_type,
        }
        
        interactions.append(interaction)
    
    df = pd.DataFrame(interactions)
    
    # Convert timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Calculate funnel metrics
    views = (df["interaction_type"] == "view").sum()
    carts = (df["interaction_type"] == "cart").sum()
    purchases = (df["interaction_type"] == "purchase").sum()
    
    logger.info(f"Generated {len(df):,} interactions")
    logger.debug(f"  Types: {df['interaction_type'].value_counts().to_dict()}")
    logger.debug(f"  View→Cart: {carts/views*100:.1f}%, Cart→Purchase: {purchases/carts*100:.1f}%" if carts > 0 else "")
    logger.debug(f"  Unique users: {df['user_id'].nunique():,}, Unique sessions: {df['session_id'].nunique():,}")
    
    return df


def inject_interaction_quality_issues(
    df: pd.DataFrame,
    valid_product_ids: set,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Inject intentional quality issues into interaction data.
    
    Issues:
    - 5% interactions with invalid product_id (deleted products)
    - 3% null session_id
    """
    if seed is not None:
        np.random.seed(seed)
    
    df = df.copy()
    n = len(df)
    
    # 5% invalid product_id
    invalid_prod_idx = np.random.choice(df.index, size=int(n * 0.05), replace=False)
    df.loc[invalid_prod_idx, "product_id"] = "PROD-DELETED"
    
    # 3% null session_id
    null_sess_idx = np.random.choice(df.index, size=int(n * 0.03), replace=False)
    df.loc[null_sess_idx, "session_id"] = None
    
    logger.debug(f"Injected interaction quality issues: "
                 f"{len(invalid_prod_idx)} invalid products, "
                 f"{len(null_sess_idx)} null sessions")
    
    return df

