"""Customer data generator."""

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from faker import Faker
from loguru import logger


def generate_customers(
    n: int = 10000,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate customer master data.
    
    Args:
        n: Number of customers to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with customer data
    """
    if seed is not None:
        np.random.seed(seed)
        Faker.seed(seed)
    
    fake = Faker()
    logger.info(f"Generating {n:,} customers...")
    
    customers = []
    
    for i in range(n):
        # Generate signup date (last 5 years)
        signup_date = fake.date_between(start_date="-5y", end_date="today")
        
        # Generate lifetime value using log-normal distribution
        lifetime_value = np.random.lognormal(mean=5, sigma=1.5)
        lifetime_value = min(lifetime_value, 50000)  # Cap at $50k
        
        # Determine loyalty tier based on LTV
        if lifetime_value > 10000:
            tier = "platinum"
        elif lifetime_value > 2000:
            tier = "gold"
        elif lifetime_value > 500:
            tier = "silver"
        else:
            tier = "bronze"
        
        # Determine segment (correlated with tier)
        if tier == "platinum":
            segment = np.random.choice(
                ["high_value", "regular"],
                p=[0.8, 0.2]
            )
        elif tier == "gold":
            segment = np.random.choice(
                ["high_value", "regular", "at_risk"],
                p=[0.5, 0.4, 0.1]
            )
        elif tier == "silver":
            segment = np.random.choice(
                ["regular", "at_risk", "churned"],
                p=[0.6, 0.3, 0.1]
            )
        else:
            segment = np.random.choice(
                ["regular", "at_risk", "churned"],
                p=[0.5, 0.3, 0.2]
            )
        
        customer = {
            "customer_id": f"CUST-{i:08d}",
            "name": fake.name(),
            "email": fake.email(),
            "phone": fake.phone_number(),
            "signup_date": signup_date,
            "loyalty_tier": tier,
            "total_lifetime_value": round(lifetime_value, 2),
            "segment": segment,
            "address_city": fake.city(),
            "address_state": fake.state_abbr(),
            "address_country": "USA",
            "date_of_birth": fake.date_of_birth(minimum_age=18, maximum_age=80),
            "gender": np.random.choice(
                ["Male", "Female", "Other", "Prefer not to say"],
                p=[0.48, 0.48, 0.02, 0.02]
            ),
            "preferred_contact_method": np.random.choice(
                ["email", "phone", "sms"],
                p=[0.6, 0.3, 0.1]
            ),
        }
        
        customers.append(customer)
    
    df = pd.DataFrame(customers)
    
    # Convert date columns
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    df["date_of_birth"] = pd.to_datetime(df["date_of_birth"])
    
    logger.info(f"Generated {len(df):,} customers")
    logger.debug(f"  Loyalty tiers: {df['loyalty_tier'].value_counts().to_dict()}")
    logger.debug(f"  Segments: {df['segment'].value_counts().to_dict()}")
    
    return df


def inject_customer_quality_issues(
    df: pd.DataFrame,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Inject intentional quality issues into customer data.
    
    Issues:
    - 5% null phone numbers
    - 2% invalid email formats
    - 1% duplicate emails
    """
    if seed is not None:
        np.random.seed(seed)
    
    df = df.copy()
    n = len(df)
    
    # 5% null phone numbers
    null_phone_idx = np.random.choice(df.index, size=int(n * 0.05), replace=False)
    df.loc[null_phone_idx, "phone"] = None
    
    # 2% invalid email formats
    invalid_email_idx = np.random.choice(df.index, size=int(n * 0.02), replace=False)
    df.loc[invalid_email_idx, "email"] = df.loc[invalid_email_idx, "name"].str.replace(" ", "_") + "_invalid"
    
    # 1% duplicate emails (copy from other customers)
    dup_count = int(n * 0.01)
    source_idx = np.random.choice(df.index, size=dup_count, replace=True)
    target_idx = np.random.choice(df.index, size=dup_count, replace=False)
    df.loc[target_idx, "email"] = df.loc[source_idx, "email"].values
    
    logger.debug(f"Injected customer quality issues: "
                 f"{len(null_phone_idx)} null phones, "
                 f"{len(invalid_email_idx)} invalid emails, "
                 f"{dup_count} duplicate emails")
    
    return df

