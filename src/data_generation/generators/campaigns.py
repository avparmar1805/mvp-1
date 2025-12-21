"""Marketing campaign data generator."""

from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd
from faker import Faker
from loguru import logger


# Channel distribution
CHANNEL_DISTRIBUTION = {
    "email": 0.30,
    "social": 0.25,
    "search": 0.20,
    "display": 0.15,
    "video": 0.10,
}

# Objective distribution
OBJECTIVE_DISTRIBUTION = {
    "awareness": 0.30,
    "consideration": 0.35,
    "conversion": 0.35,
}

# Status distribution
STATUS_DISTRIBUTION = {
    "completed": 0.60,
    "active": 0.30,
    "paused": 0.10,
}

# Target audience options
TARGET_AUDIENCES = [
    "New Customers",
    "Returning Customers",
    "High Value Customers",
    "Cart Abandoners",
    "Newsletter Subscribers",
    "Social Media Followers",
    "Mobile App Users",
    "Premium Members",
]

# Campaign owners
CAMPAIGN_OWNERS = [
    "Sarah Johnson",
    "Mike Chen",
    "Emily Rodriguez",
    "David Kim",
    "Lisa Thompson",
]


def generate_campaigns(
    n: int = 50,
    start_date: str = "2024-01-01",
    end_date: str = "2025-11-15",
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate marketing campaign data.
    
    Args:
        n: Number of campaigns to generate
        start_date: Start of date range for campaigns
        end_date: End of date range for campaigns
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with campaign data
    """
    if seed is not None:
        np.random.seed(seed)
        Faker.seed(seed)
    
    fake = Faker()
    logger.info(f"Generating {n:,} marketing campaigns...")
    
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    total_days = (end - start).days
    
    campaigns = []
    
    for i in range(n):
        # Generate campaign dates
        campaign_start_offset = np.random.randint(0, total_days - 90)
        campaign_start = start + timedelta(days=campaign_start_offset)
        
        # Campaign duration: 7-90 days
        duration = np.random.randint(7, 91)
        campaign_end = campaign_start + timedelta(days=duration)
        
        # Ensure end date doesn't exceed our range
        if campaign_end > end:
            campaign_end = end
        
        # Generate channel and related attributes
        channel = np.random.choice(
            list(CHANNEL_DISTRIBUTION.keys()),
            p=list(CHANNEL_DISTRIBUTION.values())
        )
        
        # Budget varies by channel
        base_budget = np.random.uniform(5000, 100000)
        channel_multipliers = {
            "email": 0.5,
            "social": 1.0,
            "search": 1.2,
            "display": 0.8,
            "video": 1.5,
        }
        budget = round(base_budget * channel_multipliers[channel], 2)
        
        # Determine status based on dates
        now = end  # Use end_date as "current" time
        if campaign_end < now:
            status = "completed"
        elif campaign_start > now:
            status = "paused"  # Future campaigns marked as paused
        else:
            status = np.random.choice(["active", "paused"], p=[0.85, 0.15])
        
        # Generate campaign name
        objectives_for_name = {
            "awareness": ["Brand Awareness", "Reach", "Visibility"],
            "consideration": ["Engagement", "Traffic", "Interest"],
            "conversion": ["Sales", "Conversion", "Revenue"],
        }
        objective = np.random.choice(
            list(OBJECTIVE_DISTRIBUTION.keys()),
            p=list(OBJECTIVE_DISTRIBUTION.values())
        )
        
        name_prefix = np.random.choice(objectives_for_name[objective])
        season = _get_season(campaign_start.month)
        campaign_name = f"{season} {name_prefix} - {channel.title()} #{i+1:02d}"
        
        campaign = {
            "campaign_id": f"CAMP-{i:08d}",
            "campaign_name": campaign_name,
            "channel": channel,
            "start_date": campaign_start.date(),
            "end_date": campaign_end.date(),
            "budget": budget,
            "target_audience": np.random.choice(TARGET_AUDIENCES),
            "objective": objective,
            "status": status,
            "owner": np.random.choice(CAMPAIGN_OWNERS),
        }
        
        campaigns.append(campaign)
    
    df = pd.DataFrame(campaigns)
    
    # Convert date columns
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    
    logger.info(f"Generated {len(df):,} campaigns")
    logger.debug(f"  Channels: {df['channel'].value_counts().to_dict()}")
    logger.debug(f"  Status: {df['status'].value_counts().to_dict()}")
    logger.debug(f"  Total budget: ${df['budget'].sum():,.2f}")
    
    return df


def _get_season(month: int) -> str:
    """Get season name from month."""
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"

