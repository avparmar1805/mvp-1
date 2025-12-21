"""Marketing event data generator with funnel logic."""

from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd
from faker import Faker
from loguru import logger


# Device distribution
DEVICE_DISTRIBUTION = {
    "desktop": 0.40,
    "mobile": 0.50,
    "tablet": 0.10,
}

# Browser distribution (varies by device)
BROWSER_DISTRIBUTION = {
    "desktop": {"Chrome": 0.65, "Safari": 0.15, "Firefox": 0.12, "Edge": 0.08},
    "mobile": {"Chrome Mobile": 0.45, "Safari Mobile": 0.45, "Samsung Browser": 0.10},
    "tablet": {"Safari": 0.60, "Chrome": 0.35, "Other": 0.05},
}

# Location distribution
LOCATIONS = [
    "New York, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX", "Phoenix, AZ",
    "Philadelphia, PA", "San Antonio, TX", "San Diego, CA", "Dallas, TX", "San Jose, CA",
    "Austin, TX", "Jacksonville, FL", "Fort Worth, TX", "Columbus, OH", "Charlotte, NC",
    "Seattle, WA", "Denver, CO", "Boston, MA", "Portland, OR", "Atlanta, GA",
]


def generate_events(
    campaigns_df: pd.DataFrame,
    n: int = 500000,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate marketing events (impressions, clicks, conversions) with funnel logic.
    
    Args:
        campaigns_df: Campaign DataFrame for FK relationships
        n: Total number of events to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with event data
    """
    if seed is not None:
        np.random.seed(seed)
        Faker.seed(seed)
    
    fake = Faker()
    logger.info(f"Generating {n:,} marketing events...")
    
    # Event type distribution based on funnel:
    # - impression: 80%
    # - click: 15% (~18% CTR from impressions)  
    # - conversion: 5% (~33% CVR from clicks)
    event_weights = {
        "impression": 0.80,
        "click": 0.15,
        "conversion": 0.05,
    }
    
    # Pre-generate campaign assignments (weighted by budget)
    campaign_ids = campaigns_df["campaign_id"].values
    campaign_budgets = campaigns_df["budget"].values
    campaign_weights = campaign_budgets / campaign_budgets.sum()
    
    # Create lookup dictionaries
    campaign_dates = campaigns_df.set_index("campaign_id")[["start_date", "end_date"]].to_dict("index")
    campaign_channels = campaigns_df.set_index("campaign_id")["channel"].to_dict()
    
    events = []
    user_counter = 0
    
    # Generate events in batches for better performance
    batch_size = 10000
    num_batches = (n + batch_size - 1) // batch_size
    
    for batch in range(num_batches):
        batch_start = batch * batch_size
        batch_end = min((batch + 1) * batch_size, n)
        batch_n = batch_end - batch_start
        
        # Generate batch data
        selected_campaigns = np.random.choice(campaign_ids, size=batch_n, p=campaign_weights)
        event_types = np.random.choice(
            list(event_weights.keys()),
            size=batch_n,
            p=list(event_weights.values())
        )
        devices = np.random.choice(
            list(DEVICE_DISTRIBUTION.keys()),
            size=batch_n,
            p=list(DEVICE_DISTRIBUTION.values())
        )
        
        for i in range(batch_n):
            campaign_id = selected_campaigns[i]
            event_type = event_types[i]
            device = devices[i]
            
            # Get campaign date range
            camp_info = campaign_dates.get(campaign_id)
            if camp_info:
                camp_start = pd.Timestamp(camp_info["start_date"])
                camp_end = pd.Timestamp(camp_info["end_date"])
                days_range = max((camp_end - camp_start).days, 1)
                event_date = camp_start + timedelta(
                    days=np.random.randint(0, days_range),
                    hours=np.random.randint(0, 24),
                    minutes=np.random.randint(0, 60),
                    seconds=np.random.randint(0, 60)
                )
            else:
                event_date = fake.date_time_between(start_date="-1y", end_date="now")
            
            # Generate browser based on device
            browser_dist = BROWSER_DISTRIBUTION[device]
            browser = np.random.choice(
                list(browser_dist.keys()),
                p=list(browser_dist.values())
            )
            
            # Generate user_id (some users have multiple events)
            if np.random.random() < 0.7:  # 70% returning users
                user_id = f"USR-{np.random.randint(0, user_counter + 1):08d}"
            else:
                user_counter += 1
                user_id = f"USR-{user_counter:08d}"
            
            # Calculate cost based on event type and channel
            channel = campaign_channels.get(campaign_id, "display")
            cost = _calculate_event_cost(event_type, channel)
            
            # Calculate revenue (only for conversions)
            revenue = 0.0
            if event_type == "conversion":
                revenue = round(np.random.uniform(10, 500), 2)
            
            event = {
                "event_id": f"EVT-{batch_start + i:08d}",
                "campaign_id": campaign_id,
                "event_date": event_date,
                "event_type": event_type,
                "user_id": user_id,
                "cost": cost,
                "revenue": revenue,
                "device_type": device,
                "browser": browser,
                "location": np.random.choice(LOCATIONS),
            }
            
            events.append(event)
    
    df = pd.DataFrame(events)
    
    # Convert date column
    df["event_date"] = pd.to_datetime(df["event_date"])
    
    # Calculate metrics
    impressions = (df["event_type"] == "impression").sum()
    clicks = (df["event_type"] == "click").sum()
    conversions = (df["event_type"] == "conversion").sum()
    
    logger.info(f"Generated {len(df):,} events")
    logger.debug(f"  Impressions: {impressions:,}, Clicks: {clicks:,}, Conversions: {conversions:,}")
    logger.debug(f"  CTR: {clicks/impressions*100:.2f}%, CVR: {conversions/clicks*100:.2f}%" if clicks > 0 else "")
    logger.debug(f"  Total cost: ${df['cost'].sum():,.2f}, Total revenue: ${df['revenue'].sum():,.2f}")
    
    return df


def _calculate_event_cost(event_type: str, channel: str) -> float:
    """Calculate cost for an event based on type and channel."""
    if event_type == "impression":
        # CPM basis ($1-5 per 1000 impressions)
        base_cpm = np.random.uniform(1, 5)
        return round(base_cpm / 1000, 4)
    
    elif event_type == "click":
        # CPC varies by channel
        cpc_ranges = {
            "email": (0.10, 0.50),
            "social": (0.30, 2.00),
            "search": (1.00, 5.00),
            "display": (0.20, 1.00),
            "video": (0.50, 3.00),
        }
        min_cpc, max_cpc = cpc_ranges.get(channel, (0.50, 2.00))
        return round(np.random.uniform(min_cpc, max_cpc), 2)
    
    else:  # conversion
        # No direct cost for conversions
        return 0.0


def inject_event_quality_issues(
    df: pd.DataFrame,
    valid_campaign_ids: set,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Inject intentional quality issues into event data.
    
    Issues:
    - 1% events with invalid campaign_id
    - 2% null location
    """
    if seed is not None:
        np.random.seed(seed)
    
    df = df.copy()
    n = len(df)
    
    # 1% invalid campaign_id
    invalid_camp_idx = np.random.choice(df.index, size=int(n * 0.01), replace=False)
    df.loc[invalid_camp_idx, "campaign_id"] = "CAMP-INVALID"
    
    # 2% null location
    null_loc_idx = np.random.choice(df.index, size=int(n * 0.02), replace=False)
    df.loc[null_loc_idx, "location"] = None
    
    logger.debug(f"Injected event quality issues: "
                 f"{len(invalid_camp_idx)} invalid campaigns, "
                 f"{len(null_loc_idx)} null locations")
    
    return df

