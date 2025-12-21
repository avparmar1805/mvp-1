"""Support ticket data generator."""

from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd
from faker import Faker
from loguru import logger


# Ticket category distribution
CATEGORY_DISTRIBUTION = {
    "product_issue": 0.40,
    "billing": 0.25,
    "shipping": 0.20,
    "account": 0.15,
}

# Priority distribution
PRIORITY_DISTRIBUTION = {
    "low": 0.40,
    "medium": 0.35,
    "high": 0.20,
    "critical": 0.05,
}

# Status distribution
STATUS_DISTRIBUTION = {
    "closed": 0.70,
    "resolved": 0.15,
    "in_progress": 0.10,
    "open": 0.05,
}

# Satisfaction score distribution (for resolved tickets)
SATISFACTION_DISTRIBUTION = {
    5: 0.30,
    4: 0.35,
    3: 0.20,
    2: 0.10,
    1: 0.05,
}

# Channel distribution
CHANNEL_DISTRIBUTION = {
    "email": 0.40,
    "chat": 0.30,
    "phone": 0.20,
    "self_service": 0.10,
}

# Resolution time ranges by priority (in hours)
RESOLUTION_TIME_RANGES = {
    "critical": (1, 8),
    "high": (4, 24),
    "medium": (12, 48),
    "low": (24, 72),
}

# Support agents
SUPPORT_AGENTS = [
    "AGT-001", "AGT-002", "AGT-003", "AGT-004", "AGT-005",
    "AGT-006", "AGT-007", "AGT-008", "AGT-009", "AGT-010",
]


def generate_tickets(
    customers_df: pd.DataFrame,
    n: int = 5000,
    start_date: str = "2024-01-01",
    end_date: str = "2025-11-15",
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate support ticket data.
    
    Args:
        customers_df: Customer DataFrame for FK relationships
        n: Number of tickets to generate
        start_date: Start of date range
        end_date: End of date range
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with ticket data
    """
    if seed is not None:
        np.random.seed(seed)
        Faker.seed(seed)
    
    fake = Faker()
    logger.info(f"Generating {n:,} support tickets...")
    
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    total_days = (end - start).days
    
    # Customer weights (some customers create more tickets)
    customer_ids = customers_df["customer_id"].values
    # Customers with 'at_risk' or 'churned' segment create more tickets
    segments = customers_df["segment"].values
    ticket_weights = np.where(
        np.isin(segments, ["at_risk", "churned"]),
        2.0,  # 2x more likely to create tickets
        1.0
    )
    ticket_weights = ticket_weights / ticket_weights.sum()
    
    tickets = []
    
    for i in range(n):
        # Select customer (weighted)
        customer_id = np.random.choice(customer_ids, p=ticket_weights)
        
        # Generate creation date
        day_offset = np.random.randint(0, total_days)
        created_at = start + timedelta(
            days=day_offset,
            hours=np.random.randint(8, 20),  # Business hours bias
            minutes=np.random.randint(0, 60),
            seconds=np.random.randint(0, 60)
        )
        
        # Generate category
        category = np.random.choice(
            list(CATEGORY_DISTRIBUTION.keys()),
            p=list(CATEGORY_DISTRIBUTION.values())
        )
        
        # Generate priority (correlated with category)
        if category == "billing":
            # Billing issues tend to be higher priority
            priority = np.random.choice(
                list(PRIORITY_DISTRIBUTION.keys()),
                p=[0.20, 0.30, 0.35, 0.15]
            )
        elif category == "product_issue":
            priority = np.random.choice(
                list(PRIORITY_DISTRIBUTION.keys()),
                p=[0.35, 0.35, 0.25, 0.05]
            )
        else:
            priority = np.random.choice(
                list(PRIORITY_DISTRIBUTION.keys()),
                p=list(PRIORITY_DISTRIBUTION.values())
            )
        
        # Generate status
        status = np.random.choice(
            list(STATUS_DISTRIBUTION.keys()),
            p=list(STATUS_DISTRIBUTION.values())
        )
        
        # Generate resolution time and resolved_at (for resolved/closed tickets)
        resolved_at = None
        resolution_time_hours = None
        satisfaction_score = None
        
        if status in ["resolved", "closed"]:
            # Calculate resolution time based on priority
            min_hours, max_hours = RESOLUTION_TIME_RANGES[priority]
            resolution_time_hours = round(np.random.uniform(min_hours, max_hours), 2)
            
            resolved_at = created_at + timedelta(hours=resolution_time_hours)
            
            # Ensure resolved_at doesn't exceed end date
            if resolved_at > end:
                resolved_at = end
                resolution_time_hours = round((resolved_at - created_at).total_seconds() / 3600, 2)
            
            # Generate satisfaction score (only for resolved tickets)
            # Lower satisfaction for longer resolution times
            if resolution_time_hours > max_hours * 0.8:
                # Slower resolution = lower satisfaction
                satisfaction_score = np.random.choice(
                    list(SATISFACTION_DISTRIBUTION.keys()),
                    p=[0.15, 0.25, 0.30, 0.20, 0.10]
                )
            else:
                satisfaction_score = np.random.choice(
                    list(SATISFACTION_DISTRIBUTION.keys()),
                    p=list(SATISFACTION_DISTRIBUTION.values())
                )
        
        # Generate channel
        channel = np.random.choice(
            list(CHANNEL_DISTRIBUTION.keys()),
            p=list(CHANNEL_DISTRIBUTION.values())
        )
        
        # Assign agent
        agent_id = np.random.choice(SUPPORT_AGENTS)
        
        ticket = {
            "ticket_id": f"TKT-{i:08d}",
            "customer_id": customer_id,
            "created_at": created_at,
            "resolved_at": resolved_at,
            "category": category,
            "priority": priority,
            "status": status,
            "satisfaction_score": satisfaction_score,
            "resolution_time_hours": resolution_time_hours,
            "agent_id": agent_id,
            "channel": channel,
        }
        
        tickets.append(ticket)
    
    df = pd.DataFrame(tickets)
    
    # Convert datetime columns
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["resolved_at"] = pd.to_datetime(df["resolved_at"])
    
    # Calculate metrics
    resolved_tickets = df[df["status"].isin(["resolved", "closed"])]
    avg_resolution_time = resolved_tickets["resolution_time_hours"].mean()
    avg_satisfaction = resolved_tickets["satisfaction_score"].mean()
    
    logger.info(f"Generated {len(df):,} tickets")
    logger.debug(f"  Categories: {df['category'].value_counts().to_dict()}")
    logger.debug(f"  Status: {df['status'].value_counts().to_dict()}")
    logger.debug(f"  Avg resolution time: {avg_resolution_time:.1f} hours")
    logger.debug(f"  Avg satisfaction: {avg_satisfaction:.2f}/5")
    
    return df

