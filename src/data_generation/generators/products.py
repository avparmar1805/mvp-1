"""Product data generator."""

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from faker import Faker
from loguru import logger


# Product category configurations
CATEGORY_CONFIG = {
    "Electronics": {
        "subcategories": ["Smartphones", "Laptops", "Tablets", "Accessories", "Audio", "Cameras"],
        "price_range": (50, 2000),
        "margin_range": (0.15, 0.35),
        "brands": ["TechCorp", "ElectroMax", "SmartGear", "DigiPro", "ByteWave"],
    },
    "Clothing": {
        "subcategories": ["Men's Wear", "Women's Wear", "Kids", "Sportswear", "Accessories"],
        "price_range": (20, 300),
        "margin_range": (0.40, 0.60),
        "brands": ["FashionHub", "StyleCo", "TrendSetters", "ComfortWear", "UrbanEdge"],
    },
    "Home": {
        "subcategories": ["Furniture", "Kitchen", "Decor", "Bedding", "Storage"],
        "price_range": (15, 500),
        "margin_range": (0.30, 0.50),
        "brands": ["HomeEssentials", "CozyLiving", "ModernNest", "SpaceSmart", "DwellWell"],
    },
    "Books": {
        "subcategories": ["Fiction", "Non-Fiction", "Academic", "Children's", "Self-Help"],
        "price_range": (10, 50),
        "margin_range": (0.25, 0.40),
        "brands": ["PageTurner", "ReadMore", "BookWorm", "LitHouse", "StoryTime"],
    },
    "Sports": {
        "subcategories": ["Fitness", "Outdoor", "Team Sports", "Water Sports", "Winter Sports"],
        "price_range": (25, 800),
        "margin_range": (0.30, 0.45),
        "brands": ["ActiveLife", "SportMax", "FitGear", "OutdoorPro", "GameOn"],
    },
}

# Category distribution
CATEGORY_DISTRIBUTION = {
    "Electronics": 0.25,
    "Clothing": 0.25,
    "Home": 0.20,
    "Books": 0.15,
    "Sports": 0.15,
}


def generate_products(
    n: int = 1000,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate product catalog data.
    
    Args:
        n: Number of products to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with product data
    """
    if seed is not None:
        np.random.seed(seed)
        Faker.seed(seed)
    
    fake = Faker()
    logger.info(f"Generating {n:,} products...")
    
    # Pre-calculate category assignments
    categories = list(CATEGORY_DISTRIBUTION.keys())
    probabilities = list(CATEGORY_DISTRIBUTION.values())
    category_assignments = np.random.choice(categories, size=n, p=probabilities)
    
    products = []
    
    for i in range(n):
        category = category_assignments[i]
        config = CATEGORY_CONFIG[category]
        
        # Select subcategory and brand
        subcategory = np.random.choice(config["subcategories"])
        brand = np.random.choice(config["brands"])
        
        # Generate price within category range
        min_price, max_price = config["price_range"]
        price = round(np.random.uniform(min_price, max_price), 2)
        
        # Generate margin and calculate cost
        min_margin, max_margin = config["margin_range"]
        margin_pct = round(np.random.uniform(min_margin, max_margin), 4)
        cost = round(price * (1 - margin_pct), 2)
        
        # Generate product name
        adjectives = ["Premium", "Essential", "Classic", "Pro", "Ultra", "Basic", "Deluxe"]
        product_name = f"{np.random.choice(adjectives)} {subcategory} {fake.word().title()}"
        
        # Generate rating and reviews (correlated)
        rating_avg = round(np.random.uniform(2.5, 5.0), 2)
        # More reviews for popular products (higher ratings)
        base_reviews = int(np.random.exponential(50))
        review_count = int(base_reviews * (rating_avg / 3.5))
        
        # Stock quantity
        stock_quantity = np.random.randint(0, 500)
        
        # Active status (90% active)
        is_active = np.random.random() < 0.90
        
        product = {
            "product_id": f"PROD-{i:08d}",
            "product_name": product_name,
            "category": category,
            "subcategory": subcategory,
            "brand": brand,
            "price": price,
            "cost": cost,
            "margin_pct": round(margin_pct * 100, 2),  # Store as percentage
            "stock_quantity": stock_quantity,
            "supplier_id": f"SUP-{np.random.randint(1, 51):05d}",
            "created_at": fake.date_between(start_date="-3y", end_date="today"),
            "is_active": is_active,
            "rating_avg": rating_avg,
            "review_count": review_count,
        }
        
        products.append(product)
    
    df = pd.DataFrame(products)
    
    # Convert date column
    df["created_at"] = pd.to_datetime(df["created_at"])
    
    logger.info(f"Generated {len(df):,} products")
    logger.debug(f"  Categories: {df['category'].value_counts().to_dict()}")
    logger.debug(f"  Active: {df['is_active'].sum():,}, Inactive: {(~df['is_active']).sum():,}")
    
    return df


def inject_product_quality_issues(
    df: pd.DataFrame,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Inject intentional quality issues into product data.
    
    Issues:
    - 3% null rating_avg (new products)
    - 1% negative margin_pct (pricing errors)
    """
    if seed is not None:
        np.random.seed(seed)
    
    df = df.copy()
    n = len(df)
    
    # 3% null rating_avg
    null_rating_idx = np.random.choice(df.index, size=int(n * 0.03), replace=False)
    df.loc[null_rating_idx, "rating_avg"] = None
    df.loc[null_rating_idx, "review_count"] = 0
    
    # 1% negative margin_pct
    neg_margin_idx = np.random.choice(df.index, size=int(n * 0.01), replace=False)
    df.loc[neg_margin_idx, "margin_pct"] = -abs(df.loc[neg_margin_idx, "margin_pct"])
    
    logger.debug(f"Injected product quality issues: "
                 f"{len(null_rating_idx)} null ratings, "
                 f"{len(neg_margin_idx)} negative margins")
    
    return df

