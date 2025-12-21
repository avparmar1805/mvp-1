# Bronze Layer Data Generation Plan

## Overview

This document details the strategy for generating synthetic datasets for the Bronze layer that will support all four use cases (A1, A2, B3, C2).

---

## 1. Dataset Specifications

### 1.1 orders

**Purpose**: Core transactional data for sales analytics and customer behavior

**Schema**:
```python
{
    "order_id": "VARCHAR(50)",          # Unique identifier (ORD-XXXXXXXX)
    "customer_id": "VARCHAR(50)",       # Foreign key to customers
    "product_id": "VARCHAR(50)",        # Foreign key to products
    "order_date": "TIMESTAMP",          # Order timestamp
    "quantity": "INTEGER",              # Number of units
    "unit_price": "DECIMAL(10,2)",      # Price per unit
    "total_amount": "DECIMAL(12,2)",    # quantity * unit_price
    "region": "VARCHAR(50)",            # Sales region
    "category": "VARCHAR(50)",          # Product category (denormalized)
    "status": "VARCHAR(20)",            # completed, cancelled, pending
    "discount_amount": "DECIMAL(10,2)", # Discount applied
    "tax_amount": "DECIMAL(10,2)",      # Tax charged
    "shipping_cost": "DECIMAL(10,2)"    # Shipping fee
}
```

**Row Count**: 100,000

**Date Range**: 2023-01-01 to 2025-11-15 (1,050 days)

**Distribution Characteristics**:
- **Temporal**: 
  - Weekday bias: 60% Mon-Fri, 40% Sat-Sun
  - Seasonal: +30% in Nov-Dec (holiday season)
  - Daily pattern: Peak at 12pm-2pm, 6pm-8pm
- **Regional**: 
  - North: 30%, South: 25%, East: 25%, West: 20%
- **Status**: 
  - completed: 85%, cancelled: 10%, pending: 5%
- **Revenue**: 
  - Log-normal distribution, mean=$125, std=$150
  - Range: $5 - $5,000

**Data Quality Issues** (intentional):
- 2% null values in `discount_amount`
- 1% duplicate `order_id` (to test deduplication)
- 0.5% negative `total_amount` (data entry errors)
- 3% mismatched `category` (doesn't match product table)

---

### 1.2 customers

**Purpose**: Customer master data for Customer 360 use case

**Schema**:
```python
{
    "customer_id": "VARCHAR(50)",       # Unique identifier (CUST-XXXXXXXX)
    "name": "VARCHAR(100)",             # Full name
    "email": "VARCHAR(100)",            # Email address (PII)
    "phone": "VARCHAR(20)",             # Phone number (PII)
    "signup_date": "DATE",              # Account creation date
    "loyalty_tier": "VARCHAR(20)",      # bronze, silver, gold, platinum
    "total_lifetime_value": "DECIMAL(12,2)", # Cumulative revenue
    "segment": "VARCHAR(50)",           # high_value, regular, at_risk, churned
    "address_city": "VARCHAR(100)",     # City
    "address_state": "VARCHAR(50)",     # State
    "address_country": "VARCHAR(50)",   # Country (default: USA)
    "date_of_birth": "DATE",            # DOB (PII)
    "gender": "VARCHAR(10)",            # Male, Female, Other, Prefer not to say
    "preferred_contact_method": "VARCHAR(20)" # email, phone, sms
}
```

**Row Count**: 10,000

**Date Range**: signup_date from 2020-01-01 to 2025-11-15

**Distribution Characteristics**:
- **Loyalty Tier**: 
  - bronze: 50%, silver: 30%, gold: 15%, platinum: 5%
- **Segment**: 
  - regular: 60%, high_value: 20%, at_risk: 15%, churned: 5%
- **Lifetime Value**: 
  - Correlated with loyalty tier
  - bronze: $50-$500, silver: $500-$2000, gold: $2000-$10000, platinum: $10000+

**Data Quality Issues**:
- 5% null values in `phone`
- 2% invalid email formats
- 1% duplicate emails (same person, multiple accounts)

---

### 1.3 products

**Purpose**: Product catalog for sales analytics and recommendations

**Schema**:
```python
{
    "product_id": "VARCHAR(50)",        # Unique identifier (PROD-XXXXXXXX)
    "product_name": "VARCHAR(200)",     # Product name
    "category": "VARCHAR(50)",          # Electronics, Clothing, Home, Books, Sports
    "subcategory": "VARCHAR(50)",       # Specific subcategory
    "brand": "VARCHAR(100)",            # Brand name
    "price": "DECIMAL(10,2)",           # Current selling price
    "cost": "DECIMAL(10,2)",            # Cost to company
    "margin_pct": "DECIMAL(5,2)",       # (price - cost) / price * 100
    "stock_quantity": "INTEGER",        # Available inventory
    "supplier_id": "VARCHAR(50)",       # Supplier reference
    "created_at": "DATE",               # Product launch date
    "is_active": "BOOLEAN",             # Active/discontinued
    "rating_avg": "DECIMAL(3,2)",       # Average rating (1-5)
    "review_count": "INTEGER"           # Number of reviews
}
```

**Row Count**: 1,000

**Distribution Characteristics**:
- **Category**: 
  - Electronics: 25%, Clothing: 25%, Home: 20%, Books: 15%, Sports: 15%
- **Price Range**: 
  - Electronics: $50-$2000
  - Clothing: $20-$300
  - Home: $15-$500
  - Books: $10-$50
  - Sports: $25-$800
- **Margin**: 
  - 20-50% depending on category
- **Active Status**: 
  - 90% active, 10% discontinued

**Data Quality Issues**:
- 3% null values in `rating_avg` (new products)
- 1% negative `margin_pct` (pricing errors)

---

### 1.4 marketing_campaigns

**Purpose**: Campaign master data for marketing performance analytics

**Schema**:
```python
{
    "campaign_id": "VARCHAR(50)",       # Unique identifier (CAMP-XXXXXXXX)
    "campaign_name": "VARCHAR(200)",    # Descriptive name
    "channel": "VARCHAR(50)",           # email, social, search, display, video
    "start_date": "DATE",               # Campaign start
    "end_date": "DATE",                 # Campaign end
    "budget": "DECIMAL(12,2)",          # Total budget allocated
    "target_audience": "VARCHAR(100)",  # Audience segment
    "objective": "VARCHAR(50)",         # awareness, consideration, conversion
    "status": "VARCHAR(20)",            # active, completed, paused
    "owner": "VARCHAR(100)"             # Campaign manager
}
```

**Row Count**: 50

**Date Range**: 2024-01-01 to 2025-11-15

**Distribution Characteristics**:
- **Channel**: 
  - email: 30%, social: 25%, search: 20%, display: 15%, video: 10%
- **Budget**: 
  - $5,000 - $100,000 per campaign
- **Duration**: 
  - 7-90 days

---

### 1.5 marketing_events

**Purpose**: Event-level data for marketing performance (impressions, clicks, conversions)

**Schema**:
```python
{
    "event_id": "VARCHAR(50)",          # Unique identifier (EVT-XXXXXXXX)
    "campaign_id": "VARCHAR(50)",       # Foreign key to campaigns
    "event_date": "TIMESTAMP",          # Event timestamp
    "event_type": "VARCHAR(20)",        # impression, click, conversion
    "user_id": "VARCHAR(50)",           # User/visitor ID (may not match customer_id)
    "cost": "DECIMAL(10,4)",            # Cost per event (CPC, CPM)
    "revenue": "DECIMAL(10,2)",         # Revenue if conversion
    "device_type": "VARCHAR(20)",       # desktop, mobile, tablet
    "browser": "VARCHAR(50)",           # Chrome, Safari, Firefox, etc.
    "location": "VARCHAR(100)"          # Geographic location
}
```

**Row Count**: 500,000

**Distribution Characteristics**:
- **Event Type**: 
  - impression: 80%, click: 15%, conversion: 5%
- **Funnel Conversion**: 
  - Impressions → Clicks: ~18% CTR
  - Clicks → Conversions: ~33% CVR
- **Cost**: 
  - Impression: $0.01 - $0.05 (CPM basis)
  - Click: $0.50 - $5.00 (CPC)
  - Conversion: $0 (no direct cost)
- **Revenue** (for conversions): 
  - $10 - $500 per conversion

**Data Quality Issues**:
- 1% events with `campaign_id` not in campaigns table
- 2% null values in `location`

---

### 1.6 user_interactions

**Purpose**: User-product interaction signals for recommendation features

**Schema**:
```python
{
    "interaction_id": "VARCHAR(50)",    # Unique identifier (INT-XXXXXXXX)
    "user_id": "VARCHAR(50)",           # User identifier (may map to customer_id)
    "product_id": "VARCHAR(50)",        # Foreign key to products
    "interaction_type": "VARCHAR(20)",  # view, cart, purchase, rating, wishlist
    "timestamp": "TIMESTAMP",           # Interaction time
    "rating": "INTEGER",                # 1-5 (only for rating type)
    "session_id": "VARCHAR(50)",        # Session identifier
    "duration_seconds": "INTEGER",      # Time spent (for views)
    "referrer": "VARCHAR(100)",         # Traffic source
    "device_type": "VARCHAR(20)"        # desktop, mobile, tablet
}
```

**Row Count**: 200,000

**Distribution Characteristics**:
- **Interaction Type**: 
  - view: 70%, cart: 15%, purchase: 8%, rating: 5%, wishlist: 2%
- **Funnel Conversion**: 
  - Views → Cart: ~21%
  - Cart → Purchase: ~53%
- **Rating Distribution**: 
  - 5 stars: 40%, 4 stars: 30%, 3 stars: 15%, 2 stars: 10%, 1 star: 5%
- **Session Duration**: 
  - 10 seconds - 30 minutes

**Data Quality Issues**:
- 5% interactions with `product_id` not in products table (deleted products)
- 3% null values in `session_id`

---

### 1.7 support_tickets

**Purpose**: Customer support data for Customer 360 use case

**Schema**:
```python
{
    "ticket_id": "VARCHAR(50)",         # Unique identifier (TKT-XXXXXXXX)
    "customer_id": "VARCHAR(50)",       # Foreign key to customers
    "created_at": "TIMESTAMP",          # Ticket creation time
    "resolved_at": "TIMESTAMP",         # Resolution time (null if open)
    "category": "VARCHAR(50)",          # product_issue, billing, shipping, account
    "priority": "VARCHAR(20)",          # low, medium, high, critical
    "status": "VARCHAR(20)",            # open, in_progress, resolved, closed
    "satisfaction_score": "INTEGER",    # 1-5 (null if not resolved)
    "resolution_time_hours": "DECIMAL(10,2)", # Time to resolve
    "agent_id": "VARCHAR(50)",          # Support agent
    "channel": "VARCHAR(20)"            # phone, email, chat, self_service
}
```

**Row Count**: 5,000

**Distribution Characteristics**:
- **Category**: 
  - product_issue: 40%, billing: 25%, shipping: 20%, account: 15%
- **Priority**: 
  - low: 40%, medium: 35%, high: 20%, critical: 5%
- **Status**: 
  - closed: 70%, resolved: 15%, in_progress: 10%, open: 5%
- **Satisfaction**: 
  - 5: 30%, 4: 35%, 3: 20%, 2: 10%, 1: 5%
- **Resolution Time**: 
  - low: 24-72 hours
  - medium: 12-48 hours
  - high: 4-24 hours
  - critical: 1-8 hours

---

## 2. Data Generation Strategy

### 2.1 Technology Stack

**Libraries**:
- **Faker**: Realistic names, emails, addresses, dates
- **NumPy**: Statistical distributions, random sampling
- **Pandas**: DataFrame manipulation
- **Polars**: High-performance alternative to Pandas
- **Mimesis**: Additional fake data generation
- **PyArrow**: Parquet file writing

### 2.2 Generation Approach

#### Step 1: Generate Master Data (Customers, Products)

```python
from faker import Faker
import numpy as np
import pandas as pd

fake = Faker()

def generate_customers(n=10000):
    """Generate customer master data"""
    customers = []
    
    for i in range(n):
        signup_date = fake.date_between(start_date='-5y', end_date='today')
        lifetime_value = np.random.lognormal(mean=5, sigma=1.5)
        
        # Determine loyalty tier based on LTV
        if lifetime_value > 10000:
            tier = 'platinum'
        elif lifetime_value > 2000:
            tier = 'gold'
        elif lifetime_value > 500:
            tier = 'silver'
        else:
            tier = 'bronze'
        
        customers.append({
            'customer_id': f'CUST-{i:08d}',
            'name': fake.name(),
            'email': fake.email(),
            'phone': fake.phone_number(),
            'signup_date': signup_date,
            'loyalty_tier': tier,
            'total_lifetime_value': round(lifetime_value, 2),
            'segment': np.random.choice(
                ['regular', 'high_value', 'at_risk', 'churned'],
                p=[0.6, 0.2, 0.15, 0.05]
            ),
            'address_city': fake.city(),
            'address_state': fake.state(),
            'address_country': 'USA',
            'date_of_birth': fake.date_of_birth(minimum_age=18, maximum_age=80),
            'gender': np.random.choice(
                ['Male', 'Female', 'Other', 'Prefer not to say'],
                p=[0.48, 0.48, 0.02, 0.02]
            ),
            'preferred_contact_method': np.random.choice(
                ['email', 'phone', 'sms'],
                p=[0.6, 0.3, 0.1]
            )
        })
    
    return pd.DataFrame(customers)
```

#### Step 2: Generate Transactional Data (Orders, Events, Interactions)

```python
def generate_orders(customers_df, products_df, n=100000):
    """Generate order transactions with temporal patterns"""
    orders = []
    
    # Create date range with seasonal patterns
    start_date = pd.Timestamp('2023-01-01')
    end_date = pd.Timestamp('2025-11-15')
    
    for i in range(n):
        # Temporal distribution (more orders on weekdays, holiday season)
        date = fake.date_time_between(start_date=start_date, end_date=end_date)
        
        # Apply seasonal boost
        month = date.month
        seasonal_factor = 1.3 if month in [11, 12] else 1.0
        
        # Select customer (power law: 20% customers generate 80% orders)
        customer_id = np.random.choice(
            customers_df['customer_id'],
            p=generate_power_law_weights(len(customers_df))
        )
        
        # Select product
        product = products_df.sample(1).iloc[0]
        
        quantity = np.random.randint(1, 10)
        unit_price = product['price']
        total_amount = quantity * unit_price
        
        # Apply discount (20% of orders)
        discount = 0
        if np.random.random() < 0.2:
            discount = total_amount * np.random.uniform(0.05, 0.3)
        
        orders.append({
            'order_id': f'ORD-{i:08d}',
            'customer_id': customer_id,
            'product_id': product['product_id'],
            'order_date': date,
            'quantity': quantity,
            'unit_price': unit_price,
            'total_amount': round(total_amount - discount, 2),
            'region': np.random.choice(
                ['North', 'South', 'East', 'West'],
                p=[0.3, 0.25, 0.25, 0.2]
            ),
            'category': product['category'],
            'status': np.random.choice(
                ['completed', 'cancelled', 'pending'],
                p=[0.85, 0.10, 0.05]
            ),
            'discount_amount': round(discount, 2),
            'tax_amount': round(total_amount * 0.08, 2),
            'shipping_cost': round(np.random.uniform(5, 25), 2)
        })
    
    return pd.DataFrame(orders)

def generate_power_law_weights(n, alpha=1.5):
    """Generate power law distribution for customer order frequency"""
    weights = np.array([1.0 / (i + 1) ** alpha for i in range(n)])
    return weights / weights.sum()
```

#### Step 3: Inject Data Quality Issues

```python
def inject_quality_issues(df, config):
    """Inject intentional data quality issues"""
    df_copy = df.copy()
    
    for column, issue_config in config.items():
        if issue_config['type'] == 'null':
            null_pct = issue_config['percentage']
            null_indices = np.random.choice(
                df_copy.index,
                size=int(len(df_copy) * null_pct),
                replace=False
            )
            df_copy.loc[null_indices, column] = None
        
        elif issue_config['type'] == 'duplicate':
            dup_pct = issue_config['percentage']
            dup_count = int(len(df_copy) * dup_pct)
            dup_indices = np.random.choice(df_copy.index, size=dup_count)
            df_copy = pd.concat([df_copy, df_copy.loc[dup_indices]])
        
        elif issue_config['type'] == 'invalid':
            invalid_pct = issue_config['percentage']
            invalid_indices = np.random.choice(
                df_copy.index,
                size=int(len(df_copy) * invalid_pct),
                replace=False
            )
            df_copy.loc[invalid_indices, column] = issue_config['invalid_value']
    
    return df_copy

# Example usage
quality_config = {
    'discount_amount': {'type': 'null', 'percentage': 0.02},
    'order_id': {'type': 'duplicate', 'percentage': 0.01},
    'total_amount': {'type': 'invalid', 'percentage': 0.005, 'invalid_value': -100}
}

orders_with_issues = inject_quality_issues(orders_df, quality_config)
```

#### Step 4: Maintain Referential Integrity

```python
def ensure_referential_integrity(orders_df, customers_df, products_df):
    """Ensure foreign keys are valid"""
    
    # Remove orders with invalid customer_id
    valid_customers = set(customers_df['customer_id'])
    orders_df = orders_df[orders_df['customer_id'].isin(valid_customers)]
    
    # Remove orders with invalid product_id
    valid_products = set(products_df['product_id'])
    orders_df = orders_df[orders_df['product_id'].isin(valid_products)]
    
    # Intentionally leave 3% with mismatched category (for testing)
    mismatch_indices = np.random.choice(
        orders_df.index,
        size=int(len(orders_df) * 0.03),
        replace=False
    )
    orders_df.loc[mismatch_indices, 'category'] = 'MISMATCH'
    
    return orders_df
```

#### Step 5: Export to Parquet

```python
def export_to_parquet(df, output_path, metadata):
    """Export DataFrame to Parquet with metadata"""
    
    # Write Parquet file
    df.to_parquet(
        output_path,
        engine='pyarrow',
        compression='snappy',
        index=False
    )
    
    # Write metadata JSON
    metadata_path = output_path.replace('.parquet', '_metadata.json')
    
    metadata_dict = {
        'dataset_name': metadata['name'],
        'created_at': pd.Timestamp.now().isoformat(),
        'row_count': len(df),
        'size_bytes': os.path.getsize(output_path),
        'schema': [
            {
                'name': col,
                'type': str(df[col].dtype),
                'nullable': df[col].isnull().any()
            }
            for col in df.columns
        ],
        'statistics': {
            col: {
                'min': float(df[col].min()) if pd.api.types.is_numeric_dtype(df[col]) else None,
                'max': float(df[col].max()) if pd.api.types.is_numeric_dtype(df[col]) else None,
                'mean': float(df[col].mean()) if pd.api.types.is_numeric_dtype(df[col]) else None,
                'null_count': int(df[col].isnull().sum())
            }
            for col in df.columns
        }
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata_dict, f, indent=2)
```

---

## 3. Data Relationships

### 3.1 Entity Relationship Diagram

```
┌─────────────┐         ┌─────────────┐
│  customers  │         │  products   │
│             │         │             │
│ customer_id │         │ product_id  │
│ name        │         │ product_name│
│ email       │         │ category    │
│ loyalty_tier│         │ price       │
└──────┬──────┘         └──────┬──────┘
       │                       │
       │                       │
       ├───────────┬───────────┤
       │           │           │
┌──────▼───────────▼──────┐    │
│       orders            │    │
│                         │    │
│ order_id                │    │
│ customer_id (FK)        │    │
│ product_id (FK)         │    │
│ order_date              │    │
│ total_amount            │    │
│ region                  │    │
└──────┬──────────────────┘    │
       │                       │
       │                       │
┌──────▼──────────┐     ┌──────▼──────────┐
│ support_tickets │     │ user_interactions│
│                 │     │                  │
│ ticket_id       │     │ interaction_id   │
│ customer_id (FK)│     │ user_id          │
│ category        │     │ product_id (FK)  │
│ status          │     │ interaction_type │
└─────────────────┘     └──────────────────┘

┌──────────────────────┐     ┌──────────────────┐
│ marketing_campaigns  │     │ marketing_events │
│                      │     │                  │
│ campaign_id          │◄────┤ event_id         │
│ campaign_name        │     │ campaign_id (FK) │
│ channel              │     │ event_type       │
│ budget               │     │ cost             │
└──────────────────────┘     └──────────────────┘
```

### 3.2 Cardinality

- **customers → orders**: 1:N (one customer, many orders)
- **products → orders**: 1:N (one product, many orders)
- **customers → support_tickets**: 1:N
- **products → user_interactions**: 1:N
- **marketing_campaigns → marketing_events**: 1:N

---

## 4. Validation Checks

### 4.1 Post-Generation Validation

```python
def validate_dataset(df, validation_rules):
    """Run validation checks on generated dataset"""
    results = []
    
    for rule in validation_rules:
        if rule['type'] == 'row_count':
            actual = len(df)
            expected = rule['expected']
            passed = abs(actual - expected) / expected < 0.05  # 5% tolerance
            results.append({
                'rule': f"Row count ~{expected}",
                'passed': passed,
                'actual': actual
            })
        
        elif rule['type'] == 'null_percentage':
            column = rule['column']
            max_null_pct = rule['max_percentage']
            actual_null_pct = df[column].isnull().sum() / len(df)
            passed = actual_null_pct <= max_null_pct
            results.append({
                'rule': f"{column} null % <= {max_null_pct}",
                'passed': passed,
                'actual': actual_null_pct
            })
        
        elif rule['type'] == 'foreign_key':
            fk_column = rule['fk_column']
            ref_values = rule['ref_values']
            invalid_count = (~df[fk_column].isin(ref_values)).sum()
            passed = invalid_count <= rule['max_invalid']
            results.append({
                'rule': f"{fk_column} foreign key integrity",
                'passed': passed,
                'invalid_count': invalid_count
            })
    
    return results

# Example validation
validation_rules = [
    {'type': 'row_count', 'expected': 100000},
    {'type': 'null_percentage', 'column': 'order_id', 'max_percentage': 0.0},
    {'type': 'null_percentage', 'column': 'discount_amount', 'max_percentage': 0.03},
    {'type': 'foreign_key', 'fk_column': 'customer_id', 'ref_values': customers_df['customer_id'], 'max_invalid': 0}
]

validation_results = validate_dataset(orders_df, validation_rules)
```

### 4.2 Cross-Dataset Validation

```python
def validate_cross_dataset(datasets):
    """Validate relationships across datasets"""
    
    # Check customer LTV matches order totals
    customer_order_totals = (
        datasets['orders']
        .groupby('customer_id')['total_amount']
        .sum()
        .reset_index()
    )
    
    customers_with_totals = datasets['customers'].merge(
        customer_order_totals,
        on='customer_id',
        how='left'
    )
    
    # LTV should be >= order totals (may include other revenue)
    ltv_mismatch = (
        customers_with_totals['total_lifetime_value'] < 
        customers_with_totals['total_amount'].fillna(0)
    ).sum()
    
    print(f"Customers with LTV < order totals: {ltv_mismatch}")
    
    # Check product categories match between products and orders
    product_categories = set(datasets['products']['category'])
    order_categories = set(datasets['orders']['category'])
    
    unmatched = order_categories - product_categories
    print(f"Unmatched categories in orders: {unmatched}")
```

---

## 5. Generation Script Structure

### 5.1 Main Script

```python
# data_generation/generate_bronze.py

import argparse
from generators import (
    generate_customers,
    generate_products,
    generate_orders,
    generate_campaigns,
    generate_events,
    generate_interactions,
    generate_tickets
)
from utils import export_to_parquet, validate_dataset

def main(output_dir, seed=42):
    """Generate all bronze datasets"""
    
    np.random.seed(seed)
    Faker.seed(seed)
    
    print("Generating customers...")
    customers = generate_customers(n=10000)
    export_to_parquet(customers, f"{output_dir}/customers/customers.parquet", 
                      metadata={'name': 'bronze.customers'})
    
    print("Generating products...")
    products = generate_products(n=1000)
    export_to_parquet(products, f"{output_dir}/products/products.parquet",
                      metadata={'name': 'bronze.products'})
    
    print("Generating orders...")
    orders = generate_orders(customers, products, n=100000)
    export_to_parquet(orders, f"{output_dir}/orders/orders.parquet",
                      metadata={'name': 'bronze.orders'})
    
    print("Generating marketing campaigns...")
    campaigns = generate_campaigns(n=50)
    export_to_parquet(campaigns, f"{output_dir}/marketing_campaigns/campaigns.parquet",
                      metadata={'name': 'bronze.marketing_campaigns'})
    
    print("Generating marketing events...")
    events = generate_events(campaigns, n=500000)
    export_to_parquet(events, f"{output_dir}/marketing_events/events.parquet",
                      metadata={'name': 'bronze.marketing_events'})
    
    print("Generating user interactions...")
    interactions = generate_interactions(customers, products, n=200000)
    export_to_parquet(interactions, f"{output_dir}/user_interactions/interactions.parquet",
                      metadata={'name': 'bronze.user_interactions'})
    
    print("Generating support tickets...")
    tickets = generate_tickets(customers, n=5000)
    export_to_parquet(tickets, f"{output_dir}/support_tickets/tickets.parquet",
                      metadata={'name': 'bronze.support_tickets'})
    
    print("Bronze layer generation complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='data/bronze')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    main(args.output_dir, args.seed)
```

### 5.2 Directory Structure

```
data_generation/
├── generate_bronze.py          # Main script
├── generators/
│   ├── __init__.py
│   ├── customers.py
│   ├── products.py
│   ├── orders.py
│   ├── campaigns.py
│   ├── events.py
│   ├── interactions.py
│   └── tickets.py
├── utils/
│   ├── __init__.py
│   ├── export.py
│   ├── validation.py
│   └── quality_injection.py
├── schemas/
│   ├── customers_schema.yaml
│   ├── products_schema.yaml
│   └── ...
└── tests/
    ├── test_generators.py
    └── test_validation.py
```

---

## 6. Use Case Coverage

### 6.1 A1 — Daily Sales Analytics

**Required Datasets**: ✅ orders, ✅ products

**Key Columns**:
- orders: `order_date`, `total_amount`, `quantity`, `region`, `status`
- products: `category`

### 6.2 A2 — Marketing Campaign Performance

**Required Datasets**: ✅ marketing_campaigns, ✅ marketing_events

**Key Columns**:
- campaigns: `campaign_id`, `campaign_name`, `channel`, `budget`
- events: `event_type`, `cost`, `revenue`, `event_date`

### 6.3 B3 — Product Recommendation Features

**Required Datasets**: ✅ user_interactions, ✅ products, ✅ orders

**Key Columns**:
- interactions: `user_id`, `product_id`, `interaction_type`, `rating`, `timestamp`
- products: `category`
- orders: `customer_id`, `product_id` (for purchase history)

### 6.4 C2 — Customer 360

**Required Datasets**: ✅ customers, ✅ orders, ✅ support_tickets

**Key Columns**:
- customers: `customer_id`, `name`, `email`, `loyalty_tier`, `segment`
- orders: `customer_id`, `total_amount`, `order_date`
- tickets: `customer_id`, `status`, `satisfaction_score`

---

## 7. Execution Plan

### Phase 1: Setup (Day 1)
- [ ] Install dependencies (Faker, NumPy, Pandas, PyArrow)
- [ ] Create directory structure
- [ ] Define schema YAML files

### Phase 2: Master Data Generation (Day 2)
- [ ] Implement `generate_customers()`
- [ ] Implement `generate_products()`
- [ ] Test and validate

### Phase 3: Transactional Data Generation (Days 3-4)
- [ ] Implement `generate_orders()` with temporal patterns
- [ ] Implement `generate_campaigns()` and `generate_events()`
- [ ] Implement `generate_interactions()`
- [ ] Implement `generate_tickets()`

### Phase 4: Quality Issues & Validation (Day 5)
- [ ] Implement `inject_quality_issues()`
- [ ] Implement validation functions
- [ ] Run cross-dataset validation

### Phase 5: Export & Documentation (Day 6)
- [ ] Export all datasets to Parquet
- [ ] Generate metadata JSON files
- [ ] Create data dictionary
- [ ] Document data quality issues

---

## 8. Expected Output

After running the generation script:

```
data/bronze/
├── customers/
│   ├── customers.parquet (10,000 rows, ~2 MB)
│   └── customers_metadata.json
├── products/
│   ├── products.parquet (1,000 rows, ~200 KB)
│   └── products_metadata.json
├── orders/
│   ├── orders.parquet (100,000 rows, ~15 MB)
│   └── orders_metadata.json
├── marketing_campaigns/
│   ├── campaigns.parquet (50 rows, ~10 KB)
│   └── campaigns_metadata.json
├── marketing_events/
│   ├── events.parquet (500,000 rows, ~50 MB)
│   └── events_metadata.json
├── user_interactions/
│   ├── interactions.parquet (200,000 rows, ~25 MB)
│   └── interactions_metadata.json
└── support_tickets/
    ├── tickets.parquet (5,000 rows, ~1 MB)
    └── tickets_metadata.json

Total: ~93 MB
```

---

## Summary

This plan provides:

1. **Realistic Data**: Using Faker and statistical distributions
2. **Temporal Patterns**: Seasonality, weekday/weekend effects
3. **Referential Integrity**: Foreign keys maintained (with intentional exceptions)
4. **Quality Issues**: Controlled injection of nulls, duplicates, errors
5. **Use Case Coverage**: All four use cases supported
6. **Scalability**: Can easily adjust row counts
7. **Reproducibility**: Seed-based generation for consistent results

**Next Step**: Implement the generation scripts when ready to build.

