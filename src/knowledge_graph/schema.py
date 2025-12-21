"""
Knowledge Graph Schema Definitions

Defines the node types, relationship types, and their properties
for the enterprise data catalog knowledge graph.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class NodeType(str, Enum):
    """Types of nodes in the knowledge graph."""
    DATASET = "Dataset"
    COLUMN = "Column"
    BUSINESS_TERM = "BusinessTerm"
    DOMAIN = "Domain"
    QUALITY_RULE = "QualityRule"
    DATA_PRODUCT = "DataProduct"


class RelationshipType(str, Enum):
    """Types of relationships in the knowledge graph."""
    HAS_COLUMN = "HAS_COLUMN"
    MAPS_TO = "MAPS_TO"
    BELONGS_TO_DOMAIN = "BELONGS_TO_DOMAIN"
    DERIVED_FROM = "DERIVED_FROM"
    REFERENCES = "REFERENCES"
    HAS_QUALITY_RULE = "HAS_QUALITY_RULE"
    SIMILAR_TO = "SIMILAR_TO"
    USES_DATASET = "USES_DATASET"


@dataclass
class DatasetNode:
    """Represents a dataset in the knowledge graph."""
    name: str
    description: str = ""
    layer: str = "bronze"  # bronze, silver, gold
    format: str = "parquet"
    path: str = ""
    row_count: int = 0
    size_bytes: int = 0
    created_at: str = ""
    owner: str = ""
    tags: List[str] = field(default_factory=list)
    quality_score: float = 1.0
    freshness: str = "daily"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "layer": self.layer,
            "format": self.format,
            "path": self.path,
            "row_count": self.row_count,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
            "owner": self.owner,
            "tags": self.tags,
            "quality_score": self.quality_score,
            "freshness": self.freshness,
        }


@dataclass
class ColumnNode:
    """Represents a column in the knowledge graph."""
    name: str
    dataset_name: str
    data_type: str
    description: str = ""
    nullable: bool = True
    is_primary_key: bool = False
    is_foreign_key: bool = False
    references: Optional[str] = None  # "dataset.column" format
    null_count: int = 0
    unique_count: int = 0
    sample_values: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    @property
    def full_name(self) -> str:
        return f"{self.dataset_name}.{self.name}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dataset_name": self.dataset_name,
            "full_name": self.full_name,
            "data_type": self.data_type,
            "description": self.description,
            "nullable": self.nullable,
            "is_primary_key": self.is_primary_key,
            "is_foreign_key": self.is_foreign_key,
            "references": self.references,
            "null_count": self.null_count,
            "unique_count": self.unique_count,
            "sample_values": self.sample_values,
            "statistics": self.statistics,
        }


@dataclass
class BusinessTermNode:
    """Represents a business term/concept in the knowledge graph."""
    term: str
    definition: str = ""
    synonyms: List[str] = field(default_factory=list)
    domain: str = ""
    examples: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "term": self.term,
            "definition": self.definition,
            "synonyms": self.synonyms,
            "domain": self.domain,
            "examples": self.examples,
        }


@dataclass
class DomainNode:
    """Represents a business domain in the knowledge graph."""
    name: str
    description: str = ""
    owner: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "owner": self.owner,
        }


@dataclass
class QualityRuleNode:
    """Represents a data quality rule in the knowledge graph."""
    name: str
    rule_type: str  # completeness, uniqueness, validity, consistency
    condition: str
    severity: str = "error"  # error, warning, info
    threshold: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "rule_type": self.rule_type,
            "condition": self.condition,
            "severity": self.severity,
            "threshold": self.threshold,
        }


# Business terms catalog for the e-commerce domain
BUSINESS_TERMS_CATALOG = [
    BusinessTermNode(
        term="revenue",
        definition="Total monetary value of sales transactions",
        synonyms=["sales", "income", "earnings", "total_amount"],
        domain="Sales",
        examples=["total_revenue", "order_amount", "transaction_value"],
    ),
    BusinessTermNode(
        term="order",
        definition="A customer purchase transaction",
        synonyms=["transaction", "purchase", "sale"],
        domain="Sales",
        examples=["order_id", "order_date", "order_status"],
    ),
    BusinessTermNode(
        term="customer",
        definition="A person or entity that purchases products",
        synonyms=["buyer", "client", "consumer", "user"],
        domain="Customer",
        examples=["customer_id", "customer_name", "buyer_id"],
    ),
    BusinessTermNode(
        term="product",
        definition="An item available for sale",
        synonyms=["item", "merchandise", "goods", "sku"],
        domain="Product",
        examples=["product_id", "product_name", "item_id"],
    ),
    BusinessTermNode(
        term="category",
        definition="A classification group for products",
        synonyms=["type", "class", "segment", "group"],
        domain="Product",
        examples=["product_category", "item_type"],
    ),
    BusinessTermNode(
        term="region",
        definition="A geographic area for business operations",
        synonyms=["area", "territory", "zone", "location"],
        domain="Geography",
        examples=["sales_region", "customer_region"],
    ),
    BusinessTermNode(
        term="campaign",
        definition="A marketing initiative to promote products",
        synonyms=["promotion", "advertisement", "marketing_effort"],
        domain="Marketing",
        examples=["campaign_id", "campaign_name"],
    ),
    BusinessTermNode(
        term="CTR",
        definition="Click-through rate - ratio of clicks to impressions",
        synonyms=["click_rate", "click_through_rate"],
        domain="Marketing",
        examples=["ctr", "click_rate"],
    ),
    BusinessTermNode(
        term="CVR",
        definition="Conversion rate - ratio of conversions to clicks",
        synonyms=["conversion_rate", "conv_rate"],
        domain="Marketing",
        examples=["cvr", "conversion_rate"],
    ),
    BusinessTermNode(
        term="CPA",
        definition="Cost per acquisition - cost to acquire a customer",
        synonyms=["cost_per_acquisition", "acquisition_cost"],
        domain="Marketing",
        examples=["cpa", "cost_per_conversion"],
    ),
    BusinessTermNode(
        term="ROAS",
        definition="Return on ad spend - revenue generated per dollar spent",
        synonyms=["return_on_ad_spend", "ad_roi"],
        domain="Marketing",
        examples=["roas", "ad_return"],
    ),
    BusinessTermNode(
        term="impression",
        definition="A single display of an advertisement",
        synonyms=["view", "display", "ad_view"],
        domain="Marketing",
        examples=["impressions", "ad_impressions"],
    ),
    BusinessTermNode(
        term="conversion",
        definition="A completed desired action (e.g., purchase)",
        synonyms=["purchase", "sale", "completed_action"],
        domain="Marketing",
        examples=["conversions", "completed_purchases"],
    ),
    BusinessTermNode(
        term="interaction",
        definition="User engagement with a product or content",
        synonyms=["engagement", "activity", "action"],
        domain="Customer",
        examples=["user_interaction", "product_view"],
    ),
    BusinessTermNode(
        term="rating",
        definition="Customer evaluation score for a product",
        synonyms=["score", "review_score", "stars"],
        domain="Product",
        examples=["rating_avg", "product_rating"],
    ),
    BusinessTermNode(
        term="loyalty_tier",
        definition="Customer segmentation based on engagement/spending",
        synonyms=["membership_level", "customer_tier", "loyalty_status"],
        domain="Customer",
        examples=["loyalty_tier", "membership_tier"],
    ),
    BusinessTermNode(
        term="lifetime_value",
        definition="Total revenue expected from a customer relationship",
        synonyms=["LTV", "CLV", "customer_value"],
        domain="Customer",
        examples=["total_lifetime_value", "customer_ltv"],
    ),
    BusinessTermNode(
        term="support_ticket",
        definition="A customer service request or issue report",
        synonyms=["ticket", "case", "issue", "request"],
        domain="Support",
        examples=["ticket_id", "support_case"],
    ),
    BusinessTermNode(
        term="satisfaction_score",
        definition="Customer satisfaction rating for service",
        synonyms=["CSAT", "satisfaction", "happiness_score"],
        domain="Support",
        examples=["satisfaction_score", "csat_score"],
    ),
    BusinessTermNode(
        term="resolution_time",
        definition="Time taken to resolve a support issue",
        synonyms=["response_time", "time_to_resolution", "handle_time"],
        domain="Support",
        examples=["resolution_time_hours", "avg_resolution_time"],
    ),
]


# Column to business term mappings
COLUMN_TERM_MAPPINGS = {
    # Orders
    "orders.total_amount": ["revenue", "order"],
    "orders.order_id": ["order"],
    "orders.order_date": ["order"],
    "orders.customer_id": ["customer", "order"],
    "orders.product_id": ["product", "order"],
    "orders.region": ["region"],
    "orders.category": ["category", "product"],
    "orders.quantity": ["order"],
    "orders.status": ["order"],
    
    # Customers
    "customers.customer_id": ["customer"],
    "customers.name": ["customer"],
    "customers.email": ["customer"],
    "customers.loyalty_tier": ["loyalty_tier", "customer"],
    "customers.total_lifetime_value": ["lifetime_value", "customer", "revenue"],
    "customers.segment": ["customer"],
    
    # Products
    "products.product_id": ["product"],
    "products.product_name": ["product"],
    "products.category": ["category", "product"],
    "products.price": ["product", "revenue"],
    "products.rating_avg": ["rating", "product"],
    
    # Marketing Campaigns
    "marketing_campaigns.campaign_id": ["campaign"],
    "marketing_campaigns.campaign_name": ["campaign"],
    "marketing_campaigns.budget": ["campaign"],
    "marketing_campaigns.channel": ["campaign"],
    
    # Marketing Events
    "marketing_events.event_type": ["impression", "conversion", "campaign"],
    "marketing_events.cost": ["campaign", "CPA"],
    "marketing_events.revenue": ["revenue", "ROAS", "conversion"],
    "marketing_events.campaign_id": ["campaign"],
    
    # User Interactions
    "user_interactions.interaction_type": ["interaction"],
    "user_interactions.rating": ["rating"],
    "user_interactions.user_id": ["customer"],
    "user_interactions.product_id": ["product"],
    
    # Support Tickets
    "support_tickets.ticket_id": ["support_ticket"],
    "support_tickets.customer_id": ["customer", "support_ticket"],
    "support_tickets.satisfaction_score": ["satisfaction_score"],
    "support_tickets.resolution_time_hours": ["resolution_time"],
    "support_tickets.category": ["support_ticket"],
    "support_tickets.status": ["support_ticket"],
}

