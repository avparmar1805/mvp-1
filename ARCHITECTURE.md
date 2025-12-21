# System Architecture - Agentic Data Product Builder

## 1. Architecture Overview

### 1.1 Layered Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PRESENTATION LAYER                            │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │  Streamlit UI    │  │   REST API       │  │   CLI Tool       │  │
│  │  (Interactive)   │  │   (FastAPI)      │  │   (Batch Mode)   │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────┐
│                      ORCHESTRATION LAYER                             │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    Orchestrator Agent                          │  │
│  │  - Workflow Management                                         │  │
│  │  - State Management (LangGraph StateGraph)                     │  │
│  │  - Error Handling & Retry Logic                                │  │
│  │  - Agent Coordination                                          │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────┐
│                        AGENT LAYER                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ Intent   │  │Discovery │  │ Modeling │  │Transform │            │
│  │ Agent    │  │ Agent    │  │ Agent    │  │ Agent    │            │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │
│  ┌──────────┐  ┌──────────┐                                         │
│  │ Quality  │  │Packaging │                                         │
│  │ Agent    │  │ Agent    │                                         │
│  └──────────┘  └──────────┘                                         │
└─────────────────────────────────────────────────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
┌───────▼────────┐    ┌──────────▼──────────┐    ┌────────▼─────────┐
│   LLM LAYER    │    │  KNOWLEDGE GRAPH    │    │   DATA LAYER     │
│                │    │      LAYER          │    │                  │
│ ┌────────────┐ │    │  ┌──────────────┐  │    │ ┌──────────────┐ │
│ │ GPT-4/     │ │    │  │   Neo4j      │  │    │ │ Bronze Layer │ │
│ │ Claude     │ │    │  │   Database   │  │    │ │ (Parquet)    │ │
│ └────────────┘ │    │  └──────────────┘  │    │ └──────────────┘ │
│ ┌────────────┐ │    │  ┌──────────────┐  │    │ ┌──────────────┐ │
│ │ Embeddings │ │    │  │ Graph Query  │  │    │ │ DuckDB       │ │
│ │ (OpenAI)   │ │    │  │ Engine       │  │    │ │ (Analytics)  │ │
│ └────────────┘ │    │  └──────────────┘  │    │ └──────────────┘ │
└────────────────┘    └─────────────────────┘    └──────────────────┘
```

---

## 2. Component Design

### 2.1 Orchestrator Agent

**Responsibilities**:
- Manage the overall workflow state
- Route requests to appropriate agents
- Handle inter-agent communication
- Aggregate results from multiple agents
- Implement retry and error recovery logic

**Technology**: LangGraph StateGraph

**State Schema**:
```python
class DataProductState(TypedDict):
    # Input
    user_request: str
    
    # Intent Analysis
    business_metrics: List[str]
    aggregation_levels: List[str]
    temporal_granularity: str
    
    # Discovery
    candidate_datasets: List[Dict]
    selected_datasets: List[str]
    
    # Modeling
    target_schema: List[Dict]
    grain: str
    primary_keys: List[str]
    
    # Transformation
    transformation_code: str
    transformation_language: str
    
    # Quality
    quality_rules: List[Dict]
    sla_requirements: Dict
    
    # Packaging
    data_product_spec: Dict
    
    # Metadata
    execution_log: List[Dict]
    errors: List[str]
```

**Workflow Graph**:
```python
workflow = StateGraph(DataProductState)

# Add nodes
workflow.add_node("intent", intent_agent)
workflow.add_node("discovery", discovery_agent)
workflow.add_node("modeling", modeling_agent)
workflow.add_node("transformation", transformation_agent)
workflow.add_node("quality", quality_agent)
workflow.add_node("packaging", packaging_agent)

# Define edges
workflow.set_entry_point("intent")
workflow.add_edge("intent", "discovery")
workflow.add_edge("discovery", "modeling")
workflow.add_edge("modeling", "transformation")
workflow.add_edge("transformation", "quality")
workflow.add_edge("quality", "packaging")
workflow.add_edge("packaging", END)

# Conditional edges for error handling
workflow.add_conditional_edges(
    "discovery",
    should_retry_discovery,
    {
        "retry": "discovery",
        "continue": "modeling"
    }
)
```

---

### 2.2 Intent Agent

**Purpose**: Parse natural language request and extract structured business requirements

**Input**: User's natural language request

**Output**:
```python
{
    "business_metrics": ["revenue", "order_count", "units_sold"],
    "entities": ["region", "category"],
    "temporal_granularity": "daily",
    "filters": ["status = 'completed'"],
    "business_terms": ["total_revenue", "sales"],
    "confidence_score": 0.92
}
```

**LLM Prompt Template**:
```
You are a business analyst expert. Analyze the following request and extract:

1. Business metrics to calculate (e.g., revenue, count, average)
2. Dimensions/entities for grouping (e.g., region, category, date)
3. Time granularity (hourly, daily, weekly, monthly)
4. Any filters or conditions mentioned
5. Map business terms to technical concepts

Request: {user_request}

Respond in JSON format.
```

**Knowledge Graph Integration**:
- Query for business term definitions
- Find synonyms for ambiguous terms
- Retrieve domain-specific glossary

**Cypher Query Example**:
```cypher
MATCH (bt:BusinessTerm)
WHERE bt.term IN $extracted_terms 
   OR any(syn IN bt.synonyms WHERE syn IN $extracted_terms)
RETURN bt.term, bt.definition, bt.domain
```

---

### 2.3 Discovery Agent

**Purpose**: Find relevant datasets from the Knowledge Graph

**Input**: Business metrics and entities from Intent Agent

**Output**:
```python
{
    "candidate_datasets": [
        {
            "name": "bronze.orders",
            "relevance_score": 0.95,
            "columns": ["order_id", "total_amount", "region", "order_date"],
            "quality_score": 0.88,
            "freshness": "daily",
            "owner": "data_engineering_team"
        },
        {
            "name": "bronze.products",
            "relevance_score": 0.87,
            "columns": ["product_id", "category"],
            "quality_score": 0.92,
            "freshness": "weekly",
            "owner": "product_team"
        }
    ],
    "selected_datasets": ["bronze.orders", "bronze.products"]
}
```

**Algorithm**:
1. **Semantic Search**: Use embeddings to find columns matching business terms
2. **Graph Traversal**: Follow `MAPS_TO` relationships to find datasets
3. **Ranking**: Score datasets by:
   - Column coverage (% of required metrics available)
   - Data quality score
   - Freshness alignment with SLA
   - Popularity (usage frequency)
4. **Filtering**: Remove datasets with access restrictions

**Cypher Query Example**:
```cypher
MATCH (d:Dataset)-[:HAS_COLUMN]->(c:Column)-[:MAPS_TO]->(bt:BusinessTerm)
WHERE bt.term IN $business_metrics
WITH d, COUNT(DISTINCT c) AS column_coverage
MATCH (d)-[:HAS_QUALITY_SCORE]->(qs:QualityScore)
RETURN d.name, 
       column_coverage, 
       qs.score AS quality_score,
       d.freshness
ORDER BY column_coverage DESC, quality_score DESC
LIMIT 10
```

**Embedding-Based Search** (for fuzzy matching):
```python
def find_similar_columns(query_embedding, top_k=10):
    """Find columns with similar semantic meaning"""
    query = """
    MATCH (c:Column)
    WITH c, gds.similarity.cosine(c.embedding, $query_embedding) AS similarity
    WHERE similarity > 0.7
    RETURN c.name, c.dataset_name, similarity
    ORDER BY similarity DESC
    LIMIT $top_k
    """
    return graph.run(query, query_embedding=query_embedding, top_k=top_k)
```

---

### 2.4 Modeling Agent

**Purpose**: Design the target data model schema

**Input**: 
- Business requirements from Intent Agent
- Available datasets from Discovery Agent

**Output**:
```python
{
    "target_table": "gold.daily_sales_analytics",
    "grain": "Daily, by region and category",
    "schema": [
        {
            "name": "date",
            "type": "DATE",
            "description": "Transaction date",
            "nullable": False,
            "primary_key": True
        },
        {
            "name": "region",
            "type": "VARCHAR(50)",
            "description": "Sales region",
            "nullable": False,
            "primary_key": True
        },
        {
            "name": "total_revenue",
            "type": "DECIMAL(18,2)",
            "description": "Sum of order amounts",
            "nullable": False,
            "constraints": ["total_revenue >= 0"]
        }
    ],
    "partitioning": {
        "type": "range",
        "column": "date",
        "interval": "monthly"
    },
    "indexing": ["date", "region"]
}
```

**LLM Prompt Template**:
```
You are a data modeling expert. Design a target schema for the following data product:

Business Requirements:
- Metrics: {business_metrics}
- Dimensions: {dimensions}
- Granularity: {temporal_granularity}

Available Source Columns:
{source_columns}

Design a schema that:
1. Includes all required metrics and dimensions
2. Uses appropriate data types
3. Defines primary keys based on grain
4. Adds constraints for data quality
5. Suggests partitioning strategy for performance

Respond in JSON format.
```

**Validation Rules**:
- Primary key uniqueness matches grain
- All metrics have appropriate aggregation functions
- Data types are compatible with source columns
- Nullable constraints align with business rules

---

### 2.5 Transformation Agent

**Purpose**: Generate SQL/Python code to transform source data to target schema

**Input**: 
- Source datasets and columns
- Target schema from Modeling Agent
- Business logic from Intent Agent

**Output**:
```python
{
    "language": "SQL",
    "code": """
        SELECT
            DATE(o.order_date) AS date,
            o.region,
            p.category,
            SUM(o.total_amount) AS total_revenue,
            COUNT(DISTINCT o.order_id) AS order_count,
            SUM(o.quantity) AS units_sold
        FROM bronze.orders o
        JOIN bronze.products p ON o.product_id = p.product_id
        WHERE o.status = 'completed'
        GROUP BY DATE(o.order_date), o.region, p.category
        HAVING SUM(o.total_amount) > 0
    """,
    "dependencies": ["bronze.orders", "bronze.products"],
    "estimated_rows": 50000
}
```

**Code Generation Strategy**:

1. **Template-Based** (for simple aggregations):
```python
template = """
SELECT
    {group_by_columns},
    {aggregations}
FROM {source_table}
{joins}
WHERE {filters}
GROUP BY {group_by_columns}
"""
```

2. **LLM-Based** (for complex logic):
```
You are a SQL expert. Generate a SQL query to:

Source Tables:
- bronze.orders (order_id, customer_id, total_amount, order_date, region, status)
- bronze.products (product_id, category)

Target Schema:
- date (DATE)
- region (VARCHAR)
- category (VARCHAR)
- total_revenue (DECIMAL) = SUM(total_amount)
- order_count (INTEGER) = COUNT(DISTINCT order_id)

Requirements:
- Join orders and products on product_id
- Filter for status = 'completed'
- Group by date, region, category
- Only include rows where revenue > 0

Generate optimized SQL with proper formatting.
```

3. **Validation**:
   - Parse SQL using `sqlparse` library
   - Check for syntax errors
   - Verify all source columns exist
   - Ensure aggregations match target schema

**Alternative: PySpark Code Generation** (for large-scale processing):
```python
{
    "language": "PySpark",
    "code": """
        from pyspark.sql import functions as F
        
        orders = spark.table("bronze.orders")
        products = spark.table("bronze.products")
        
        result = (
            orders
            .filter(F.col("status") == "completed")
            .join(products, "product_id")
            .groupBy(
                F.to_date("order_date").alias("date"),
                "region",
                "category"
            )
            .agg(
                F.sum("total_amount").alias("total_revenue"),
                F.countDistinct("order_id").alias("order_count"),
                F.sum("quantity").alias("units_sold")
            )
            .filter(F.col("total_revenue") > 0)
        )
    """
}
```

---

### 2.6 Quality Agent

**Purpose**: Define data quality rules and validation tests

**Input**: 
- Target schema from Modeling Agent
- Business requirements from Intent Agent
- Historical quality metrics from Knowledge Graph

**Output**:
```python
{
    "quality_rules": [
        {
            "rule_id": "QR001",
            "rule_type": "completeness",
            "column": "date",
            "condition": "date IS NOT NULL",
            "severity": "error",
            "description": "Date must be present for all records"
        },
        {
            "rule_id": "QR002",
            "rule_type": "accuracy",
            "column": "total_revenue",
            "condition": "total_revenue >= 0",
            "severity": "error",
            "description": "Revenue cannot be negative"
        },
        {
            "rule_id": "QR003",
            "rule_type": "consistency",
            "condition": "order_count >= units_sold",
            "severity": "warning",
            "description": "Order count should be at least units sold"
        }
    ],
    "sla": {
        "freshness": "Daily at 6:00 AM UTC",
        "latency": "< 30 minutes",
        "completeness": "> 99%",
        "accuracy": "> 95%"
    },
    "tests": [
        {
            "test_type": "dbt_test",
            "code": """
                SELECT *
                FROM {{ ref('daily_sales_analytics') }}
                WHERE total_revenue < 0
            """
        }
    ]
}
```

**Rule Generation Strategy**:

1. **Schema-Based Rules** (automatic):
   - NOT NULL constraints → completeness tests
   - Primary keys → uniqueness tests
   - Foreign keys → referential integrity tests

2. **Domain-Based Rules** (from Knowledge Graph):
```cypher
MATCH (c:Column {name: $column_name})-[:HAS_QUALITY_RULE]->(qr:QualityRule)
RETURN qr.rule_type, qr.condition, qr.severity
```

3. **Business Logic Rules** (LLM-generated):
```
You are a data quality expert. Define validation rules for:

Target Schema:
- total_revenue (DECIMAL): Sum of order amounts
- order_count (INTEGER): Count of distinct orders
- units_sold (INTEGER): Sum of quantities

Business Context:
- Daily sales analytics
- Used for executive dashboards

Generate rules for:
1. Range checks (min/max values)
2. Relationship checks (between columns)
3. Temporal checks (date ranges)
4. Statistical checks (outliers)

Respond in JSON format with severity levels (error/warning).
```

**Great Expectations Integration**:
```python
def generate_ge_suite(quality_rules):
    """Convert quality rules to Great Expectations suite"""
    suite = {
        "expectation_suite_name": "daily_sales_analytics",
        "expectations": []
    }
    
    for rule in quality_rules:
        if rule["rule_type"] == "completeness":
            suite["expectations"].append({
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {"column": rule["column"]}
            })
        elif rule["rule_type"] == "accuracy":
            suite["expectations"].append({
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {
                    "column": rule["column"],
                    "min_value": 0
                }
            })
    
    return suite
```

---

### 2.7 Packaging Agent

**Purpose**: Compile all outputs into final Data Product specification

**Input**: Outputs from all previous agents

**Output**: Complete YAML/JSON specification (see PROJECT_PLAN.md section 6.1)

**Process**:
1. Aggregate all agent outputs
2. Add metadata (version, timestamp, owner)
3. Generate documentation
4. Create lineage graph
5. Validate against Data Product schema
6. Export to YAML/JSON

**Validation**:
```python
import jsonschema

def validate_data_product(spec, schema_path):
    """Validate data product spec against JSON Schema"""
    with open(schema_path) as f:
        schema = json.load(f)
    
    try:
        jsonschema.validate(instance=spec, schema=schema)
        return True, None
    except jsonschema.ValidationError as e:
        return False, str(e)
```

**Lineage Generation**:
```python
def generate_lineage(source_datasets, target_table):
    """Create lineage graph"""
    lineage = {
        "nodes": [],
        "edges": []
    }
    
    # Add source nodes
    for ds in source_datasets:
        lineage["nodes"].append({
            "id": ds["name"],
            "type": "source",
            "layer": "bronze"
        })
    
    # Add target node
    lineage["nodes"].append({
        "id": target_table,
        "type": "target",
        "layer": "gold"
    })
    
    # Add edges
    for ds in source_datasets:
        lineage["edges"].append({
            "from": ds["name"],
            "to": target_table,
            "transformation": "aggregation"
        })
    
    return lineage
```

---

## 3. Knowledge Graph Design

### 3.1 Graph Schema (Neo4j)

**Node Labels**:
```cypher
// Dataset node
CREATE (d:Dataset {
    name: 'bronze.orders',
    type: 'table',
    layer: 'bronze',
    description: 'Raw order transactions',
    owner: 'data_engineering',
    created_at: datetime(),
    updated_at: datetime(),
    row_count: 100000,
    size_bytes: 5242880,
    freshness_sla: 'daily'
})

// Column node
CREATE (c:Column {
    name: 'total_amount',
    data_type: 'DECIMAL(18,2)',
    description: 'Total order amount in USD',
    is_nullable: false,
    is_pii: false,
    sample_values: '[10.50, 25.99, 100.00]',
    distinct_count: 50000,
    null_percentage: 0.0,
    embedding: [0.1, 0.2, ..., 0.768]  // OpenAI embedding
})

// Business Term node
CREATE (bt:BusinessTerm {
    term: 'revenue',
    definition: 'Total monetary value from sales',
    synonyms: ['sales', 'income', 'proceeds'],
    domain: 'finance',
    steward: 'finance_team'
})

// Quality Rule node
CREATE (qr:QualityRule {
    rule_id: 'QR001',
    rule_type: 'accuracy',
    condition: 'total_amount >= 0',
    threshold: 1.0,
    severity: 'error'
})
```

**Relationships**:
```cypher
// Dataset has columns
MATCH (d:Dataset {name: 'bronze.orders'})
MATCH (c:Column {name: 'total_amount'})
CREATE (d)-[:HAS_COLUMN]->(c)

// Column maps to business term
MATCH (c:Column {name: 'total_amount'})
MATCH (bt:BusinessTerm {term: 'revenue'})
CREATE (c)-[:MAPS_TO {confidence: 0.95}]->(bt)

// Dataset lineage
MATCH (source:Dataset {name: 'bronze.orders'})
MATCH (target:Dataset {name: 'gold.daily_sales'})
CREATE (target)-[:DERIVED_FROM {transformation: 'aggregation'}]->(source)

// Column quality rules
MATCH (c:Column {name: 'total_amount'})
MATCH (qr:QualityRule {rule_id: 'QR001'})
CREATE (c)-[:HAS_QUALITY_RULE]->(qr)
```

### 3.2 Graph Population Script

```python
from neo4j import GraphDatabase

class KnowledgeGraphBuilder:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def populate_bronze_metadata(self, datasets):
        """Populate metadata for bronze layer datasets"""
        with self.driver.session() as session:
            for ds in datasets:
                # Create dataset node
                session.run("""
                    CREATE (d:Dataset {
                        name: $name,
                        type: $type,
                        layer: 'bronze',
                        description: $description,
                        row_count: $row_count
                    })
                """, **ds)
                
                # Create column nodes and relationships
                for col in ds['columns']:
                    session.run("""
                        MATCH (d:Dataset {name: $dataset_name})
                        CREATE (c:Column {
                            name: $col_name,
                            data_type: $data_type,
                            description: $description
                        })
                        CREATE (d)-[:HAS_COLUMN]->(c)
                    """, dataset_name=ds['name'], **col)
    
    def create_business_term_mappings(self, mappings):
        """Create column-to-business-term mappings"""
        with self.driver.session() as session:
            for mapping in mappings:
                session.run("""
                    MATCH (c:Column {name: $column_name})
                    MERGE (bt:BusinessTerm {term: $business_term})
                    ON CREATE SET bt.definition = $definition
                    CREATE (c)-[:MAPS_TO {confidence: $confidence}]->(bt)
                """, **mapping)
```

### 3.3 Query Optimization

**Indexes**:
```cypher
// Create indexes for fast lookups
CREATE INDEX dataset_name_idx FOR (d:Dataset) ON (d.name);
CREATE INDEX column_name_idx FOR (c:Column) ON (c.name);
CREATE INDEX business_term_idx FOR (bt:BusinessTerm) ON (bt.term);

// Create full-text search index
CREATE FULLTEXT INDEX column_description_idx 
FOR (c:Column) ON EACH [c.description, c.name];
```

**Vector Similarity Search** (using Neo4j GDS):
```cypher
// Find similar columns using embeddings
CALL gds.similarity.cosine.stream({
    nodeProjection: 'Column',
    relationshipProjection: '*',
    nodeProperties: ['embedding']
})
YIELD node1, node2, similarity
WHERE similarity > 0.8
RETURN gds.util.asNode(node1).name AS column1,
       gds.util.asNode(node2).name AS column2,
       similarity
ORDER BY similarity DESC
```

---

## 4. Data Layer Architecture

### 4.1 Bronze Layer Structure

```
data/bronze/
├── orders/
│   ├── orders.parquet
│   └── _metadata.json
├── customers/
│   ├── customers.parquet
│   └── _metadata.json
├── products/
│   ├── products.parquet
│   └── _metadata.json
├── marketing_campaigns/
│   ├── campaigns.parquet
│   └── _metadata.json
├── marketing_events/
│   ├── events.parquet
│   └── _metadata.json
├── user_interactions/
│   ├── interactions.parquet
│   └── _metadata.json
└── support_tickets/
    ├── tickets.parquet
    └── _metadata.json
```

**Metadata Format** (`_metadata.json`):
```json
{
    "dataset_name": "bronze.orders",
    "created_at": "2025-11-15T10:00:00Z",
    "row_count": 100000,
    "size_bytes": 5242880,
    "schema": [
        {
            "name": "order_id",
            "type": "VARCHAR",
            "nullable": false
        },
        {
            "name": "total_amount",
            "type": "DECIMAL(18,2)",
            "nullable": false
        }
    ],
    "statistics": {
        "total_amount": {
            "min": 5.00,
            "max": 5000.00,
            "mean": 125.50,
            "median": 75.00,
            "std": 150.25
        }
    }
}
```

### 4.2 DuckDB Integration

**Why DuckDB**:
- Embedded analytics database (no server needed)
- Native Parquet support
- Fast analytical queries
- SQL interface

**Usage**:
```python
import duckdb

class BronzeDataAccess:
    def __init__(self, data_path):
        self.conn = duckdb.connect(':memory:')
        self.data_path = data_path
    
    def query(self, sql):
        """Execute SQL query on bronze data"""
        return self.conn.execute(sql).fetchdf()
    
    def register_tables(self):
        """Register all bronze tables"""
        self.conn.execute(f"""
            CREATE VIEW bronze_orders AS 
            SELECT * FROM read_parquet('{self.data_path}/orders/orders.parquet')
        """)
        
        self.conn.execute(f"""
            CREATE VIEW bronze_products AS 
            SELECT * FROM read_parquet('{self.data_path}/products/products.parquet')
        """)
    
    def validate_transformation(self, sql, expected_columns):
        """Validate generated SQL before execution"""
        try:
            result = self.conn.execute(sql).fetchdf()
            actual_columns = set(result.columns)
            expected_columns = set(expected_columns)
            
            if actual_columns != expected_columns:
                return False, f"Column mismatch: {actual_columns} vs {expected_columns}"
            
            return True, result
        except Exception as e:
            return False, str(e)
```

---

## 5. Integration Points

### 5.1 LLM Integration

**OpenAI Client Wrapper**:
```python
from openai import OpenAI
import json

class LLMClient:
    def __init__(self, api_key, model="gpt-4-turbo"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def generate_structured_output(self, prompt, response_schema):
        """Generate JSON output matching schema"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a data engineering expert."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        
        return json.loads(response.choices[0].message.content)
    
    def generate_code(self, prompt, language="SQL"):
        """Generate SQL/Python code"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": f"You are a {language} expert. Generate clean, optimized code."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        return response.choices[0].message.content
    
    def get_embedding(self, text):
        """Get text embedding for semantic search"""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        
        return response.data[0].embedding
```

### 5.2 API Design

**FastAPI Server**:
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Agentic Data Product Builder API")

class DataProductRequest(BaseModel):
    user_request: str
    use_case_id: Optional[str] = None

class DataProductResponse(BaseModel):
    data_product_spec: Dict
    execution_time_ms: int
    agent_logs: List[Dict]

@app.post("/api/v1/generate", response_model=DataProductResponse)
async def generate_data_product(request: DataProductRequest):
    """Generate data product from natural language request"""
    try:
        orchestrator = OrchestratorAgent()
        result = orchestrator.run(request.user_request)
        
        return DataProductResponse(
            data_product_spec=result["spec"],
            execution_time_ms=result["execution_time"],
            agent_logs=result["logs"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/datasets")
async def list_datasets():
    """List available bronze datasets"""
    kg_client = KnowledgeGraphClient()
    datasets = kg_client.get_all_datasets()
    return {"datasets": datasets}

@app.get("/api/v1/business-terms")
async def list_business_terms():
    """List business glossary terms"""
    kg_client = KnowledgeGraphClient()
    terms = kg_client.get_all_business_terms()
    return {"terms": terms}
```

---

## 6. Error Handling & Resilience

### 6.1 Retry Strategy

```python
from tenacity import retry, stop_after_attempt, wait_exponential

class ResilientAgent:
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def call_llm(self, prompt):
        """Call LLM with automatic retry"""
        return self.llm_client.generate(prompt)
    
    def execute_with_fallback(self, primary_fn, fallback_fn):
        """Execute with fallback strategy"""
        try:
            return primary_fn()
        except Exception as e:
            logger.warning(f"Primary execution failed: {e}. Using fallback.")
            return fallback_fn()
```

### 6.2 Validation Pipeline

```python
class ValidationPipeline:
    def __init__(self):
        self.validators = []
    
    def add_validator(self, validator_fn):
        self.validators.append(validator_fn)
    
    def validate(self, data):
        """Run all validators"""
        errors = []
        
        for validator in self.validators:
            is_valid, error_msg = validator(data)
            if not is_valid:
                errors.append(error_msg)
        
        return len(errors) == 0, errors

# Usage
pipeline = ValidationPipeline()
pipeline.add_validator(validate_schema)
pipeline.add_validator(validate_sql_syntax)
pipeline.add_validator(validate_data_types)

is_valid, errors = pipeline.validate(transformation_output)
```

---

## 7. Monitoring & Observability

### 7.1 Logging Strategy

```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    def __init__(self, agent_name):
        self.agent_name = agent_name
        self.logger = logging.getLogger(agent_name)
    
    def log_execution(self, input_data, output_data, execution_time_ms):
        """Log agent execution"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent": self.agent_name,
            "input": input_data,
            "output": output_data,
            "execution_time_ms": execution_time_ms
        }
        
        self.logger.info(json.dumps(log_entry))
```

### 7.2 Metrics Collection

```python
class MetricsCollector:
    def __init__(self):
        self.metrics = {
            "agent_execution_times": {},
            "llm_token_usage": 0,
            "kg_query_count": 0,
            "success_rate": 0.0
        }
    
    def record_agent_execution(self, agent_name, execution_time_ms):
        if agent_name not in self.metrics["agent_execution_times"]:
            self.metrics["agent_execution_times"][agent_name] = []
        
        self.metrics["agent_execution_times"][agent_name].append(execution_time_ms)
    
    def get_summary(self):
        """Get metrics summary"""
        summary = {}
        
        for agent, times in self.metrics["agent_execution_times"].items():
            summary[agent] = {
                "avg_time_ms": sum(times) / len(times),
                "min_time_ms": min(times),
                "max_time_ms": max(times),
                "executions": len(times)
            }
        
        return summary
```

---

## 8. Security & Governance

### 8.1 Access Control

```python
class AccessController:
    def __init__(self, kg_client):
        self.kg_client = kg_client
    
    def check_dataset_access(self, user_id, dataset_name):
        """Check if user has access to dataset"""
        query = """
        MATCH (u:User {id: $user_id})-[:HAS_ROLE]->(r:Role)
        MATCH (d:Dataset {name: $dataset_name})-[:REQUIRES_ROLE]->(r)
        RETURN COUNT(*) > 0 AS has_access
        """
        
        result = self.kg_client.query(query, user_id=user_id, dataset_name=dataset_name)
        return result[0]["has_access"]
    
    def filter_datasets_by_access(self, user_id, datasets):
        """Filter datasets based on user access"""
        accessible = []
        
        for ds in datasets:
            if self.check_dataset_access(user_id, ds["name"]):
                accessible.append(ds)
        
        return accessible
```

### 8.2 PII Detection

```python
class PIIDetector:
    def __init__(self):
        self.pii_patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b"
        }
    
    def detect_pii_columns(self, dataset_name):
        """Detect PII columns in dataset"""
        query = """
        MATCH (d:Dataset {name: $dataset_name})-[:HAS_COLUMN]->(c:Column)
        WHERE c.is_pii = true
        RETURN c.name AS column_name, c.pii_type AS pii_type
        """
        
        return self.kg_client.query(query, dataset_name=dataset_name)
    
    def mask_pii(self, value, pii_type):
        """Mask PII value"""
        if pii_type == "email":
            return "***@***.com"
        elif pii_type == "phone":
            return "***-***-****"
        else:
            return "***"
```

---

## Summary

This architecture provides:

1. **Modularity**: Each agent is independent and replaceable
2. **Scalability**: Can add new agents without changing core orchestration
3. **Extensibility**: Knowledge Graph can grow with new metadata
4. **Resilience**: Retry logic and fallback strategies
5. **Observability**: Comprehensive logging and metrics
6. **Security**: Access control and PII detection

**Next Steps**: Implement each component incrementally, starting with bronze data generation and Knowledge Graph setup.

