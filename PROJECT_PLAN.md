# Agentic Data Product Builder - Project Plan

## Project Overview

**Title**: Agentic Data Product Builder: A Knowledge-Graph-Driven Multi-Agent System for Automated Data Product Generation

**Goal**: Create a prototype system where multiple AI agents convert natural-language business requests into complete Data Product specifications, automating the entire data engineering pipeline.

---

## 1. System Architecture

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     User Interface Layer                         │
│  (Natural Language Request Input + Data Product Output Display)  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                   Orchestrator Agent                             │
│  (Coordinates multi-agent workflow, manages state)               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐  ┌──────▼──────┐  ┌────────▼────────┐
│ Intent Agent   │  │ Discovery   │  │ Modeling Agent  │
│                │  │ Agent       │  │                 │
└───────┬────────┘  └──────┬──────┘  └────────┬────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐  ┌──────▼──────┐  ┌────────▼────────┐
│ Transform      │  │ Quality     │  │ Packaging       │
│ Agent          │  │ Agent       │  │ Agent           │
└───────┬────────┘  └──────┬──────┘  └────────┬────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    Knowledge Graph Layer                         │
│  (Neo4j/NetworkX: Datasets, Columns, Lineage, Glossary, etc.)  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    Bronze Data Layer                             │
│         (Synthetic datasets for 4 use cases)                     │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Technology Stack (Proposed)

**Core Framework**:
- Python 3.10+
- LangGraph / CrewAI / AutoGen (for multi-agent orchestration)
- OpenAI GPT-4 / Anthropic Claude (LLM backbone)

**Knowledge Graph**:
- Neo4j (graph database) OR NetworkX (in-memory graph)
- Cypher query language

**Data Layer**:
- Pandas / Polars (data manipulation)
- DuckDB (embedded analytics)
- Faker / Mimesis (synthetic data generation)

**Output Format**:
- YAML / JSON (Data Product specification)
- Jinja2 (template generation)

**Infrastructure**:
- Docker (containerization)
- FastAPI (REST API)
- Streamlit / Gradio (UI)

---

## 2. Multi-Agent System Design

### 2.1 Agent Roles & Responsibilities

#### **Orchestrator Agent**
- **Role**: Master coordinator
- **Responsibilities**:
  - Parse initial user request
  - Manage agent workflow (sequential/parallel execution)
  - Maintain conversation state
  - Handle errors and retries
  - Compile final Data Product specification

#### **Intent Agent**
- **Role**: Business requirement analyzer
- **Responsibilities**:
  - Extract business metrics (e.g., "daily revenue", "CTR", "ROAS")
  - Identify aggregation levels (region, category, time period)
  - Determine data freshness requirements
  - Map business terms to technical concepts
  - Query Knowledge Graph for glossary terms

#### **Discovery Agent**
- **Role**: Dataset finder
- **Responsibilities**:
  - Search Knowledge Graph for relevant datasets
  - Match business entities to available tables/columns
  - Evaluate data quality scores
  - Check data availability and freshness
  - Identify data owners and access policies
  - Return ranked list of candidate datasets

#### **Modeling Agent**
- **Role**: Data model designer
- **Responsibilities**:
  - Design target schema (columns, types, constraints)
  - Define grain/granularity of output
  - Specify primary keys and partitioning strategy
  - Create logical data model (star schema, wide table, etc.)
  - Define relationships and join logic

#### **Transformation Agent**
- **Role**: Code generator
- **Responsibilities**:
  - Generate SQL/Python transformation code
  - Implement business logic (calculations, aggregations)
  - Handle data type conversions
  - Add data validation logic
  - Create incremental load logic
  - Generate dbt models or PySpark scripts

#### **Quality Agent**
- **Role**: Data quality enforcer
- **Responsibilities**:
  - Define data quality rules (completeness, accuracy, consistency)
  - Generate validation tests (Great Expectations, dbt tests)
  - Set SLA expectations (freshness, latency)
  - Define monitoring metrics
  - Query Knowledge Graph for existing quality rules

#### **Packaging Agent**
- **Role**: Specification compiler
- **Responsibilities**:
  - Compile all outputs into YAML/JSON specification
  - Add metadata (version, owner, tags, description)
  - Generate documentation
  - Create lineage information
  - Package code artifacts
  - Validate final specification against schema

### 2.2 Agent Communication Protocol

**Message Format**:
```json
{
  "agent_id": "intent_agent",
  "timestamp": "2025-11-15T10:30:00Z",
  "status": "completed",
  "input": { ... },
  "output": { ... },
  "metadata": {
    "execution_time_ms": 1234,
    "confidence_score": 0.92
  }
}
```

**Workflow Patterns**:
1. **Sequential**: Intent → Discovery → Modeling → Transform → Quality → Packaging
2. **Parallel**: Discovery + Quality rules lookup (when independent)
3. **Iterative**: Modeling ↔ Transform (refinement loop)

---

## 3. Knowledge Graph Design

### 3.1 Node Types

#### **Dataset**
- Properties: `name`, `type` (table/view/file), `layer` (bronze/silver/gold), `description`, `owner`, `created_at`, `updated_at`, `row_count`, `size_bytes`, `freshness_sla`

#### **Column**
- Properties: `name`, `data_type`, `description`, `is_nullable`, `is_pii`, `sample_values`, `distinct_count`, `null_percentage`

#### **BusinessTerm**
- Properties: `term`, `definition`, `synonyms`, `domain`, `steward`

#### **QualityRule**
- Properties: `rule_id`, `rule_type` (completeness/accuracy/consistency), `condition`, `threshold`, `severity`

#### **DataOwner**
- Properties: `name`, `email`, `team`, `role`

#### **Transformation**
- Properties: `transform_id`, `code`, `language` (SQL/Python), `description`

### 3.2 Relationship Types

- `(Dataset)-[:HAS_COLUMN]->(Column)`
- `(Column)-[:MAPS_TO]->(BusinessTerm)`
- `(Dataset)-[:OWNED_BY]->(DataOwner)`
- `(Dataset)-[:DERIVED_FROM]->(Dataset)` [lineage]
- `(Column)-[:HAS_QUALITY_RULE]->(QualityRule)`
- `(Dataset)-[:TRANSFORMED_BY]->(Transformation)`
- `(Column)-[:SIMILAR_TO]->(Column)` [semantic similarity]

### 3.3 Sample Graph Queries

**Find datasets containing revenue data**:
```cypher
MATCH (d:Dataset)-[:HAS_COLUMN]->(c:Column)-[:MAPS_TO]->(bt:BusinessTerm)
WHERE bt.term IN ['revenue', 'sales', 'amount']
RETURN d.name, c.name, bt.term
```

**Get lineage for a dataset**:
```cypher
MATCH path = (source:Dataset)-[:DERIVED_FROM*]->(target:Dataset)
WHERE source.name = 'daily_sales_analytics'
RETURN path
```

---

## 4. Bronze Layer Design

### 4.1 Synthetic Datasets

#### **orders**
- Columns: `order_id`, `customer_id`, `product_id`, `order_date`, `quantity`, `unit_price`, `total_amount`, `region`, `category`, `status`
- Rows: ~100,000
- Date Range: 2023-01-01 to 2025-11-15

#### **customers**
- Columns: `customer_id`, `name`, `email`, `phone`, `signup_date`, `loyalty_tier`, `total_lifetime_value`, `segment`
- Rows: ~10,000

#### **products**
- Columns: `product_id`, `product_name`, `category`, `subcategory`, `brand`, `price`, `cost`, `margin_pct`
- Rows: ~1,000

#### **marketing_campaigns**
- Columns: `campaign_id`, `campaign_name`, `channel`, `start_date`, `end_date`, `budget`, `target_audience`
- Rows: ~50

#### **marketing_events**
- Columns: `event_id`, `campaign_id`, `event_date`, `event_type` (impression/click/conversion), `user_id`, `cost`, `revenue`
- Rows: ~500,000

#### **user_interactions**
- Columns: `interaction_id`, `user_id`, `product_id`, `interaction_type` (view/cart/purchase/rating), `timestamp`, `rating`, `session_id`
- Rows: ~200,000

#### **support_tickets**
- Columns: `ticket_id`, `customer_id`, `created_at`, `resolved_at`, `category`, `priority`, `status`, `satisfaction_score`
- Rows: ~5,000

### 4.2 Data Generation Strategy

**Approach**:
1. Use Faker library for realistic names, emails, dates
2. Use NumPy for statistical distributions (revenue follows log-normal, etc.)
3. Maintain referential integrity (customer_id in orders exists in customers)
4. Inject intentional quality issues (5% nulls, duplicates, outliers) for testing
5. Create temporal patterns (seasonality, trends)

**Script Structure**:
```
data_generation/
├── generate_bronze.py
├── schemas/
│   ├── orders_schema.yaml
│   ├── customers_schema.yaml
│   └── ...
└── output/
    ├── orders.parquet
    ├── customers.parquet
    └── ...
```

---

## 5. Use Case Specifications

### 5.1 A1 — Daily Sales Analytics

**Input Request**: 
> "I need a daily sales analytics data product showing revenue, order count, and units sold, broken down by region and product category."

**Expected Output**:
- **Datasets Used**: orders, products
- **Target Schema**: `date`, `region`, `category`, `total_revenue`, `order_count`, `units_sold`
- **Grain**: Daily, by region and category
- **Transformations**: 
  - `SUM(total_amount)` AS total_revenue
  - `COUNT(DISTINCT order_id)` AS order_count
  - `SUM(quantity)` AS units_sold
- **Quality Rules**: 
  - Revenue > 0
  - No nulls in date, region, category
- **Freshness**: Daily at 6 AM

### 5.2 A2 — Marketing Campaign Performance

**Input Request**: 
> "Create a weekly marketing performance report with CTR, CVR, CPA, and ROAS for each campaign."

**Expected Output**:
- **Datasets Used**: marketing_campaigns, marketing_events
- **Target Schema**: `week`, `campaign_id`, `campaign_name`, `impressions`, `clicks`, `conversions`, `spend`, `revenue`, `ctr`, `cvr`, `cpa`, `roas`
- **Grain**: Weekly, by campaign
- **Transformations**: 
  - `CTR = clicks / impressions`
  - `CVR = conversions / clicks`
  - `CPA = spend / conversions`
  - `ROAS = revenue / spend`
- **Quality Rules**: 
  - Impressions >= clicks >= conversions
  - Spend > 0 for active campaigns

### 5.3 B3 — Product Recommendation Features

**Input Request**: 
> "Build a feature table for product recommendations based on user interaction signals."

**Expected Output**:
- **Datasets Used**: user_interactions, products, orders
- **Target Schema**: `user_id`, `product_id`, `view_count`, `cart_count`, `purchase_count`, `avg_rating`, `days_since_last_interaction`, `category_affinity_score`
- **Grain**: User-product pair
- **Transformations**: 
  - Aggregate interaction counts by type
  - Calculate recency features
  - Compute category-level preferences
- **Quality Rules**: 
  - All counts >= 0
  - Ratings between 1-5

### 5.4 C2 — Customer 360

**Input Request**: 
> "Create a unified customer 360 view combining profile, transaction history, loyalty status, and support interactions."

**Expected Output**:
- **Datasets Used**: customers, orders, support_tickets
- **Target Schema**: `customer_id`, `name`, `email`, `signup_date`, `loyalty_tier`, `total_orders`, `total_revenue`, `avg_order_value`, `last_order_date`, `open_tickets`, `avg_satisfaction_score`, `segment`
- **Grain**: One row per customer
- **Transformations**: 
  - Join customers with aggregated orders and tickets
  - Calculate lifetime metrics
  - Derive customer segment
- **Quality Rules**: 
  - Unique customer_id
  - Email format validation

---

## 6. Data Product Specification Schema

### 6.1 Output YAML Structure

```yaml
data_product:
  metadata:
    name: "daily_sales_analytics"
    version: "1.0.0"
    description: "Daily sales metrics by region and category"
    owner: "analytics_team"
    created_at: "2025-11-15T10:30:00Z"
    tags: ["sales", "analytics", "daily"]
  
  business_context:
    use_case: "A1 - Daily Sales Analytics"
    business_metrics:
      - "total_revenue"
      - "order_count"
      - "units_sold"
    stakeholders:
      - "Sales Team"
      - "Finance Team"
  
  data_model:
    target_table: "gold.daily_sales_analytics"
    grain: "Daily, by region and category"
    schema:
      - name: "date"
        type: "DATE"
        description: "Transaction date"
        nullable: false
        primary_key: true
      - name: "region"
        type: "VARCHAR"
        description: "Sales region"
        nullable: false
        primary_key: true
      - name: "category"
        type: "VARCHAR"
        description: "Product category"
        nullable: false
        primary_key: true
      - name: "total_revenue"
        type: "DECIMAL(18,2)"
        description: "Sum of order amounts"
        nullable: false
      - name: "order_count"
        type: "INTEGER"
        description: "Count of distinct orders"
        nullable: false
      - name: "units_sold"
        type: "INTEGER"
        description: "Sum of quantities"
        nullable: false
  
  source_datasets:
    - name: "bronze.orders"
      columns: ["order_id", "order_date", "quantity", "total_amount", "region"]
    - name: "bronze.products"
      columns: ["product_id", "category"]
  
  transformations:
    language: "SQL"
    code: |
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
  
  quality_rules:
    - rule: "total_revenue > 0"
      severity: "error"
    - rule: "date IS NOT NULL"
      severity: "error"
    - rule: "order_count >= units_sold"
      severity: "warning"
  
  sla:
    freshness: "Daily at 6:00 AM UTC"
    latency: "< 30 minutes"
    completeness: "> 99%"
  
  lineage:
    upstream:
      - "bronze.orders"
      - "bronze.products"
    downstream:
      - "gold.regional_sales_dashboard"
```

---

## 7. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- [ ] Set up project structure and dependencies
- [ ] Generate synthetic bronze layer datasets
- [ ] Build Knowledge Graph schema
- [ ] Populate Knowledge Graph with metadata
- [ ] Create Data Product specification schema (JSON Schema)

### Phase 2: Core Agents (Weeks 3-5)
- [ ] Implement Orchestrator Agent
- [ ] Implement Intent Agent (with LLM integration)
- [ ] Implement Discovery Agent (with graph queries)
- [ ] Implement Modeling Agent
- [ ] Create agent communication framework

### Phase 3: Transformation & Quality (Weeks 6-7)
- [ ] Implement Transformation Agent (SQL generation)
- [ ] Implement Quality Agent
- [ ] Implement Packaging Agent
- [ ] Build YAML/JSON output generator

### Phase 4: Use Case Implementation (Weeks 8-9)
- [ ] Test with A1 (Daily Sales Analytics)
- [ ] Test with A2 (Marketing Campaign Performance)
- [ ] Test with B3 (Product Recommendation Features)
- [ ] Test with C2 (Customer 360)
- [ ] Refine agents based on results

### Phase 5: UI & Integration (Week 10)
- [ ] Build Streamlit/Gradio UI
- [ ] Add conversation history
- [ ] Add Data Product preview/validation
- [ ] Create REST API endpoints

### Phase 6: Evaluation & Documentation (Weeks 11-12)
- [ ] Define evaluation metrics (accuracy, completeness, execution time)
- [ ] Run benchmark tests
- [ ] Write dissertation documentation
- [ ] Create demo video
- [ ] Prepare presentation

---

## 8. Evaluation Criteria

### 8.1 Quantitative Metrics

**Accuracy**:
- % of correctly identified source datasets
- % of correctly mapped business terms
- % of valid SQL/Python code generated

**Completeness**:
- % of Data Product specification fields populated
- % of required quality rules defined

**Performance**:
- End-to-end execution time (target: < 2 minutes)
- Agent-level execution time breakdown

### 8.2 Qualitative Assessment

**Code Quality**:
- Readability of generated SQL/Python
- Adherence to best practices
- Error handling

**Specification Quality**:
- Clarity of documentation
- Completeness of metadata
- Usefulness of lineage information

---

## 9. Key Challenges & Mitigation

| Challenge | Mitigation Strategy |
|-----------|---------------------|
| LLM hallucination in dataset discovery | Use Knowledge Graph as ground truth; implement validation layer |
| Complex SQL generation for joins | Provide few-shot examples; use chain-of-thought prompting |
| Agent coordination complexity | Use LangGraph for explicit state management |
| Knowledge Graph schema evolution | Version the graph schema; use migration scripts |
| Handling ambiguous user requests | Implement clarification dialog; Intent Agent asks follow-up questions |

---

## 10. Project Structure (Proposed)

```
mvp-1/
├── README.md
├── requirements.txt
├── .env.example
├── config/
│   ├── agents_config.yaml
│   └── kg_schema.yaml
├── data/
│   ├── bronze/
│   │   ├── orders.parquet
│   │   ├── customers.parquet
│   │   └── ...
│   └── generation/
│       └── generate_bronze.py
├── knowledge_graph/
│   ├── schema.py
│   ├── populate.py
│   └── queries.py
├── agents/
│   ├── base_agent.py
│   ├── orchestrator.py
│   ├── intent_agent.py
│   ├── discovery_agent.py
│   ├── modeling_agent.py
│   ├── transformation_agent.py
│   ├── quality_agent.py
│   └── packaging_agent.py
├── schemas/
│   ├── data_product_schema.json
│   └── templates/
│       └── data_product_template.yaml
├── utils/
│   ├── llm_client.py
│   ├── graph_client.py
│   └── validators.py
├── tests/
│   ├── test_agents.py
│   ├── test_kg_queries.py
│   └── use_cases/
│       ├── test_a1_sales.py
│       ├── test_a2_marketing.py
│       ├── test_b3_recommendations.py
│       └── test_c2_customer360.py
├── ui/
│   └── streamlit_app.py
├── api/
│   └── fastapi_server.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_kg_visualization.ipynb
│   └── 03_agent_testing.ipynb
└── docs/
    ├── architecture.md
    ├── agent_design.md
    └── evaluation.md
```

---

## 11. Research Contributions

### 11.1 Novel Aspects

1. **Knowledge-Graph-Driven Agent Coordination**: Using KG as shared context for multi-agent systems
2. **End-to-End Data Product Automation**: From NL request to executable code + specification
3. **Hybrid Reasoning**: Combining LLM reasoning with graph-based retrieval
4. **Domain-Specific Agent Specialization**: Each agent has focused expertise

### 11.2 Potential Publications

- Conference: IEEE BigData, ICDE, VLDB (demo track)
- Workshop: MLOps, Data Engineering, Knowledge Graphs
- Thesis: M.Tech dissertation

---

## 12. Next Steps (When Ready to Build)

1. **Set up Python environment** with LangGraph/CrewAI
2. **Generate bronze layer** synthetic data
3. **Build Knowledge Graph** with Neo4j/NetworkX
4. **Implement Intent Agent** as first proof-of-concept
5. **Test with A1 use case** end-to-end
6. **Iterate and expand** to remaining agents and use cases

---

## Appendix: References

- **Multi-Agent Systems**: LangGraph, CrewAI, AutoGen documentation
- **Knowledge Graphs**: Neo4j Graph Data Science, RDF/OWL standards
- **Data Products**: Data Mesh principles, Data Product Canvas
- **Code Generation**: CodeLlama, GPT-4 Code Interpreter
- **Data Quality**: Great Expectations, dbt testing framework

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-15  
**Author**: Anshul Parmar  
**Status**: Planning Phase

