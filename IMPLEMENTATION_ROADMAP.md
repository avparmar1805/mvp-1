# Implementation Roadmap - Agentic Data Product Builder

## Overview

This roadmap breaks down the implementation into 12 weeks, with clear milestones, deliverables, and success criteria for each phase.

---

## Timeline Summary

| Phase | Duration | Focus Area | Key Deliverables |
|-------|----------|------------|------------------|
| Phase 1 | Weeks 1-2 | Foundation & Data | Bronze layer, Knowledge Graph, Schemas |
| Phase 2 | Weeks 3-5 | Core Agents | Orchestrator, Intent, Discovery, Modeling |
| Phase 3 | Weeks 6-7 | Transformation & Quality | Code generation, Quality rules |
| Phase 4 | Weeks 8-9 | Use Case Testing | A1, A2, B3, C2 validation |
| Phase 5 | Week 10 | UI & Integration | Streamlit UI, REST API |
| Phase 6 | Weeks 11-12 | Evaluation & Documentation | Benchmarks, Dissertation |

---

## Phase 1: Foundation (Weeks 1-2)

### Week 1: Project Setup & Bronze Layer

#### Day 1-2: Environment Setup
**Tasks**:
- [ ] Create Python virtual environment
- [ ] Install dependencies (requirements.txt)
- [ ] Set up project structure
- [ ] Configure environment variables (.env)
- [ ] Set up Git repository with .gitignore

**Dependencies**:
```txt
# requirements.txt
python>=3.10

# Data Generation
faker==20.1.0
numpy==1.26.2
pandas==2.1.4
polars==0.19.19
pyarrow==14.0.1
mimesis==11.1.0

# Knowledge Graph
neo4j==5.15.0
networkx==3.2.1
py2neo==2021.2.3

# LLM & Embeddings
openai==1.6.1
anthropic==0.8.1
langchain==0.1.0
langgraph==0.0.20

# Multi-Agent Frameworks
crewai==0.1.0  # Alternative: autogen

# Data Processing
duckdb==0.9.2
sqlparse==0.4.4

# API & UI
fastapi==0.108.0
uvicorn==0.25.0
streamlit==1.29.0
pydantic==2.5.3

# Utilities
python-dotenv==1.0.0
pyyaml==6.0.1
jsonschema==4.20.0
tenacity==8.2.3

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1

# Logging & Monitoring
loguru==0.7.2
```

**Deliverables**:
- ✅ Working Python environment
- ✅ Project directory structure
- ✅ requirements.txt with all dependencies
- ✅ README.md with setup instructions

**Success Criteria**:
- All dependencies install without errors
- Can run `python --version` and see 3.10+

---

#### Day 3-4: Bronze Layer Generation

**Tasks**:
- [ ] Implement customer generator
- [ ] Implement product generator
- [ ] Implement order generator (with temporal patterns)
- [ ] Implement campaign & event generators
- [ ] Implement interaction generator
- [ ] Implement ticket generator
- [ ] Add data quality issue injection
- [ ] Create export to Parquet function
- [ ] Generate metadata JSON files

**Code Structure**:
```
data_generation/
├── generate_bronze.py
├── generators/
│   ├── customers.py
│   ├── products.py
│   ├── orders.py
│   ├── campaigns.py
│   ├── events.py
│   ├── interactions.py
│   └── tickets.py
├── utils/
│   ├── export.py
│   ├── validation.py
│   └── quality_injection.py
└── tests/
    └── test_generators.py
```

**Deliverables**:
- ✅ 7 Parquet files in `data/bronze/`
- ✅ Metadata JSON for each dataset
- ✅ Data generation script (`generate_bronze.py`)
- ✅ Validation report showing data quality

**Success Criteria**:
- 100,000 orders generated
- 10,000 customers generated
- 1,000 products generated
- Referential integrity maintained (95%+)
- Intentional quality issues present (as specified)

---

#### Day 5-6: Knowledge Graph Setup

**Tasks**:
- [ ] Install Neo4j (Docker or local)
- [ ] Define graph schema (nodes & relationships)
- [ ] Create indexes for performance
- [ ] Implement KG population script
- [ ] Populate dataset metadata
- [ ] Populate column metadata
- [ ] Create business term mappings
- [ ] Add quality rules
- [ ] Generate column embeddings (OpenAI)
- [ ] Test Cypher queries

**Code Structure**:
```
knowledge_graph/
├── schema.py          # Node & relationship definitions
├── populate.py        # Population script
├── queries.py         # Common query functions
├── embeddings.py      # Generate column embeddings
└── tests/
    └── test_kg_queries.py
```

**Key Cypher Queries to Implement**:
```cypher
// Find datasets by business term
MATCH (d:Dataset)-[:HAS_COLUMN]->(c:Column)-[:MAPS_TO]->(bt:BusinessTerm)
WHERE bt.term = $term
RETURN d, c

// Get dataset lineage
MATCH path = (d:Dataset)-[:DERIVED_FROM*]->(source:Dataset)
WHERE d.name = $dataset_name
RETURN path

// Find similar columns (by embedding)
CALL gds.similarity.cosine.stream(...)
```

**Deliverables**:
- ✅ Neo4j database running
- ✅ Graph populated with 7 datasets, ~50 columns
- ✅ 20+ business terms mapped
- ✅ Column embeddings generated
- ✅ Query functions tested

**Success Criteria**:
- Can query datasets by business term
- Can retrieve column metadata
- Similarity search returns relevant columns
- Query response time < 100ms

---

#### Day 7: Data Product Schema Definition

**Tasks**:
- [ ] Define Data Product JSON Schema
- [ ] Create YAML template
- [ ] Implement schema validator
- [ ] Create sample Data Product specs for 4 use cases
- [ ] Test validation

**Code Structure**:
```
schemas/
├── data_product_schema.json    # JSON Schema definition
├── templates/
│   └── data_product_template.yaml
└── examples/
    ├── a1_daily_sales.yaml
    ├── a2_marketing_performance.yaml
    ├── b3_recommendations.yaml
    └── c2_customer360.yaml
```

**JSON Schema Structure**:
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["metadata", "data_model", "transformations"],
  "properties": {
    "metadata": {
      "type": "object",
      "required": ["name", "version", "description"],
      "properties": {
        "name": {"type": "string"},
        "version": {"type": "string", "pattern": "^\\d+\\.\\d+\\.\\d+$"},
        "description": {"type": "string"}
      }
    },
    "data_model": {
      "type": "object",
      "required": ["target_table", "grain", "schema"],
      "properties": {
        "schema": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["name", "type"],
            "properties": {
              "name": {"type": "string"},
              "type": {"type": "string"},
              "nullable": {"type": "boolean"}
            }
          }
        }
      }
    }
  }
}
```

**Deliverables**:
- ✅ JSON Schema file
- ✅ YAML template
- ✅ 4 example Data Product specs
- ✅ Validation script

**Success Criteria**:
- Schema validates valid specs
- Schema rejects invalid specs
- Examples pass validation

---

### Week 2 Checkpoint

**Completed**:
- ✅ Bronze layer with 7 datasets (~100K rows total)
- ✅ Knowledge Graph with metadata
- ✅ Data Product schema defined

**Metrics**:
- Bronze data size: ~93 MB
- KG nodes: ~70 (datasets + columns + terms)
- KG relationships: ~150

**Ready for**: Agent development

---

## Phase 2: Core Agents (Weeks 3-5)

### Week 3: Orchestrator & Intent Agent

#### Day 1-2: Orchestrator Agent

**Tasks**:
- [ ] Set up LangGraph StateGraph
- [ ] Define DataProductState schema
- [ ] Implement workflow routing
- [ ] Add error handling & retry logic
- [ ] Implement state persistence
- [ ] Add execution logging

**Code Structure**:
```python
# agents/orchestrator.py

from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict

class DataProductState(TypedDict):
    user_request: str
    business_metrics: List[str]
    candidate_datasets: List[Dict]
    target_schema: List[Dict]
    transformation_code: str
    quality_rules: List[Dict]
    data_product_spec: Dict
    execution_log: List[Dict]
    errors: List[str]

class OrchestratorAgent:
    def __init__(self):
        self.workflow = self._build_workflow()
    
    def _build_workflow(self):
        workflow = StateGraph(DataProductState)
        
        # Add agent nodes
        workflow.add_node("intent", self.intent_agent)
        workflow.add_node("discovery", self.discovery_agent)
        workflow.add_node("modeling", self.modeling_agent)
        workflow.add_node("transformation", self.transformation_agent)
        workflow.add_node("quality", self.quality_agent)
        workflow.add_node("packaging", self.packaging_agent)
        
        # Define edges
        workflow.set_entry_point("intent")
        workflow.add_edge("intent", "discovery")
        workflow.add_edge("discovery", "modeling")
        workflow.add_edge("modeling", "transformation")
        workflow.add_edge("transformation", "quality")
        workflow.add_edge("quality", "packaging")
        workflow.add_edge("packaging", END)
        
        return workflow.compile()
    
    def run(self, user_request: str) -> Dict:
        initial_state = {
            "user_request": user_request,
            "execution_log": [],
            "errors": []
        }
        
        result = self.workflow.invoke(initial_state)
        return result
```

**Deliverables**:
- ✅ Orchestrator agent class
- ✅ State management
- ✅ Workflow graph
- ✅ Error handling

**Success Criteria**:
- Can execute full workflow (even with stub agents)
- State persists across agent calls
- Errors are captured and logged

---

#### Day 3-5: Intent Agent

**Tasks**:
- [ ] Implement LLM prompt template
- [ ] Add business metric extraction
- [ ] Add entity/dimension extraction
- [ ] Add temporal granularity detection
- [ ] Integrate with Knowledge Graph (business terms)
- [ ] Add confidence scoring
- [ ] Test with sample requests

**Code Structure**:
```python
# agents/intent_agent.py

from utils.llm_client import LLMClient
from knowledge_graph.queries import KnowledgeGraphClient

class IntentAgent:
    def __init__(self, llm_client: LLMClient, kg_client: KnowledgeGraphClient):
        self.llm = llm_client
        self.kg = kg_client
    
    def analyze(self, user_request: str) -> Dict:
        # Step 1: Extract structured intent using LLM
        prompt = self._build_prompt(user_request)
        llm_output = self.llm.generate_structured_output(
            prompt,
            response_schema={
                "business_metrics": ["string"],
                "dimensions": ["string"],
                "temporal_granularity": "string",
                "filters": ["string"]
            }
        )
        
        # Step 2: Map to business terms in KG
        business_terms = self.kg.find_business_terms(
            llm_output["business_metrics"]
        )
        
        # Step 3: Calculate confidence
        confidence = self._calculate_confidence(llm_output, business_terms)
        
        return {
            "business_metrics": llm_output["business_metrics"],
            "dimensions": llm_output["dimensions"],
            "temporal_granularity": llm_output["temporal_granularity"],
            "business_terms": business_terms,
            "confidence_score": confidence
        }
    
    def _build_prompt(self, user_request: str) -> str:
        return f"""
You are a business analyst expert. Analyze the following request:

Request: {user_request}

Extract:
1. Business metrics to calculate (e.g., revenue, count, average)
2. Dimensions for grouping (e.g., region, category, date)
3. Time granularity (hourly, daily, weekly, monthly)
4. Any filters or conditions

Respond in JSON format.
"""
```

**Test Cases**:
```python
# Test 1: A1 - Daily Sales
request = "I need daily sales analytics showing revenue, order count, and units sold by region and category"
expected = {
    "business_metrics": ["revenue", "order_count", "units_sold"],
    "dimensions": ["region", "category"],
    "temporal_granularity": "daily"
}

# Test 2: A2 - Marketing Performance
request = "Create a weekly marketing report with CTR, CVR, CPA, and ROAS"
expected = {
    "business_metrics": ["CTR", "CVR", "CPA", "ROAS"],
    "dimensions": ["campaign"],
    "temporal_granularity": "weekly"
}
```

**Deliverables**:
- ✅ Intent agent class
- ✅ LLM integration
- ✅ KG business term lookup
- ✅ Test suite with 4 use cases

**Success Criteria**:
- Correctly extracts metrics for all 4 use cases
- Confidence score > 0.8 for clear requests
- Handles ambiguous requests gracefully

---

### Week 4: Discovery & Modeling Agents

#### Day 1-3: Discovery Agent

**Tasks**:
- [ ] Implement dataset search by business terms
- [ ] Add semantic search using embeddings
- [ ] Implement dataset ranking algorithm
- [ ] Add quality score filtering
- [ ] Add access control checks
- [ ] Test with different queries

**Code Structure**:
```python
# agents/discovery_agent.py

class DiscoveryAgent:
    def __init__(self, kg_client: KnowledgeGraphClient, llm_client: LLMClient):
        self.kg = kg_client
        self.llm = llm_client
    
    def discover(self, business_metrics: List[str], dimensions: List[str]) -> Dict:
        # Step 1: Find datasets by business terms
        term_matches = self.kg.find_datasets_by_terms(
            business_metrics + dimensions
        )
        
        # Step 2: Semantic search using embeddings
        query_embedding = self.llm.get_embedding(
            " ".join(business_metrics + dimensions)
        )
        embedding_matches = self.kg.find_similar_columns(query_embedding)
        
        # Step 3: Combine and rank
        candidates = self._merge_results(term_matches, embedding_matches)
        ranked = self._rank_datasets(candidates, business_metrics, dimensions)
        
        # Step 4: Select top datasets
        selected = ranked[:5]  # Top 5
        
        return {
            "candidate_datasets": ranked,
            "selected_datasets": [ds["name"] for ds in selected]
        }
    
    def _rank_datasets(self, datasets, metrics, dimensions):
        """Rank datasets by relevance"""
        for ds in datasets:
            score = 0
            
            # Column coverage (40%)
            coverage = len(ds["matched_columns"]) / (len(metrics) + len(dimensions))
            score += coverage * 0.4
            
            # Quality score (30%)
            score += ds["quality_score"] * 0.3
            
            # Freshness (20%)
            freshness_score = self._calculate_freshness_score(ds["freshness"])
            score += freshness_score * 0.2
            
            # Popularity (10%)
            score += min(ds["usage_count"] / 100, 1.0) * 0.1
            
            ds["relevance_score"] = score
        
        return sorted(datasets, key=lambda x: x["relevance_score"], reverse=True)
```

**Deliverables**:
- ✅ Discovery agent class
- ✅ Dataset ranking algorithm
- ✅ Semantic search integration
- ✅ Test suite

**Success Criteria**:
- Finds correct datasets for all 4 use cases
- Relevance score > 0.7 for primary datasets
- Query time < 500ms

---

#### Day 4-5: Modeling Agent

**Tasks**:
- [ ] Implement schema design logic
- [ ] Add grain determination
- [ ] Add primary key inference
- [ ] Add data type mapping
- [ ] Add constraint generation
- [ ] Test with use cases

**Code Structure**:
```python
# agents/modeling_agent.py

class ModelingAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    def design_schema(self, intent: Dict, datasets: List[Dict]) -> Dict:
        # Step 1: Determine grain
        grain = self._determine_grain(
            intent["temporal_granularity"],
            intent["dimensions"]
        )
        
        # Step 2: Design target schema
        prompt = self._build_schema_prompt(intent, datasets, grain)
        schema = self.llm.generate_structured_output(prompt, schema_format)
        
        # Step 3: Infer primary keys
        primary_keys = self._infer_primary_keys(grain, intent["dimensions"])
        
        # Step 4: Add constraints
        schema_with_constraints = self._add_constraints(schema)
        
        return {
            "target_table": self._generate_table_name(intent),
            "grain": grain,
            "schema": schema_with_constraints,
            "primary_keys": primary_keys
        }
    
    def _determine_grain(self, temporal, dimensions):
        """Determine data granularity"""
        grain_parts = []
        
        if temporal:
            grain_parts.append(temporal.capitalize())
        
        if dimensions:
            grain_parts.append(f"by {', '.join(dimensions)}")
        
        return ", ".join(grain_parts)
```

**Deliverables**:
- ✅ Modeling agent class
- ✅ Schema generation logic
- ✅ Grain determination
- ✅ Test suite

**Success Criteria**:
- Generates correct schema for all 4 use cases
- Primary keys match grain
- Data types are appropriate

---

### Week 5: Integration & Testing

#### Day 1-3: Agent Integration

**Tasks**:
- [ ] Integrate Intent → Discovery → Modeling flow
- [ ] Test end-to-end with Orchestrator
- [ ] Add inter-agent validation
- [ ] Implement feedback loops
- [ ] Add comprehensive logging

**Test Flow**:
```python
# Test end-to-end flow
orchestrator = OrchestratorAgent()

request = "I need daily sales analytics by region and category"

result = orchestrator.run(request)

assert "business_metrics" in result
assert "selected_datasets" in result
assert "target_schema" in result
assert len(result["errors"]) == 0
```

**Deliverables**:
- ✅ Integrated agent flow
- ✅ End-to-end tests
- ✅ Logging infrastructure

**Success Criteria**:
- All 3 agents execute successfully
- State flows correctly between agents
- No data loss between steps

---

#### Day 4-5: Performance Optimization

**Tasks**:
- [ ] Profile agent execution times
- [ ] Optimize KG queries
- [ ] Add caching for LLM calls
- [ ] Parallelize independent operations
- [ ] Set performance baselines

**Target Metrics**:
- Intent Agent: < 2 seconds
- Discovery Agent: < 3 seconds
- Modeling Agent: < 2 seconds
- Total (3 agents): < 7 seconds

**Deliverables**:
- ✅ Performance benchmarks
- ✅ Optimization report
- ✅ Caching layer

**Success Criteria**:
- Meet or exceed target metrics
- 90th percentile < 10 seconds

---

### Week 3-5 Checkpoint

**Completed**:
- ✅ Orchestrator with workflow management
- ✅ Intent Agent with LLM integration
- ✅ Discovery Agent with KG queries
- ✅ Modeling Agent with schema design

**Metrics**:
- End-to-end execution time: < 10 seconds
- Intent accuracy: > 85%
- Dataset discovery accuracy: > 90%

**Ready for**: Transformation & Quality agents

---

## Phase 3: Transformation & Quality (Weeks 6-7)

### Week 6: Transformation Agent

#### Day 1-3: SQL Generation

**Tasks**:
- [ ] Implement SQL template engine
- [ ] Add LLM-based SQL generation
- [ ] Add SQL validation (sqlparse)
- [ ] Implement join logic generation
- [ ] Add aggregation logic
- [ ] Test with DuckDB

**Code Structure**:
```python
# agents/transformation_agent.py

class TransformationAgent:
    def __init__(self, llm_client: LLMClient, db_client: DuckDBClient):
        self.llm = llm_client
        self.db = db_client
    
    def generate_transformation(self, intent: Dict, datasets: List[Dict], 
                                target_schema: Dict) -> Dict:
        # Step 1: Determine if template or LLM generation
        if self._is_simple_aggregation(intent):
            sql = self._generate_from_template(intent, datasets, target_schema)
        else:
            sql = self._generate_with_llm(intent, datasets, target_schema)
        
        # Step 2: Validate SQL
        is_valid, error = self._validate_sql(sql)
        if not is_valid:
            # Retry with error feedback
            sql = self._regenerate_with_feedback(sql, error)
        
        # Step 3: Test execution
        is_executable, result = self._test_execution(sql)
        
        return {
            "language": "SQL",
            "code": sql,
            "is_valid": is_executable,
            "estimated_rows": len(result) if is_executable else None
        }
    
    def _generate_with_llm(self, intent, datasets, target_schema):
        prompt = f"""
Generate a SQL query to create the following data product:

Source Tables:
{self._format_source_tables(datasets)}

Target Schema:
{self._format_target_schema(target_schema)}

Business Logic:
- Metrics: {intent['business_metrics']}
- Dimensions: {intent['dimensions']}
- Granularity: {intent['temporal_granularity']}

Requirements:
1. Join tables on appropriate keys
2. Apply aggregations for metrics
3. Group by dimensions
4. Add filters if needed
5. Ensure output matches target schema

Generate clean, optimized SQL.
"""
        
        return self.llm.generate_code(prompt, language="SQL")
```

**Test Cases**:
```sql
-- Test 1: A1 - Daily Sales Analytics
SELECT
    DATE(o.order_date) AS date,
    o.region,
    p.category,
    SUM(o.total_amount) AS total_revenue,
    COUNT(DISTINCT o.order_id) AS order_count,
    SUM(o.quantity) AS units_sold
FROM bronze_orders o
JOIN bronze_products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY DATE(o.order_date), o.region, p.category
```

**Deliverables**:
- ✅ Transformation agent class
- ✅ SQL generation (template + LLM)
- ✅ SQL validation
- ✅ Test suite with 4 use cases

**Success Criteria**:
- Generates valid SQL for all 4 use cases
- SQL executes successfully on bronze data
- Output schema matches target schema

---

#### Day 4-5: Alternative Code Generation (PySpark)

**Tasks**:
- [ ] Implement PySpark code generation
- [ ] Add language selection logic
- [ ] Test with sample data
- [ ] Compare SQL vs PySpark performance

**Code Example**:
```python
# PySpark generation for A1
code = """
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
)
"""
```

**Deliverables**:
- ✅ PySpark generation
- ✅ Language selection logic
- ✅ Performance comparison

**Success Criteria**:
- Can generate both SQL and PySpark
- Both produce same results
- Language selection based on data size

---

### Week 7: Quality Agent & Packaging

#### Day 1-3: Quality Agent

**Tasks**:
- [ ] Implement quality rule generation
- [ ] Add schema-based rules (automatic)
- [ ] Add domain-based rules (from KG)
- [ ] Add business logic rules (LLM)
- [ ] Generate Great Expectations suite
- [ ] Generate dbt tests
- [ ] Define SLA requirements

**Code Structure**:
```python
# agents/quality_agent.py

class QualityAgent:
    def __init__(self, llm_client: LLMClient, kg_client: KnowledgeGraphClient):
        self.llm = llm_client
        self.kg = kg_client
    
    def generate_quality_rules(self, target_schema: Dict, intent: Dict) -> Dict:
        rules = []
        
        # Step 1: Schema-based rules
        rules.extend(self._generate_schema_rules(target_schema))
        
        # Step 2: Domain-based rules from KG
        rules.extend(self._get_kg_rules(target_schema["target_table"]))
        
        # Step 3: Business logic rules from LLM
        rules.extend(self._generate_business_rules(intent, target_schema))
        
        # Step 4: Generate test code
        ge_suite = self._generate_ge_suite(rules)
        dbt_tests = self._generate_dbt_tests(rules)
        
        # Step 5: Define SLA
        sla = self._define_sla(intent)
        
        return {
            "quality_rules": rules,
            "great_expectations_suite": ge_suite,
            "dbt_tests": dbt_tests,
            "sla": sla
        }
    
    def _generate_schema_rules(self, schema):
        """Generate rules from schema constraints"""
        rules = []
        
        for column in schema["schema"]:
            # NOT NULL constraint
            if not column.get("nullable", True):
                rules.append({
                    "rule_type": "completeness",
                    "column": column["name"],
                    "condition": f"{column['name']} IS NOT NULL",
                    "severity": "error"
                })
            
            # Primary key uniqueness
            if column.get("primary_key", False):
                rules.append({
                    "rule_type": "uniqueness",
                    "column": column["name"],
                    "condition": f"DISTINCT {column['name']}",
                    "severity": "error"
                })
        
        return rules
```

**Deliverables**:
- ✅ Quality agent class
- ✅ Rule generation (3 types)
- ✅ Great Expectations integration
- ✅ SLA definition

**Success Criteria**:
- Generates 5+ rules per use case
- Rules are executable
- SLA requirements are realistic

---

#### Day 4-5: Packaging Agent

**Tasks**:
- [ ] Implement spec compilation
- [ ] Add metadata generation
- [ ] Add lineage graph generation
- [ ] Implement YAML export
- [ ] Add spec validation
- [ ] Test with all 4 use cases

**Code Structure**:
```python
# agents/packaging_agent.py

class PackagingAgent:
    def __init__(self, schema_validator):
        self.validator = schema_validator
    
    def package(self, state: DataProductState) -> Dict:
        # Step 1: Compile all outputs
        spec = {
            "metadata": self._generate_metadata(state),
            "business_context": self._extract_business_context(state),
            "data_model": state["target_schema"],
            "source_datasets": state["selected_datasets"],
            "transformations": {
                "language": state["transformation_language"],
                "code": state["transformation_code"]
            },
            "quality_rules": state["quality_rules"],
            "sla": state["sla_requirements"],
            "lineage": self._generate_lineage(state)
        }
        
        # Step 2: Validate against schema
        is_valid, errors = self.validator.validate(spec)
        
        if not is_valid:
            raise ValidationError(f"Invalid spec: {errors}")
        
        # Step 3: Export to YAML
        yaml_output = self._export_to_yaml(spec)
        
        return {
            "data_product_spec": spec,
            "yaml_output": yaml_output,
            "is_valid": is_valid
        }
```

**Deliverables**:
- ✅ Packaging agent class
- ✅ YAML export
- ✅ Lineage generation
- ✅ Validation

**Success Criteria**:
- Generates complete spec for all 4 use cases
- All specs pass validation
- YAML is well-formatted

---

### Week 6-7 Checkpoint

**Completed**:
- ✅ Transformation Agent (SQL + PySpark)
- ✅ Quality Agent (rules + tests)
- ✅ Packaging Agent (YAML export)

**Metrics**:
- SQL generation success rate: > 90%
- Quality rules per use case: 5-10
- Spec validation pass rate: 100%

**Ready for**: Use case testing

---

## Phase 4: Use Case Testing (Weeks 8-9)

### Week 8: Use Cases A1 & A2

#### Day 1-2: A1 - Daily Sales Analytics

**Tasks**:
- [ ] Run end-to-end with A1 request
- [ ] Validate generated SQL
- [ ] Execute transformation on bronze data
- [ ] Validate output data
- [ ] Check quality rules
- [ ] Review generated spec
- [ ] Document issues and fixes

**Test Request**:
```
"I need a daily sales analytics data product showing revenue, order count, 
and units sold, broken down by region and product category."
```

**Expected Output**:
- Target table: `gold.daily_sales_analytics`
- Columns: date, region, category, total_revenue, order_count, units_sold
- Rows: ~30,000 (1,050 days × ~30 region-category combinations)

**Validation Checks**:
- [ ] All dates present (2023-01-01 to 2025-11-15)
- [ ] Revenue > 0 for all rows
- [ ] Order count >= units sold (warning if violated)
- [ ] No null values in key columns

**Deliverables**:
- ✅ A1 test results
- ✅ Generated Data Product spec (YAML)
- ✅ Execution report
- ✅ Issue log

---

#### Day 3-4: A2 - Marketing Campaign Performance

**Tasks**:
- [ ] Run end-to-end with A2 request
- [ ] Validate CTR, CVR, CPA, ROAS calculations
- [ ] Execute transformation
- [ ] Validate output data
- [ ] Check quality rules
- [ ] Review spec

**Test Request**:
```
"Create a weekly marketing performance report with CTR, CVR, CPA, and ROAS 
for each campaign."
```

**Expected Output**:
- Target table: `gold.marketing_campaign_performance`
- Columns: week, campaign_id, campaign_name, impressions, clicks, conversions, spend, revenue, ctr, cvr, cpa, roas
- Rows: ~500 (50 campaigns × ~10 weeks average)

**Validation Checks**:
- [ ] CTR = clicks / impressions
- [ ] CVR = conversions / clicks
- [ ] CPA = spend / conversions
- [ ] ROAS = revenue / spend
- [ ] Impressions >= clicks >= conversions

**Deliverables**:
- ✅ A2 test results
- ✅ Generated spec
- ✅ Execution report

---

#### Day 5: Refinement

**Tasks**:
- [ ] Review issues from A1 & A2
- [ ] Fix agent bugs
- [ ] Improve prompts
- [ ] Optimize queries
- [ ] Re-test

---

### Week 9: Use Cases B3 & C2

#### Day 1-2: B3 - Product Recommendation Features

**Tasks**:
- [ ] Run end-to-end with B3 request
- [ ] Validate feature engineering logic
- [ ] Execute transformation
- [ ] Validate output data
- [ ] Check quality rules
- [ ] Review spec

**Test Request**:
```
"Build a feature table for product recommendations based on user interaction 
signals including views, cart adds, purchases, and ratings."
```

**Expected Output**:
- Target table: `gold.product_recommendation_features`
- Columns: user_id, product_id, view_count, cart_count, purchase_count, avg_rating, days_since_last_interaction, category_affinity_score
- Rows: ~50,000 (user-product pairs with interactions)

**Validation Checks**:
- [ ] All counts >= 0
- [ ] Ratings between 1-5 (if not null)
- [ ] days_since_last_interaction >= 0
- [ ] category_affinity_score between 0-1

**Deliverables**:
- ✅ B3 test results
- ✅ Generated spec
- ✅ Execution report

---

#### Day 3-4: C2 - Customer 360

**Tasks**:
- [ ] Run end-to-end with C2 request
- [ ] Validate customer aggregations
- [ ] Execute transformation
- [ ] Validate output data
- [ ] Check quality rules
- [ ] Review spec

**Test Request**:
```
"Create a unified customer 360 view combining profile, transaction history, 
loyalty status, and support interactions."
```

**Expected Output**:
- Target table: `gold.customer_360`
- Columns: customer_id, name, email, signup_date, loyalty_tier, total_orders, total_revenue, avg_order_value, last_order_date, open_tickets, avg_satisfaction_score, segment
- Rows: 10,000 (one per customer)

**Validation Checks**:
- [ ] Unique customer_id
- [ ] Email format validation
- [ ] total_revenue = SUM(orders.total_amount)
- [ ] avg_order_value = total_revenue / total_orders
- [ ] avg_satisfaction_score between 1-5

**Deliverables**:
- ✅ C2 test results
- ✅ Generated spec
- ✅ Execution report

---

#### Day 5: Final Refinement

**Tasks**:
- [ ] Review all 4 use cases
- [ ] Compare results
- [ ] Identify common issues
- [ ] Final bug fixes
- [ ] Performance tuning
- [ ] Documentation updates

---

### Week 8-9 Checkpoint

**Completed**:
- ✅ All 4 use cases tested end-to-end
- ✅ Data Product specs generated for each
- ✅ Transformations executed successfully
- ✅ Quality rules validated

**Metrics**:
- Use case success rate: 100% (4/4)
- Average execution time: < 15 seconds
- Spec completeness: > 95%

**Ready for**: UI development

---

## Phase 5: UI & Integration (Week 10)

### Week 10: User Interface & API

#### Day 1-2: Streamlit UI

**Tasks**:
- [ ] Create Streamlit app structure
- [ ] Add input form for user request
- [ ] Add use case selector (A1, A2, B3, C2)
- [ ] Display execution progress
- [ ] Show generated Data Product spec
- [ ] Add YAML download button
- [ ] Add visualization of lineage graph
- [ ] Add execution logs viewer

**UI Structure**:
```python
# ui/streamlit_app.py

import streamlit as st
from agents.orchestrator import OrchestratorAgent

st.title("🤖 Agentic Data Product Builder")

# Sidebar
use_case = st.sidebar.selectbox(
    "Select Use Case",
    ["Custom", "A1: Daily Sales", "A2: Marketing Performance", 
     "B3: Recommendations", "C2: Customer 360"]
)

if use_case != "Custom":
    user_request = st.sidebar.text_area(
        "Request (pre-filled)",
        value=get_use_case_template(use_case)
    )
else:
    user_request = st.text_area("Enter your data product request:")

if st.button("Generate Data Product"):
    with st.spinner("Generating..."):
        orchestrator = OrchestratorAgent()
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Execute
        result = orchestrator.run(user_request)
        
        # Display results
        st.success("Data Product Generated!")
        
        # Show spec
        st.subheader("Data Product Specification")
        st.yaml(result["yaml_output"])
        
        # Download button
        st.download_button(
            "Download YAML",
            data=result["yaml_output"],
            file_name="data_product.yaml"
        )
        
        # Show lineage
        st.subheader("Data Lineage")
        st.graphviz_chart(generate_lineage_graph(result["lineage"]))
        
        # Show logs
        with st.expander("Execution Logs"):
            st.json(result["execution_log"])
```

**Deliverables**:
- ✅ Streamlit app
- ✅ Input form
- ✅ Spec viewer
- ✅ Lineage visualization
- ✅ Download functionality

**Success Criteria**:
- UI loads without errors
- Can generate specs for all 4 use cases
- YAML downloads correctly
- Lineage graph renders

---

#### Day 3-4: REST API

**Tasks**:
- [ ] Create FastAPI server
- [ ] Add `/generate` endpoint
- [ ] Add `/datasets` endpoint
- [ ] Add `/business-terms` endpoint
- [ ] Add authentication (API key)
- [ ] Add rate limiting
- [ ] Write API documentation (OpenAPI)
- [ ] Test with Postman/curl

**API Structure**:
```python
# api/fastapi_server.py

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel

app = FastAPI(
    title="Agentic Data Product Builder API",
    version="1.0.0"
)

class DataProductRequest(BaseModel):
    user_request: str
    use_case_id: Optional[str] = None

class DataProductResponse(BaseModel):
    data_product_spec: Dict
    execution_time_ms: int
    agent_logs: List[Dict]

@app.post("/api/v1/generate", response_model=DataProductResponse)
async def generate_data_product(
    request: DataProductRequest,
    api_key: str = Depends(verify_api_key)
):
    """Generate data product from natural language request"""
    start_time = time.time()
    
    orchestrator = OrchestratorAgent()
    result = orchestrator.run(request.user_request)
    
    execution_time = int((time.time() - start_time) * 1000)
    
    return DataProductResponse(
        data_product_spec=result["data_product_spec"],
        execution_time_ms=execution_time,
        agent_logs=result["execution_log"]
    )

@app.get("/api/v1/datasets")
async def list_datasets():
    """List available bronze datasets"""
    kg_client = KnowledgeGraphClient()
    datasets = kg_client.get_all_datasets()
    return {"datasets": datasets}

@app.get("/api/v1/health")
async def health_check():
    return {"status": "healthy"}
```

**Deliverables**:
- ✅ FastAPI server
- ✅ 3+ endpoints
- ✅ API documentation
- ✅ Authentication

**Success Criteria**:
- API responds to requests
- OpenAPI docs accessible at `/docs`
- Can generate specs via API
- Authentication works

---

#### Day 5: Integration Testing

**Tasks**:
- [ ] Test UI → Orchestrator integration
- [ ] Test API → Orchestrator integration
- [ ] Load testing (10 concurrent requests)
- [ ] Error handling testing
- [ ] Documentation updates

**Deliverables**:
- ✅ Integration test suite
- ✅ Load test results
- ✅ User guide

**Success Criteria**:
- UI and API both work
- Can handle 10 concurrent requests
- Error messages are clear

---

### Week 10 Checkpoint

**Completed**:
- ✅ Streamlit UI
- ✅ REST API
- ✅ Integration tests

**Metrics**:
- UI response time: < 20 seconds
- API throughput: 10 req/min
- User satisfaction: Qualitative feedback

**Ready for**: Evaluation

---

## Phase 6: Evaluation & Documentation (Weeks 11-12)

### Week 11: Evaluation

#### Day 1-2: Quantitative Evaluation

**Tasks**:
- [ ] Define evaluation metrics
- [ ] Create benchmark dataset (20 requests)
- [ ] Run automated evaluation
- [ ] Calculate accuracy metrics
- [ ] Measure performance metrics
- [ ] Generate evaluation report

**Evaluation Metrics**:

1. **Accuracy**:
   - Dataset Discovery Accuracy: % of correct datasets identified
   - Schema Accuracy: % of correct columns in target schema
   - SQL Correctness: % of valid, executable SQL
   - Overall Accuracy: Weighted average

2. **Completeness**:
   - Spec Completeness: % of required fields populated
   - Quality Rule Coverage: Average # of rules per use case

3. **Performance**:
   - End-to-end execution time (p50, p90, p99)
   - Agent-level execution time breakdown
   - LLM token usage

**Benchmark Dataset**:
```python
benchmark_requests = [
    # A1 variants
    "Daily sales by region",
    "Weekly revenue by product category",
    "Monthly sales trends",
    
    # A2 variants
    "Campaign performance metrics",
    "Marketing ROI analysis",
    
    # B3 variants
    "Product recommendation features",
    "User-product affinity scores",
    
    # C2 variants
    "Customer 360 view",
    "Customer lifetime value analysis",
    
    # Edge cases
    "Show me all customer data",  # Ambiguous
    "Revenue",  # Too vague
    # ... 10 more
]
```

**Deliverables**:
- ✅ Evaluation framework
- ✅ Benchmark results
- ✅ Metrics report

**Success Criteria**:
- Dataset discovery accuracy > 85%
- SQL correctness > 90%
- Execution time p90 < 20 seconds

---

#### Day 3-4: Qualitative Evaluation

**Tasks**:
- [ ] Manual review of generated specs
- [ ] Code quality assessment
- [ ] Documentation quality assessment
- [ ] Usability testing (with 3-5 users)
- [ ] Collect feedback
- [ ] Identify improvement areas

**Evaluation Criteria**:
- SQL readability (1-5 scale)
- Spec clarity (1-5 scale)
- Usefulness of quality rules (1-5 scale)
- Overall satisfaction (1-5 scale)

**Deliverables**:
- ✅ Qualitative assessment report
- ✅ User feedback summary
- ✅ Improvement recommendations

---

#### Day 5: Comparison with Baselines

**Tasks**:
- [ ] Compare with manual approach (time, effort)
- [ ] Compare with template-based approach
- [ ] Analyze advantages and limitations
- [ ] Document findings

**Comparison Dimensions**:
- Time to generate Data Product spec
- Accuracy of generated artifacts
- Flexibility (handling variations)
- Scalability

**Deliverables**:
- ✅ Comparison report
- ✅ Advantages/limitations analysis

---

### Week 12: Documentation & Presentation

#### Day 1-3: Dissertation Writing

**Tasks**:
- [ ] Write abstract
- [ ] Write introduction
- [ ] Write literature review
- [ ] Write methodology chapter
- [ ] Write implementation chapter
- [ ] Write evaluation chapter
- [ ] Write conclusion
- [ ] Add figures and diagrams
- [ ] Format references

**Dissertation Structure**:
1. Abstract
2. Introduction
   - Problem statement
   - Research objectives
   - Contributions
3. Literature Review
   - Multi-agent systems
   - Knowledge graphs
   - Data products
   - Code generation
4. Methodology
   - System architecture
   - Agent design
   - Knowledge graph design
5. Implementation
   - Technology stack
   - Agent implementation
   - Use case implementation
6. Evaluation
   - Quantitative results
   - Qualitative results
   - Comparison with baselines
7. Conclusion
   - Summary
   - Limitations
   - Future work

**Deliverables**:
- ✅ Complete dissertation draft (40-60 pages)

---

#### Day 4: Demo Video & Presentation

**Tasks**:
- [ ] Record demo video (5-10 minutes)
- [ ] Create presentation slides (20-30 slides)
- [ ] Prepare speaker notes
- [ ] Practice presentation

**Demo Video Outline**:
1. Problem introduction (1 min)
2. System overview (1 min)
3. Live demo: A1 use case (2 min)
4. Live demo: C2 use case (2 min)
5. Architecture walkthrough (2 min)
6. Results and evaluation (1 min)
7. Conclusion (1 min)

**Deliverables**:
- ✅ Demo video
- ✅ Presentation slides

---

#### Day 5: Final Review & Submission

**Tasks**:
- [ ] Code cleanup
- [ ] Final testing
- [ ] README update
- [ ] Documentation review
- [ ] Dissertation final review
- [ ] Submit dissertation
- [ ] Prepare for defense

**Deliverables**:
- ✅ Clean codebase
- ✅ Complete documentation
- ✅ Final dissertation
- ✅ Presentation materials

---

### Week 11-12 Checkpoint

**Completed**:
- ✅ Quantitative evaluation
- ✅ Qualitative evaluation
- ✅ Dissertation written
- ✅ Demo video created
- ✅ Presentation prepared

**Metrics**:
- Dissertation: 40-60 pages
- Evaluation: 20 benchmark requests
- Demo video: 5-10 minutes

**Ready for**: Defense

---

## Risk Management

### High-Priority Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| LLM hallucination in SQL generation | High | High | Add validation layer, use templates for simple cases |
| Knowledge Graph complexity | Medium | Medium | Start with simple schema, iterate |
| Agent coordination issues | Medium | High | Use LangGraph for explicit state management |
| Performance bottlenecks | Medium | Medium | Profile early, optimize critical paths |
| Scope creep | High | High | Stick to 4 use cases, defer enhancements |

### Contingency Plans

**If SQL generation fails**:
- Fall back to template-based generation
- Use simpler prompts
- Add more few-shot examples

**If Knowledge Graph is too complex**:
- Use simpler in-memory graph (NetworkX)
- Reduce node types
- Focus on essential relationships

**If behind schedule**:
- Reduce use cases to 2 (A1, C2)
- Skip PySpark generation
- Simplify UI

---

## Success Criteria Summary

### Technical Success
- ✅ All 4 use cases working end-to-end
- ✅ Dataset discovery accuracy > 85%
- ✅ SQL correctness > 90%
- ✅ Execution time < 20 seconds (p90)
- ✅ Spec validation pass rate: 100%

### Research Success
- ✅ Novel approach to Data Product automation
- ✅ Knowledge Graph-driven agent coordination
- ✅ Publishable results
- ✅ Complete dissertation

### Deliverables Checklist
- ✅ Working prototype system
- ✅ Bronze layer with 7 datasets
- ✅ Knowledge Graph with metadata
- ✅ 6 specialized agents
- ✅ Streamlit UI
- ✅ REST API
- ✅ 4 Data Product specs (YAML)
- ✅ Evaluation report
- ✅ Dissertation (40-60 pages)
- ✅ Demo video
- ✅ Presentation slides

---

## Post-Submission (Optional)

### Future Enhancements
- [ ] Add support for more data sources (APIs, databases)
- [ ] Implement incremental load logic generation
- [ ] Add data catalog integration
- [ ] Support for real-time data products
- [ ] Multi-language support (beyond SQL/PySpark)
- [ ] Integration with dbt, Airflow
- [ ] Feedback loop for continuous improvement

### Publication Opportunities
- [ ] Submit to IEEE BigData conference
- [ ] Submit to VLDB demo track
- [ ] Write blog post
- [ ] Open-source the project

---

## Weekly Time Allocation

| Week | Hours | Focus |
|------|-------|-------|
| 1-2 | 40 | Foundation |
| 3-5 | 60 | Core Agents |
| 6-7 | 40 | Transformation & Quality |
| 8-9 | 40 | Use Case Testing |
| 10 | 20 | UI & API |
| 11-12 | 40 | Evaluation & Documentation |
| **Total** | **240 hours** | **12 weeks** |

---

## Summary

This roadmap provides:

1. **Clear Milestones**: Weekly checkpoints with deliverables
2. **Incremental Development**: Build and test iteratively
3. **Risk Management**: Contingency plans for common issues
4. **Realistic Timeline**: 12 weeks with buffer time
5. **Success Criteria**: Measurable goals at each phase

**Next Step**: Begin Phase 1, Day 1 - Environment Setup when ready to implement.

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-15  
**Status**: Ready for Implementation

