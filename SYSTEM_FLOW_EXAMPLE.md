# System Flow Example: Marketing Performance Report

## User Request

> "Create a weekly marketing performance report with CTR, CVR, CPA, and ROAS for each campaign."

---

## Backend Processing Flow

### Step 1: Request Reception
**Entry Point**: User submits request via Streamlit UI or REST API

**What Happens**:
- Request is received by the system
- Initial validation (non-empty string)
- Request is passed to the Orchestrator Agent
- A new workflow session is created with unique ID
- Initial state object is initialized with the user request

---

### Step 2: Orchestrator Initialization
**Component**: Orchestrator Agent

**What Happens**:
- Orchestrator creates a LangGraph StateGraph instance
- Initializes the workflow state with empty fields for each agent's output
- Sets up the execution pipeline: Intent → Discovery → Modeling → Transformation → Quality → Packaging
- Starts logging execution metadata (timestamp, request ID)
- Invokes the first agent in the chain: Intent Agent

---

### Step 3: Intent Analysis
**Component**: Intent Agent

**What Happens**:
- Receives the raw user request
- Constructs an LLM prompt asking to extract:
  - Business metrics mentioned (CTR, CVR, CPA, ROAS)
  - Dimensions for grouping (campaign)
  - Time granularity (weekly)
  - Any filters or conditions
- Sends prompt to OpenAI GPT-4
- Receives structured JSON response from LLM
- Queries Knowledge Graph to map business terms:
  - "CTR" → Click-Through Rate (clicks/impressions)
  - "CVR" → Conversion Rate (conversions/clicks)
  - "CPA" → Cost Per Acquisition (spend/conversions)
  - "ROAS" → Return on Ad Spend (revenue/spend)
- Calculates confidence score based on term matches
- Updates state with extracted intent
- Logs execution time and passes control to Discovery Agent

**State Updated With**:
- business_metrics: ["CTR", "CVR", "CPA", "ROAS"]
- dimensions: ["campaign", "week"]
- temporal_granularity: "weekly"
- business_terms: [mapped glossary entries]
- confidence_score: 0.92

---

### Step 4: Dataset Discovery
**Component**: Discovery Agent

**What Happens**:
- Receives business metrics and dimensions from state
- Performs parallel searches in Knowledge Graph:
  
  **Search 1: Term-Based Lookup**
  - Queries for datasets containing columns mapped to business terms
  - Finds: marketing_campaigns, marketing_events tables
  - Retrieves column metadata for each dataset
  
  **Search 2: Semantic Search**
  - Generates embedding for query: "CTR CVR CPA ROAS campaign weekly"
  - Performs vector similarity search against column embeddings
  - Finds semantically similar columns (impressions, clicks, conversions, spend, revenue)
  
- Merges results from both searches
- Ranks datasets by relevance score:
  - Column coverage: Does dataset have required columns? (40% weight)
  - Quality score: Historical data quality metrics (30% weight)
  - Freshness: How often is data updated? (20% weight)
  - Popularity: Usage frequency by other data products (10% weight)
- Selects top-ranked datasets
- Retrieves detailed metadata for selected datasets (schema, row counts, sample values)
- Checks access permissions for current user
- Updates state with candidate and selected datasets
- Passes control to Modeling Agent

**State Updated With**:
- candidate_datasets: [list of 5 ranked datasets]
- selected_datasets: ["bronze.marketing_campaigns", "bronze.marketing_events"]
- dataset_metadata: [detailed schema information]

---

### Step 5: Data Model Design
**Component**: Modeling Agent

**What Happens**:
- Receives intent and selected datasets from state
- Determines data granularity:
  - Temporal: Weekly
  - Dimensions: Campaign
  - Grain: "Weekly, by campaign"
- Constructs LLM prompt with:
  - Business requirements (metrics to calculate)
  - Source dataset schemas
  - Target granularity
- Sends prompt to GPT-4 asking to design target schema
- Receives schema design from LLM
- Infers primary keys based on grain: [week, campaign_id]
- Maps data types:
  - Metrics (CTR, CVR, CPA, ROAS) → DECIMAL
  - Dimensions (campaign_id) → VARCHAR
  - Time (week) → DATE
- Adds constraints:
  - NOT NULL for primary keys
  - Check constraints (impressions >= clicks >= conversions)
  - Range constraints (CTR between 0 and 1)
- Generates target table name: "gold.marketing_campaign_performance"
- Validates schema completeness
- Updates state with target schema
- Passes control to Transformation Agent

**State Updated With**:
- target_table: "gold.marketing_campaign_performance"
- grain: "Weekly, by campaign"
- target_schema: [list of columns with types and constraints]
- primary_keys: ["week", "campaign_id"]

---

### Step 6: Transformation Code Generation
**Component**: Transformation Agent

**What Happens**:
- Receives source datasets, target schema, and business logic from state
- Determines complexity of transformation:
  - Requires joins? Yes (campaigns + events)
  - Requires calculations? Yes (CTR, CVR, CPA, ROAS)
  - Complexity level: Medium
- Decides generation strategy: LLM-based (too complex for templates)
- Constructs detailed prompt with:
  - Source table schemas
  - Target schema
  - Business metric definitions
  - Join conditions
  - Aggregation requirements
- Sends prompt to GPT-4 requesting SQL code
- Receives SQL query from LLM
- Validates SQL syntax using sqlparse library
- If validation fails, regenerates with error feedback
- Tests SQL execution on bronze data using DuckDB:
  - Registers bronze tables as views
  - Executes generated SQL
  - Checks if output schema matches target schema
  - Validates row count is reasonable
- If execution fails, attempts retry with error context
- Once successful, captures execution statistics
- Updates state with transformation code
- Passes control to Quality Agent

**State Updated With**:
- transformation_language: "SQL"
- transformation_code: [complete SQL query]
- is_valid: true
- estimated_rows: 450

**Generated SQL** (conceptually):
- SELECT week, campaign info, aggregated metrics
- FROM marketing_events joined with marketing_campaigns
- GROUP BY week, campaign
- Calculate CTR, CVR, CPA, ROAS in SELECT clause

---

### Step 7: Quality Rule Generation
**Component**: Quality Agent

**What Happens**:
- Receives target schema and business context from state
- Generates quality rules from three sources:

  **Source 1: Schema-Based Rules (Automatic)**
  - NOT NULL constraints → completeness checks
  - Primary key columns → uniqueness checks
  - Data types → type validation checks
  
  **Source 2: Knowledge Graph Rules (Historical)**
  - Queries KG for existing quality rules on similar datasets
  - Finds rules like "impressions > 0", "spend >= 0"
  - Adapts rules to current context
  
  **Source 3: Business Logic Rules (LLM-Generated)**
  - Constructs prompt asking for domain-specific rules
  - LLM suggests rules based on business logic:
    - Funnel consistency: impressions >= clicks >= conversions
    - Metric ranges: CTR between 0 and 1
    - Relationship checks: revenue >= 0 when conversions > 0
  
- Assigns severity levels (error vs warning) to each rule
- Generates executable test code:
  - Great Expectations suite format
  - dbt test format
- Defines SLA requirements:
  - Freshness: Weekly, Monday 9 AM
  - Latency: < 1 hour from source data availability
  - Completeness: > 99%
  - Accuracy: > 95%
- Updates state with quality rules and SLA
- Passes control to Packaging Agent

**State Updated With**:
- quality_rules: [list of 8-10 rules with conditions and severity]
- great_expectations_suite: [GE configuration]
- dbt_tests: [dbt test definitions]
- sla_requirements: [freshness, latency, completeness targets]

---

### Step 8: Specification Packaging
**Component**: Packaging Agent

**What Happens**:
- Receives all outputs from previous agents via state
- Compiles complete Data Product specification:
  
  **Metadata Section**:
  - Generates name from intent: "marketing_campaign_performance"
  - Sets version: "1.0.0"
  - Creates description from user request
  - Assigns owner (from user context or default)
  - Adds timestamp and tags
  
  **Business Context Section**:
  - Extracts use case identifier: "A2"
  - Lists business metrics
  - Identifies stakeholders (marketing team)
  
  **Data Model Section**:
  - Includes target schema from Modeling Agent
  - Adds grain definition
  - Lists primary keys and constraints
  
  **Source Datasets Section**:
  - Lists selected datasets with columns used
  
  **Transformations Section**:
  - Includes SQL code from Transformation Agent
  - Adds language identifier
  
  **Quality Rules Section**:
  - Includes all generated rules
  - Adds test code
  
  **SLA Section**:
  - Includes freshness, latency, completeness requirements
  
  **Lineage Section**:
  - Generates lineage graph:
    - Upstream: bronze.marketing_campaigns, bronze.marketing_events
    - Downstream: (empty for now, could be dashboards)
    - Transformation type: aggregation + join

- Validates compiled spec against JSON Schema
- If validation fails, identifies missing fields and attempts to fill them
- Converts spec to YAML format for human readability
- Generates documentation (markdown format)
- Updates state with final specification
- Marks workflow as complete

**State Updated With**:
- data_product_spec: [complete specification dictionary]
- yaml_output: [formatted YAML string]
- is_valid: true
- lineage: [lineage graph structure]

---

### Step 9: Response Preparation
**Component**: Orchestrator Agent (Final Step)

**What Happens**:
- Workflow execution completes
- Orchestrator collects final state
- Compiles execution log:
  - Each agent's execution time
  - Total end-to-end time
  - Any warnings or errors encountered
  - LLM token usage
  - Knowledge Graph query count
- Prepares response payload:
  - Data Product specification (YAML)
  - Execution metadata
  - Agent logs
  - Confidence scores
- Returns response to presentation layer

---

### Step 10: Response Delivery
**Component**: Presentation Layer (UI/API)

**What Happens**:

**If via Streamlit UI**:
- Displays success message
- Renders YAML specification in formatted view
- Shows syntax-highlighted SQL code
- Displays lineage graph visualization
- Provides download button for YAML file
- Shows execution time and metrics
- Displays agent execution log in expandable section

**If via REST API**:
- Returns HTTP 200 response
- JSON payload containing:
  - data_product_spec (nested JSON)
  - execution_time_ms
  - agent_logs
  - yaml_output (as string)
- Sets appropriate headers

---

## Summary of Data Flow

```
User Request
    ↓
[Orchestrator] Creates workflow state
    ↓
[Intent Agent] Extracts: CTR, CVR, CPA, ROAS, weekly, campaign
    ↓
[Discovery Agent] Finds: marketing_campaigns, marketing_events tables
    ↓
[Modeling Agent] Designs: 12-column schema with calculated metrics
    ↓
[Transformation Agent] Generates: SQL with JOIN, GROUP BY, calculations
    ↓
[Quality Agent] Defines: 8 rules + SLA (weekly, <1hr latency)
    ↓
[Packaging Agent] Compiles: Complete YAML specification
    ↓
[Orchestrator] Returns: Final spec + metadata
    ↓
User receives complete Data Product specification
```

---

## Execution Metrics (Estimated)

- **Total Time**: 12-18 seconds
  - Intent Agent: 2 seconds (LLM call)
  - Discovery Agent: 3 seconds (KG queries + embedding search)
  - Modeling Agent: 2 seconds (LLM call)
  - Transformation Agent: 4 seconds (LLM call + SQL validation + test execution)
  - Quality Agent: 2 seconds (rule generation)
  - Packaging Agent: 1 second (compilation + validation)

- **LLM Calls**: 4 total
  - Intent extraction
  - Schema design
  - SQL generation
  - Quality rule generation

- **Knowledge Graph Queries**: 5-7
  - Business term lookup
  - Dataset search by terms
  - Column embedding search
  - Dataset metadata retrieval
  - Historical quality rules lookup

- **Database Operations**: 1
  - SQL test execution on bronze data

---

## Key Decision Points

### 1. **Intent Agent**: Recognizes Calculated Metrics
- Understands CTR, CVR, CPA, ROAS are not raw columns but need calculation
- Maps to underlying data: impressions, clicks, conversions, spend, revenue

### 2. **Discovery Agent**: Finds Two Related Tables
- Identifies need for both campaigns (metadata) and events (metrics)
- Understands they need to be joined

### 3. **Modeling Agent**: Designs Denormalized Schema
- Creates wide table with both dimensions and metrics
- Includes both raw counts and calculated ratios

### 4. **Transformation Agent**: Generates Complex SQL
- Implements JOIN between two tables
- Adds GROUP BY for weekly aggregation
- Calculates derived metrics in SELECT clause
- Handles division by zero (clicks=0 for CVR)

### 5. **Quality Agent**: Enforces Funnel Logic
- Recognizes marketing funnel: impressions → clicks → conversions
- Creates consistency rules across metrics

---

## Error Handling Examples

### If SQL Generation Fails:
1. Transformation Agent detects syntax error
2. Extracts error message from sqlparse
3. Constructs new prompt with error feedback
4. Retries generation (up to 3 attempts)
5. If still fails, falls back to template-based generation
6. If template unavailable, returns error to user with explanation

### If Dataset Not Found:
1. Discovery Agent returns empty candidate list
2. Orchestrator detects missing datasets
3. Intent Agent is re-invoked with clarification request
4. User is prompted: "Could not find datasets for marketing metrics. Please specify data sources."

### If Validation Fails:
1. Packaging Agent detects schema validation error
2. Identifies missing required fields
3. Attempts to infer missing values from context
4. If unable to infer, marks spec as incomplete
5. Returns spec with warnings about missing fields

---

## State Persistence

Throughout execution, the workflow state is maintained in memory and includes:
- All agent inputs and outputs
- Intermediate results
- Error messages and warnings
- Execution timestamps
- Confidence scores
- Retry attempts

This allows for:
- Debugging failed workflows
- Auditing decisions
- Explaining results to users
- Recovering from partial failures

---

**End of Flow**

The final YAML specification is now ready to be:
- Reviewed by data engineers
- Deployed to production
- Used to generate actual data pipelines
- Integrated with orchestration tools (Airflow, dbt)
- Cataloged in data catalog systems

