# Technology Stack by Flow

## Overview

This document maps which technologies are used in each of the two main system flows.

---

## 📊 Flow 1: Data Ingestion & Processing Flow (Offline/Backend)

**Purpose:** One-time setup of Bronze layer and Knowledge Graph metadata

**When:** Executed once during system initialization or when adding new datasets

**Duration:** ~5-10 minutes (one-time)

### Technology Stack

| Component | Technologies | Role |
|-----------|-------------|------|
| **Data Generation** | Python, Faker, NumPy, Pandas | Generate synthetic datasets with realistic patterns |
| **Bronze Storage** | Parquet, PyArrow, File System | Store raw data in columnar format |
| **Metadata Extraction** | Pandas, OpenAI Embeddings API, JSON | Extract schema and generate semantic embeddings |
| **Knowledge Graph** | Neo4j, Cypher Query Language | Store and query metadata relationships |
| **Query Layer** | DuckDB, SQL | Validate data and test transformations |
| **Transformation** | GPT-4, sqlparse, PySpark (optional) | Generate and validate transformation code |
| **Gold Products** | YAML, JSON Schema, Jinja2 | Package final specifications |

### Core Dependencies
```python
# requirements.txt (Flow 1)
faker>=20.0.0              # Synthetic data generation
numpy>=1.24.0              # Statistical distributions
pandas>=2.0.0              # Data manipulation
pyarrow>=12.0.0            # Parquet file I/O
neo4j>=5.12.0              # Graph database driver
openai>=1.0.0              # LLM and embeddings
duckdb>=0.9.0              # In-memory analytics
pyyaml>=6.0                # YAML parsing
jsonschema>=4.19.0         # Validation
```

---

## 👤 Flow 2: User Flow (Online/Real-Time)

**Purpose:** Convert natural language request to Data Product specification

**When:** Every time a user submits a request

**Duration:** ~12-18 seconds per request

### Technology Stack

| Component | Technologies | Role |
|-----------|-------------|------|
| **User Interface** | Streamlit, FastAPI, Python CLI | Accept user input via multiple channels |
| **Orchestration** | LangGraph StateGraph, Python | Coordinate agent execution and state |
| **Intent Agent** | GPT-4, OpenAI API, Neo4j | Extract business requirements from NL |
| **Discovery Agent** | Neo4j Cypher, Vector Search, OpenAI Embeddings | Find relevant datasets semantically |
| **Modeling Agent** | GPT-4, JSON Schema | Design target data schema |
| **Transform Agent** | GPT-4, sqlparse, DuckDB | Generate and validate SQL/PySpark |
| **Quality Agent** | GPT-4, Great Expectations, dbt | Define quality rules and tests |
| **Packaging Agent** | YAML, JSON Schema, Jinja2 | Compile final specification |
| **Output Delivery** | Streamlit, FastAPI (JSON), File I/O | Return results to user |

### Core Dependencies
```python
# requirements.txt (Flow 2)
streamlit>=1.28.0          # Web UI
fastapi>=0.104.0           # REST API
uvicorn>=0.24.0            # ASGI server
langgraph>=0.0.20          # Multi-agent orchestration
langchain>=0.1.0           # LLM integration
openai>=1.0.0              # LLM API (GPT-4)
neo4j>=5.12.0              # Knowledge Graph queries
duckdb>=0.9.0              # Validation queries
sqlparse>=0.4.4            # SQL validation
great-expectations>=0.18.0 # Quality framework
pyyaml>=6.0                # YAML generation
jinja2>=3.1.2              # Template rendering
```

---

## 🔄 Shared Technologies (Used in Both Flows)

| Technology | Flow 1 Usage | Flow 2 Usage |
|------------|--------------|--------------|
| **Python** | Base language for all scripts | Base language for all agents |
| **OpenAI API (GPT-4)** | Generate transformation code | Power all 4 LLM-based agents |
| **OpenAI Embeddings** | Generate column embeddings | Search Knowledge Graph semantically |
| **Neo4j** | Populate metadata graph | Query metadata for discovery |
| **DuckDB** | Setup validation queries | Test generated SQL code |
| **Pandas** | Process bronze data | Data manipulation in agents |
| **YAML/JSON** | Store metadata | Output specification format |
| **JSON Schema** | Validate data structures | Validate final specs |
| **sqlparse** | Parse SQL during generation | Validate agent-generated SQL |

---

## 🎯 Technology Selection by Use Case

### Data Storage & Processing
```
Bronze Layer:    Parquet (columnar), PyArrow (writer)
Analytics:       DuckDB (embedded SQL)
Metadata:        Neo4j (graph database)
Output Format:   YAML + JSON
```

### LLM & AI
```
LLM Provider:    OpenAI GPT-4
Embeddings:      OpenAI text-embedding-3-small (768-dim)
Agent Framework: LangGraph (state management)
Orchestration:   LangChain (optional wrapper)
```

### User Interfaces
```
Web UI:          Streamlit (rapid prototyping)
API:             FastAPI (REST endpoints)
CLI:             argparse + Python
```

### Data Quality & Testing
```
Quality Rules:   Great Expectations
Transformation:  dbt (future integration)
Validation:      JSON Schema, sqlparse
```

---

## 📋 Complete Technology Stack Matrix

| Category | Technology | Flow 1 | Flow 2 | Purpose |
|----------|-----------|--------|--------|---------|
| **Language** | Python 3.10+ | ✅ | ✅ | Primary development language |
| **Data Generation** | Faker | ✅ | ❌ | Generate synthetic data |
| **Data Generation** | NumPy | ✅ | ✅ | Statistical distributions |
| **Data Processing** | Pandas | ✅ | ✅ | DataFrame operations |
| **Data Storage** | Parquet | ✅ | ✅ | Columnar file format |
| **Data Storage** | PyArrow | ✅ | ✅ | Parquet I/O |
| **Database** | DuckDB | ✅ | ✅ | In-memory SQL analytics |
| **Database** | Neo4j | ✅ | ✅ | Graph database for metadata |
| **Query Language** | Cypher | ✅ | ✅ | Neo4j graph queries |
| **LLM** | OpenAI GPT-4 | ✅ | ✅ | Code generation, NLP |
| **Embeddings** | OpenAI Embeddings | ✅ | ✅ | Semantic search vectors |
| **Agent Framework** | LangGraph | ❌ | ✅ | Multi-agent orchestration |
| **Agent Framework** | LangChain | ❌ | ✅ | LLM integration (optional) |
| **Web UI** | Streamlit | ❌ | ✅ | Interactive interface |
| **API** | FastAPI | ❌ | ✅ | REST endpoints |
| **API** | Uvicorn | ❌ | ✅ | ASGI web server |
| **Validation** | sqlparse | ✅ | ✅ | SQL syntax validation |
| **Validation** | JSON Schema | ✅ | ✅ | Schema validation |
| **Quality** | Great Expectations | ✅ | ✅ | Data quality framework |
| **Quality** | dbt | ✅ | ✅ | Transformation testing |
| **Serialization** | YAML | ✅ | ✅ | Human-readable config |
| **Serialization** | JSON | ✅ | ✅ | Machine-readable data |
| **Templating** | Jinja2 | ✅ | ✅ | Template rendering |
| **PySpark** | PySpark | ✅ | ✅ | Big data processing (optional) |

**Legend:**
- ✅ = Used in this flow
- ❌ = Not used in this flow

---

## 🔧 Installation by Flow

### Flow 1: Data Ingestion & Processing Setup

```bash
# Install Flow 1 dependencies
pip install faker numpy pandas pyarrow neo4j openai duckdb pyyaml jsonschema sqlparse great-expectations pyspark
```

**Estimated Size:** ~500 MB (including dependencies)

**One-Time Setup:**
```bash
# 1. Start Neo4j database
docker run -p 7474:7474 -p 7687:7687 neo4j:5.12

# 2. Generate bronze data
python data_generation/generate_bronze.py --output-dir data/bronze

# 3. Populate Knowledge Graph
python knowledge_graph/populate.py
```

---

### Flow 2: User Flow Setup

```bash
# Install Flow 2 dependencies
pip install streamlit fastapi uvicorn langgraph langchain openai neo4j duckdb sqlparse great-expectations pyyaml jinja2 jsonschema
```

**Estimated Size:** ~800 MB (including dependencies)

**Runtime Setup:**
```bash
# Option 1: Streamlit UI
streamlit run ui/streamlit_app.py

# Option 2: REST API
uvicorn api.fastapi_server:app --reload

# Option 3: CLI
python cli/generate.py --request "Your request here"
```

---

## 💰 Cost Breakdown

### Flow 1 (One-Time)
| Component | Cost | Frequency |
|-----------|------|-----------|
| OpenAI Embeddings (87 columns) | ~$0.01 | One-time |
| OpenAI GPT-4 (validation) | ~$0.05 | One-time |
| Neo4j Cloud (optional) | $65/month | If hosted |
| **Total One-Time** | **~$0.06** | - |

### Flow 2 (Per Request)
| Component | Cost | Notes |
|-----------|------|-------|
| OpenAI GPT-4 (4 agents) | ~$0.15 | 6-8K tokens |
| OpenAI Embeddings (queries) | ~$0.001 | 1-2 embedding calls |
| Neo4j Queries | Free | Self-hosted |
| **Total Per Request** | **~$0.15** | - |

**Monthly Estimate (100 requests):** ~$15

---

## 🚀 Performance Characteristics

| Metric | Flow 1 | Flow 2 |
|--------|--------|--------|
| **Execution Time** | 5-10 minutes | 12-18 seconds |
| **Frequency** | One-time + updates | On-demand |
| **Scalability** | Batch processing | Real-time |
| **Bottleneck** | Embedding generation | LLM API calls |
| **Optimization** | Parallel processing | Caching, batching |

---

## 🔍 Technology Rationale

### Why Two Separate Flows?

**Flow 1 (Offline):**
- Heavy metadata processing (embeddings, graph population)
- Infrequent execution (setup + occasional updates)
- Can be optimized for batch processing

**Flow 2 (Online):**
- Real-time user interaction required
- Frequent execution (every user request)
- Optimized for latency and responsiveness

### Shared Infrastructure Benefits
- **Neo4j:** Populated once (Flow 1), queried many times (Flow 2)
- **Embeddings:** Generated once (Flow 1), searched many times (Flow 2)
- **Bronze Data:** Created once (Flow 1), validated against repeatedly (Flow 2)

---

## 📦 Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      INFRASTRUCTURE                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────┐    ┌──────────────────────┐      │
│  │   Neo4j Database     │    │   Bronze Data Store  │      │
│  │   (Graph Metadata)   │    │   (Parquet Files)    │      │
│  └──────────────────────┘    └──────────────────────┘      │
│           ▲                            ▲                     │
│           │                            │                     │
│  ┌────────┴────────────────────────────┴──────────┐        │
│  │              FLOW 1 (OFFLINE)                   │        │
│  │   ┌────────────┐    ┌─────────────────────┐   │        │
│  │   │ Data Gen   │───>│  Metadata Extractor │   │        │
│  │   └────────────┘    └─────────────────────┘   │        │
│  └──────────────────────────────────────────────┘          │
│                                                              │
│           │                            │                     │
│           ▼                            ▼                     │
│  ┌───────────────────────────────────────────────────┐     │
│  │              FLOW 2 (ONLINE)                       │     │
│  │   ┌────────────┐    ┌─────────────────────┐      │     │
│  │   │ Streamlit  │───>│  LangGraph Agents   │      │     │
│  │   │  FastAPI   │<───│  (6 Agents)         │      │     │
│  │   └────────────┘    └─────────────────────┘      │     │
│  └───────────────────────────────────────────────────┘     │
│                                                              │
│  External:  ┌──────────────────┐                            │
│             │   OpenAI API     │                            │
│             │  (GPT-4 + Embed) │                            │
│             └──────────────────┘                            │
└─────────────────────────────────────────────────────────────┘
```

---

**Created:** November 28, 2025  
**Purpose:** Technology stack reference for both system flows

