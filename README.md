# Agentic Data Product Builder

> **M.Tech Dissertation Project**  
> A Knowledge-Graph-Driven Multi-Agent System for Automated Data Product Generation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Project Overview

This project automates the creation of **Data Product specifications** from natural language business requests using a multi-agent AI system guided by a Knowledge Graph.

### What Problem Does This Solve?

Traditional data engineering requires manual:
- Understanding business requirements
- Finding relevant datasets
- Designing data models
- Writing transformation code
- Defining quality rules
- Creating documentation

This system **automates the entire pipeline** in seconds.

### How It Works

```
Natural Language Request
         ↓
   [Intent Agent] ────→ Extract business metrics, dimensions
         ↓
   [Discovery Agent] ──→ Find relevant datasets from Knowledge Graph
         ↓
   [Modeling Agent] ───→ Design target schema
         ↓
   [Transform Agent] ──→ Generate SQL/PySpark code
         ↓
   [Quality Agent] ────→ Define data quality rules
         ↓
   [Packaging Agent] ──→ Compile into YAML specification
         ↓
   Complete Data Product Specification
```

---

## 🚀 Key Features

- **🤖 Multi-Agent System**: 6 specialized agents working in coordination
- **🧠 Knowledge Graph**: Enterprise metadata (datasets, columns, lineage, glossary)
- **💬 Natural Language Interface**: Describe what you need in plain English
- **🔍 Intelligent Discovery**: Semantic search for relevant datasets
- **⚙️ Code Generation**: Automatic SQL/PySpark transformation code
- **✅ Quality Rules**: Automated data quality validation logic
- **📦 Complete Specs**: Machine-readable YAML/JSON output
- **🎨 User-Friendly UI**: Streamlit interface + REST API

---

## 📋 Supported Use Cases

### A1: Daily Sales Analytics
> "I need daily sales analytics showing revenue, order count, and units sold by region and category."

**Output**: Aggregated sales metrics with SQL transformation

### A2: Marketing Campaign Performance
> "Create a weekly marketing report with CTR, CVR, CPA, and ROAS."

**Output**: Marketing KPIs with calculated metrics

### B3: Product Recommendation Features
> "Build a feature table for product recommendations based on user interactions."

**Output**: ML feature engineering pipeline

### C2: Customer 360
> "Create a unified customer view combining profile, transactions, and support history."

**Output**: Customer master data with joins and aggregations

---

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Presentation Layer                        │
│           [Streamlit UI]    [REST API]    [CLI]              │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                   Orchestrator Agent                         │
│              (LangGraph State Management)                    │
└────────────────────────────┬────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼────────┐  ┌────────▼────────┐  ┌───────▼────────┐
│ Intent Agent   │  │ Discovery Agent │  │ Modeling Agent │
└────────────────┘  └─────────────────┘  └────────────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼────────┐  ┌────────▼────────┐  ┌───────▼────────┐
│Transform Agent │  │ Quality Agent   │  │Packaging Agent │
└────────────────┘  └─────────────────┘  └────────────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│              Knowledge Graph (Neo4j/NetworkX)                │
│    [Datasets] [Columns] [Business Terms] [Lineage]          │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                    Bronze Data Layer                         │
│              [Parquet Files] [DuckDB]                        │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

- **Language**: Python 3.10+
- **Multi-Agent Framework**: LangGraph / CrewAI
- **LLM**: OpenAI GPT-4 / Anthropic Claude
- **Knowledge Graph**: Neo4j (or NetworkX)
- **Data Processing**: Pandas, DuckDB
- **API**: FastAPI
- **UI**: Streamlit
- **Data Format**: Parquet, YAML, JSON

---

## 📂 Project Structure

```
mvp-1/
├── README.md                          # This file
├── PROJECT_PLAN.md                    # Detailed project plan
├── ARCHITECTURE.md                    # System architecture documentation
├── DATA_GENERATION_PLAN.md            # Bronze layer data generation strategy
├── IMPLEMENTATION_ROADMAP.md          # 12-week implementation timeline
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variables template
│
├── data/                              # Data storage
│   └── bronze/                        # Bronze layer (synthetic data)
│       ├── orders/
│       ├── customers/
│       ├── products/
│       ├── marketing_campaigns/
│       ├── marketing_events/
│       ├── user_interactions/
│       └── support_tickets/
│
├── data_generation/                   # Data generation scripts
│   ├── generate_bronze.py
│   ├── generators/
│   ├── utils/
│   └── tests/
│
├── knowledge_graph/                   # Knowledge Graph components
│   ├── schema.py
│   ├── populate.py
│   ├── queries.py
│   └── embeddings.py
│
├── agents/                            # Multi-agent system
│   ├── base_agent.py
│   ├── orchestrator.py
│   ├── intent_agent.py
│   ├── discovery_agent.py
│   ├── modeling_agent.py
│   ├── transformation_agent.py
│   ├── quality_agent.py
│   └── packaging_agent.py
│
├── schemas/                           # Data Product schemas
│   ├── data_product_schema.json
│   ├── templates/
│   └── examples/
│
├── utils/                             # Utility modules
│   ├── llm_client.py
│   ├── graph_client.py
│   └── validators.py
│
├── ui/                                # User interfaces
│   └── streamlit_app.py
│
├── api/                               # REST API
│   └── fastapi_server.py
│
├── tests/                             # Test suite
│   ├── test_agents.py
│   ├── test_kg_queries.py
│   └── use_cases/
│
└── docs/                              # Additional documentation
    ├── architecture.md
    ├── agent_design.md
    └── evaluation.md
```

---

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.10 or higher
- Neo4j (optional, can use NetworkX)
- OpenAI API key or Anthropic API key

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd mvp-1
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

Example `.env`:
```
OPENAI_API_KEY=sk-...
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

5. **Generate bronze layer data**:
```bash
python data_generation/generate_bronze.py --output-dir data/bronze
```

6. **Populate Knowledge Graph**:
```bash
python knowledge_graph/populate.py
```

7. **Run the application**:

**Streamlit UI**:
```bash
streamlit run ui/streamlit_app.py
```

**REST API**:
```bash
uvicorn api.fastapi_server:app --reload
```

---

## 🎮 Usage

### Via Streamlit UI

1. Open browser to `http://localhost:8501`
2. Select a use case or enter custom request
3. Click "Generate Data Product"
4. View and download the generated YAML specification

### Via REST API

```bash
curl -X POST "http://localhost:8000/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "user_request": "I need daily sales analytics by region and category"
  }'
```

### Via Python

```python
from agents.orchestrator import OrchestratorAgent

orchestrator = OrchestratorAgent()

request = "Create a weekly marketing performance report with CTR and CVR"
result = orchestrator.run(request)

print(result["data_product_spec"])
```

---

## 📊 Example Output

**Input Request**:
> "I need daily sales analytics showing revenue, order count, and units sold by region and category."

**Generated Data Product Specification** (YAML):

```yaml
data_product:
  metadata:
    name: "daily_sales_analytics"
    version: "1.0.0"
    description: "Daily sales metrics by region and category"
    owner: "analytics_team"
    created_at: "2025-11-15T10:30:00Z"
    tags: ["sales", "analytics", "daily"]
  
  data_model:
    target_table: "gold.daily_sales_analytics"
    grain: "Daily, by region and category"
    schema:
      - name: "date"
        type: "DATE"
        nullable: false
        primary_key: true
      - name: "region"
        type: "VARCHAR"
        nullable: false
        primary_key: true
      - name: "category"
        type: "VARCHAR"
        nullable: false
        primary_key: true
      - name: "total_revenue"
        type: "DECIMAL(18,2)"
        nullable: false
      - name: "order_count"
        type: "INTEGER"
        nullable: false
      - name: "units_sold"
        type: "INTEGER"
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
  
  sla:
    freshness: "Daily at 6:00 AM UTC"
    latency: "< 30 minutes"
    completeness: "> 99%"
```

---

## 🧪 Testing

### Run all tests:
```bash
pytest tests/
```

### Run specific test suite:
```bash
pytest tests/test_agents.py
pytest tests/use_cases/test_a1_sales.py
```

### Run with coverage:
```bash
pytest --cov=agents --cov-report=html
```

---

## 📈 Evaluation Metrics

### Quantitative Metrics

- **Dataset Discovery Accuracy**: % of correct datasets identified
- **Schema Accuracy**: % of correct columns in target schema
- **SQL Correctness**: % of valid, executable SQL
- **Execution Time**: End-to-end latency (p50, p90, p99)

### Target Performance

- Dataset discovery accuracy: **> 85%**
- SQL correctness: **> 90%**
- Execution time (p90): **< 20 seconds**
- Spec completeness: **> 95%**

---

## 🗺️ Roadmap

### Phase 1: Foundation (Weeks 1-2) ✅
- [x] Bronze layer generation
- [x] Knowledge Graph setup
- [x] Data Product schema definition

### Phase 2: Core Agents (Weeks 3-5) 🚧
- [ ] Orchestrator Agent
- [ ] Intent Agent
- [ ] Discovery Agent
- [ ] Modeling Agent

### Phase 3: Transformation & Quality (Weeks 6-7)
- [ ] Transformation Agent
- [ ] Quality Agent
- [ ] Packaging Agent

### Phase 4: Use Case Testing (Weeks 8-9)
- [ ] A1: Daily Sales Analytics
- [ ] A2: Marketing Performance
- [ ] B3: Product Recommendations
- [ ] C2: Customer 360

### Phase 5: UI & Integration (Week 10)
- [ ] Streamlit UI
- [ ] REST API
- [ ] Documentation

### Phase 6: Evaluation (Weeks 11-12)
- [ ] Quantitative evaluation
- [ ] Qualitative evaluation
- [ ] Dissertation writing

---

## 🤝 Contributing

This is an academic research project. Contributions, suggestions, and feedback are welcome!

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📚 Documentation

- **[PROJECT_PLAN.md](PROJECT_PLAN.md)**: Comprehensive project plan with use cases, architecture, and evaluation strategy
- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Detailed system architecture and component design
- **[DATA_GENERATION_PLAN.md](DATA_GENERATION_PLAN.md)**: Bronze layer data generation specifications
- **[IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md)**: 12-week implementation timeline with milestones

---

## 🔬 Research Contributions

### Novel Aspects

1. **Knowledge-Graph-Driven Agent Coordination**: Using KG as shared context for multi-agent systems
2. **End-to-End Data Product Automation**: From NL request to executable code + specification
3. **Hybrid Reasoning**: Combining LLM reasoning with graph-based retrieval
4. **Domain-Specific Agent Specialization**: Each agent has focused expertise

### Potential Publications

- IEEE BigData, ICDE, VLDB (demo track)
- MLOps, Data Engineering, Knowledge Graphs workshops
- M.Tech Dissertation

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍🎓 Author

**Anshul Parmar**  
M.Tech Student  
Dissertation Project  

---

## 🙏 Acknowledgments

- OpenAI for GPT-4 API
- Neo4j for graph database
- LangChain/LangGraph for multi-agent framework
- Streamlit for rapid UI development

---

## 📧 Contact

For questions or feedback, please open an issue or contact [your-email@example.com].

---

## 🔖 Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{parmar2025agentic,
  title={Agentic Data Product Builder: A Knowledge-Graph-Driven Multi-Agent System for Automated Data Product Generation},
  author={Parmar, Anshul},
  year={2025},
  school={[Your University]},
  type={M.Tech Dissertation}
}
```

---

**Status**: 🚧 In Development (Planning Phase Complete)  
**Last Updated**: November 15, 2025

