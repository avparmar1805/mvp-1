# Project Implementation Summary: Agentic Data Product Builder
**MTech Dissertation Project**

## 1. Introduction
The objective of this project was to build an **Agentic Data Product Builder**—an autonomous AI system capable of converting natural language business requests into fully specified, executable data products. The system leverages Large Language Models (LLMs) and a multi-agent architecture to handle the complex reasoning required for semantic understanding, schema design, and code generation.

This document summarizes the technical implementation across the five phases of development.

---

## 2. System Architecture
The system follows a **Multi-Agent Orchestration Architecture** implemented using **LangGraph**. A central `OrchestratorAgent` manages a directed acyclic graph (DAG) of specialized agents, passing a shared state (`DataProductState`) between them.

**Key Components:**
*   **Orchestrator**: Manages workflow control and error handling.
*   **Cognitive Agents**: Intent, Discovery, Modeling (Reasoning & Planning).
*   **Operational Agents**: Transformation, Quality, Packaging (Code Generation).
*   **Knowledge Graph**: A semantic layer metadata store (NetworkX + Vector Embeddings) used for context-aware retrieval.

---

## 3. Implementation Phases

### Phase 1: Foundation & Knowledge Graph
**Goal**: Establish the semantic layer to ground the agents' reasoning.

*   **Implementation**:
    *   Constructed a **Metadata Service** using `NetworkX` to represent relationships between Tables, Columns, and Business Metrics.
    *   Implemented **Semantic Search** using OpenAI Embeddings (`text-embedding-3-small`) and Cosine Similarity to allow agents to find relevant datasets based on vague descriptions (e.g., finding "revenue" from "orders" table).
    *   **Outcome**: A queryable Knowledge Graph capable of mapping business terms to physical data assets.

### Phase 2: Core Cognitive Agents (The "Brain")
**Goal**: Enable the system to understand *what* to build.

1.  **Intent Agent (`IntentAgent`)**:
    *   **Role**: Disambiguate user requests.
    *   **Logic**: Parses natural language (e.g., "sales by region") into structured JSON extracting `metrics` ("sales"), `dimensions` ("region"), and `filters`.
    *   **Validation**: Uses strict Pydantic models to ensure output structure.

2.  **Discovery Agent (`DiscoveryAgent`)**:
    *   **Role**: Identify necessary data assets.
    *   **Logic**: Queries the Knowledge Graph using the terms extracted by the Intent Agent.
    *   **Outcome**: Selects relevant Bronze tables (e.g., `orders`, `customers`) required to fulfill the request.

3.  **Modeling Agent (`ModelingAgent`)**:
    *   **Role**: Design the target schema.
    *   **Logic**: Synthesizes the discovered assets into a **Target Data Model** (Gold Layer schema), defining column names, data types, and grain (e.g., `daily`, `weekly`).

### Phase 3: Operational Agents (The "Hands")
**Goal**: Generate the executable artifacts.

4.  **Transformation Agent (`TransformationAgent`)**:
    *   **Role**: Write the SQL logic.
    *   **Logic**: Generates ANSI SQL (DuckDB compatible) to transform Bronze data into the Gold model designed by the Modeling Agent.
    *   **Constraint**: Enforces strictly read-only operations (`SELECT`, `CTEs`) to prevent data corruption.

5.  **Quality Agent (`QualityAgent`)**:
    *   **Role**: Ensure data reliability.
    *   **Logic**: Generates **Data Quality Checks** (e.g., `not_null`, `unique`, `min_value`) appropriate for the target columns.

6.  **Packaging Agent (`PackagingAgent`)**:
    *   **Role**: Assemble the final deliverable.
    *   **Logic**: Compiles the outputs of all previous agents into a standardized **YAML Data Product Specification**. This serves as the "Product Contract" for deployment.

### Phase 4: Use Case Verification
**Goal**: Prove the system works on real data, not just theoretical code generation.

*   **Execution Engine**:
    *   Built a lightweight verification engine using **DuckDB**.
    *   Functionality: Automatically loads raw Parquet files from the `data/bronze` directory as tables and performs in-memory SQL execution.

*   **Validated Scenarios**:
    The system was tested against 4 distinct business domains:
    1.  **Use Case A1 (Retail Sales)**: successfully aggregated 100k+ rows of daily revenue data.
    2.  **Use Case A2 (Marketing)**: successfully calculated complex ratios (CTR, ROAS) from ad campaign data.
    3.  **Use Case B3 (Recommendation System)**: successfully joined user interactions (views, cart adds) to build a feature table (200k+ rows).
    4.  **Use Case C2 (Customer 360)**: successfully unified profile, transaction, and support data into a single customer view.

### Phase 5: Access Layer (UI & API)
**Goal**: Make the system accessible to end-users and external systems.

1.  **Streamlit User Interface (`ui/app.py`)**:
    *   Interactive web application allowing users to select templates or type custom requests.
    *   **Features**: Real-time pipeline visualization, "Data Preview" tab showing live results from DuckDB, and "Download YAML" capability.
    
2.  **REST API (`api/server.py`)**:
    *   **FastAPI** server exposing the core engine.
    *   **Endpoint**: `POST /api/v1/generate` accepts a text prompt and returns the full JSON specification.
    *   Enables integration into CI/CD pipelines (e.g., triggering a dbt job creation automatically).

---

## 4. Conclusion
The MVP successfully demonstrates the viability of **Agentic Engineering**. By decomposing the complex task of "Data Engineering" into specialized, autonomous agents, the system achieves a high degree of reliability and flexibility. It moves beyond simple "Text-to-SQL" by managing the entire lifecycle: from ambiguous intent -> semantic discovery -> schema modeling -> code generation -> validation -> packaging.

**Final Deliverable**: A functional end-to-end Python application that accepts a sentence and outputs a verified, deployment-ready Data Product Specification.
