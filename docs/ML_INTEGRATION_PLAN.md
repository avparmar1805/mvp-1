# ML Integration Plan for Agentic Data Product Builder

## 1. Executive Summary
This document outlines the strategy for extending the existing Agentic Data Product Builder to support **Machine Learning (ML)** and **Advanced Analytics** capabilities. The goal is to evolve the system from descriptive analytics (SQL-based) to predictive and prescriptive analytics (Python/ML-based).

## 2. Architecture Extension

### 2.1 New Component: MachineLearningAgent
A new agent will be introduced to handle non-SQL processing tasks.
- **Role**: Execute Python code for training models, running inference, and performing complex statistical analysis.
- **Input**:
    - Source Data (result of a SQL query from Transformation Agent).
    - Data Product Goal (e.g., "Predict revenue").
- **Output**:
    - Trained Model (serialized).
    - Inference Results (DataFrame).
    - Accuracy Metrics (RMSE, Accuracy Score).
    - Visualization (Matplotlib/Seaborn plots).

### 2.2 Orchestrator Logic Update
The Orchestrator state machine (`StateGraph`) needs to be updated to support a "forked" execution path:
1.  **Intent Analysis**: Detect `task_type` (Analytics vs. ML).
2.  **Routing**:
    - If `Analytics` → Transformation Agent (SQL).
    - If `ML` → Transformation Agent (SQL for data prep) → MachineLearningAgent (Python for modeling).

## 3. Supported ML Use Cases

### 3.1 Sales Forecasting (Time Series)
- **Goal**: Predict future metrics based on historical trends.
- **Example Request**: "Predict daily revenue for the next 7 days."
- **Technique**: Prophet, ARIMA, or valid Linear Regression on lag features.
- **Data Requirement**: Aggregated time-series data (from `orders`).

### 3.2 Customer Segmentation (Clustering)
- **Goal**: Group entities based on behavioral attributes.
- **Example Request**: "Segment customers into VIP, Loyal, and At-Risk groups."
- **Technique**: K-Means Clustering.
- **Data Requirement**: Customer dimensions + Transaction aggregates (RFM metrics).

### 3.3 Sentiment Analysis (NLP)
- **Goal**: Extract qualitative insights from text.
- **Example Request**: "Analyze sentiment of recent support tickets."
- **Technique**: Pre-trained Transformer models (HuggingFace) or VADER.
- **Data Requirement**: Text columns (e.g., `support_tickets.description`, `reviews.comment`).

## 4. Implementation Tasks

### Phase 1: Foundation
- [ ] **Create `MachineLearningAgent` Class**
    - Implement `generate_python_code()` method using LLM.
    - Implement `execute_python_code()` method (sandboxed environment).
- [ ] **Update `IntentAgent`**
    - Enhance prompt to classify `task_type` (Descriptive, Predictive, Prescriptive).
    - Extract ML parameters (target variable, forecast horizon, number of clusters).

### Phase 2: Integration
- [ ] **Update Orchestrator Workflow**
    - Add `machine_learning` node to LangGraph.
    - Define conditional edges based on `task_type`.
- [ ] **Data Handover Mechanism**
    - Implement a mechanism to pass SQL query results (from Transformation Agent) directly to the ML Agent (e.g., as a Pandas DataFrame).

### Phase 3: Use Case Verification
- [ ] **Implement Forecasting Template**
    - Test end-to-end flow with "Predict revenue" request.
- [ ] **Implement Segmentation Template**
    - Test end-to-end flow with "Clustering" request.
- [ ] **Update `PackagingAgent`**
    - Support bundling Python scripts and model artifacts in the final Data Product Specification.

## 5. Security & Safety
- **Sandboxing**: Python code execution must be sandboxed (using Docker or restricted execution environments like `E2B`) to prevent malicious system access.
- **Library Whitelist**: Only allow approved libraries (`pandas`, `numpy`, `scikit-learn`, `prophet`).
