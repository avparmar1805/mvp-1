# Product Roadmap: UI & Lifecycle Extensions

This document outlines the plan for extending the Agentic Data Product Builder with **Advanced Visualization** and **Automated Lifecycle Management**.

## 1. UI Extensions: Intelligent Data Visualization

**Goal**: Transform the "Data Preview" from a static table into an interactive dashboard that automatically selects the best chart type for the data.

### 1.1 Architecture: The "Visualization Agent"
We will introduce a lightweight heuristic step (or agent) at the end of the pipeline.

*   **Input**: The final Data Product dataframe (from Transformation or ML Agent).
*   **Process**: Analyze data types and cardinality.
*   **Output**: A `viz_config` JSON object.

### 1.2 Heuristic Logic (Auto-Detect)
The system will automatically decide the chart type:
*   **Line Chart**: If schema has `Date/Time` column + `Numeric` metric.
    *   *Example*: "Daily Revenue" -> X=Date, Y=Revenue.
*   **Bar Chart**: If schema has `Categorical` (low cardinality < 20) + `Numeric` metric.
    *   *Example*: "Revenue by Region" -> X=Region, Y=Revenue.
*   **Scatter Plot**: If schema has 2+ `Numeric` metrics.
    *   *Example*: "Price vs. Quantity" -> X=Price, Y=Quantity.
*   **Map**: If schema has `Latitude/Longitude` or `Region/Country` codes.
    *   *Example*: "Sales by Zipcode".

### 1.3 UI Implementation Details (Streamlit)
The `ui/app.py` will be updated to read the `viz_config` and render dynamically:

```python
# Pseudo-code for UI logic
if viz_config['type'] == 'line':
    st.line_chart(data, x=viz_config['x'], y=viz_config['y'])
elif viz_config['type'] == 'bar':
    st.bar_chart(data, x=viz_config['x'], y=viz_config['y'])
else:
    st.dataframe(data) # Fallback
```

---

## 2. Lifecycle Management: "Set and Forget"

**Goal**: define how often a Data Product updates and execute that schedule automatically.

### 2.1 Deciding Frequency (Intent Layer)
The **Intent Agent** is responsible for extracting the *implied* or *explicit* schedule from the user's natural language request.

*   **User**: "Give me a **daily** sales report."
*   **Intent**: `frequency: "daily"`, `cron: "0 8 * * *"` (e.g., Run at 8 AM).
*   **User**: "Analyze monthly churn."
*   **Intent**: `frequency: "monthly"`, `cron: "0 0 1 * *"` (Run on 1st of month).
*   **Default**: If unspecified, default to `on_demand` (run once).

### 2.2 Execution Architecture (The Scheduler)
We need a background service to handle recurring runs.

**Option A: Lightweight (Python Loop)**
*   A simple script `scheduler.py` runs a loop checking the DB for "due" jobs.
*   *Pros*: No extra infrastructure. Good for MVP.
*   *Cons*: Not scalable.

**Option B: Robust (Airflow / Workflow Engine)**
*   The **Packaging Agent** acts as a "DAG Generator".
*   It generates a `.py` file (Airflow DAG) representing the pipeline.
*   *Pros*: Production-grade, handles retries, logs, backfill.
*   *Cons*: Higher complexity.

### 2.3 Proposed MVP Workflow
1.  **Tagging**: The `packaging_agent` saves the spec with a `schedule` metadata field.
    ```yaml
    metadata:
      name: "Daily Sales"
      owner: "User"
      schedule: "daily"  <-- NEW
    ```
2.  **Execution**: We implement a simple **Cron Job** or specific endpoint `/api/v1/run_scheduled` that:
    *   Scans all YAML specs in `output/specs/`.
    *   Checks if `last_run + schedule < current_time`.
    *   If yes, triggers `orchestrator.run(spec)`.
    *   Updates the "Data Preview" cache (so the user always sees fresh data).
