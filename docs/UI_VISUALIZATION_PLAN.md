# Implementation Plan: Intelligent UI Visualization

This document details the technical plan to implement **Automated Data Visualization** in the Agentic Data Product Builder.

## 1. Objective
Transform the static "Data Preview" table in the Streamlit UI into an **Interactive Dashboard** that automatically selects and renders the most appropriate chart type based on the generated data's structure.

## 2. Architecture

We will implement a **Frontend-Driven** approach for the MVP to keep the backend agents focused on data generation.

### Components
1.  **Visualization Service (`src/services/visualization.py`)**: A Python utility class that takes a DataFrame and returns a "Visualization Configuration".
2.  **UI Update (`ui/app.py`)**: Logic to consume the configuration and render Streamlit charts.

## 3. Detailed Logic: The "Auto-Viz" Heuristics

The `VisualizationService` will analyze the DataFrame output from the pipeline and apply the following rules:

### Step 3.1: Column Classification
First, classify every column in the result dataset:
*   **Temporal**: `datetime`, `date` types (e.g., `date`, `month`).
*   **Numeric**: `int`, `float`, `decimal` (e.g., `revenue`, `count`, `score`).
*   **Categorical**: `string`, `object` with low cardinality (< 20 unique values) (e.g., `region`, `category`, `status`).
*   **Text/ID**: High cardinality strings (e.g., `review_text`, `customer_id`).

### Step 3.2: Chart Selection Rules
Based on the available columns, select the primary chart type:

1.  **Time Series Trend (Line/Area Chart)**
    *   *Condition*: Contains at least 1 **Temporal** column + 1 **Numeric** metric.
    *   *Mapping*: X-Axis = Time, Y-Axis = Metric.
    *   *Example*: "Daily Revenue" (A1 Use Case).

2.  **Category Comparison (Bar Chart)**
    *   *Condition*: Contains 1 **Categorical** column + 1 **Numeric** metric.
    *   *Mapping*: X-Axis = Category, Y-Axis = Metric.
    *   *Example*: "Sales by Region".

3.  **Part-to-Whole (Pie/Donut Chart)**
    *   *Condition*: Contains 1 **Categorical** column + 1 **Numeric** metric + User intent implies "distribution" or "share".
    *   *Example*: "Market Share by Campaign".

4.  **Correlation (Scatter Plot)**
    *   *Condition*: Contains 2+ **Numeric** metrics and NO Temporal column.
    *   *Mapping*: X-Axis = Metric 1, Y-Axis = Metric 2.
    *   *Example*: "Price vs Units Sold".

5.  **Pivot/Heatmap**
    *   *Condition*: Contains 2 **Categorical** columns + 1 **Numeric** metric.
    *   *Mapping*: X=Cat1, Y=Cat2, Color=Metric.
    *   *Example*: "Performance by Region AND Category".

## 4. Implementation Steps

### Task 1: Create Visualization Service
Create `src/utils/visualization.py`:
- `classify_columns(df)`: Returns dict of column types.
- `recommend_chart(df)`: Returns config dict:
  ```python
  {
      "chart_type": "line",
      "primary_metric": "total_revenue",
      "dimensions": ["date"],
      "title": "Total Revenue over Time"
  }
  ```

### Task 2: Integrate into Streamlit
Update `ui/app.py`:
- Import `recommend_chart`.
- Inside the `if success:` block (where dataframe is available):
  ```python
  # 1. Get Recommendation
  viz_config = recommend_chart(result_df)
  
  # 2. Render Chart
  st.subheader("📈 Visual Insights")
  if viz_config['type'] == 'line':
      st.line_chart(result_df, x=viz_config['dimensions'][0], y=viz_config['primary_metric'])
  elif viz_config['type'] == 'bar':
      st.bar_chart(result_df, x=viz_config['dimensions'][0], y=viz_config['primary_metric'])
  # ... etc
  ```

### Task 3: Add Manual Overrides (Optional Extension)
Add a sidebar section in Streamlit:
- "Customize Chart": Dropdown to force a specific chart type (Bar/Line/Area) ignoring the recommendation.

## 5. Mockup Validation
*   **Input**: "Daily Sales Report"
*   **Data**: `date`, `region`, `revenue`
*   **Result**: 
    1.  **Auto-Detect**: Line Chart (X=`date`, Y=`revenue`, Color=`region`).
    2.  **User View**: Sees a multi-line graph showing revenue trends for each region.

This creates an immediate "WOW" factor for the user, moving beyond simple rows and columns.
