import streamlit as st
import sys
import os
import yaml
import pandas as pd
from dotenv import load_dotenv

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.agents.orchestrator import OrchestratorAgent
from src.utils.execution_engine import ExecutionEngine

# Page Config
st.set_page_config(
    page_title="Agentic Data Product Builder",
    page_icon="🤖",
    layout="wide"
)

# Load Env
load_dotenv(os.path.join(project_root, ".env"))

def main():
    st.title("🤖 Agentic Data Product Builder")
    st.markdown("Build data products from natural language using AI agents.")

    # Sidebar: Use Case Selection
    st.sidebar.header("Configuration")
    use_case = st.sidebar.selectbox(
        "Select Use Case Template",
        [
            "Custom Request",
            "A1: Daily Sales Analytics",
            "A2: Marketing Performance",
            "B3: Product Recommendations",
            "C2: Customer 360"
        ]
    )

    # Pre-fill request based on selection
    default_request = ""
    if use_case == "A1: Daily Sales Analytics":
        default_request = "I need a daily sales report showing total revenue and order count by region for the last month."
    elif use_case == "A2: Marketing Performance":
        default_request = "Create a weekly marketing report with CTR, CVR, CPA, and ROAS"
    elif use_case == "B3: Product Recommendations":
        default_request = "Build a feature table for product recommendations based on user interaction signals including views, cart adds, purchases, and ratings."
    elif use_case == "C2: Customer 360":
        default_request = "Create a unified customer 360 view combining profile, transaction history, loyalty status, and support interactions."

    # Input Area
    with st.form("request_form"):
        user_request = st.text_area("Describe your Data Product:", value=default_request, height=100)
        submitted = st.form_submit_button("Generate Data Product", type="primary")

    if submitted and user_request:
        run_generation(user_request)

def run_generation(request):
    # Initialize Agents
    with st.spinner("Initializing Agents..."):
        try:
            orchestrator = OrchestratorAgent()
            engine = ExecutionEngine() # For data preview
        except Exception as e:
            st.error(f"Failed to initialize agents: {e}")
            return

    # Tabs
    tab_spec, tab_data, tab_sql, tab_quality, tab_logs = st.tabs([
        "📄 Specification", "📊 Data Preview", "🛠 Transformation", "✅ Quality", "📝 Logs"
    ])

    # Run Pipeline
    with st.status("Running Agent Pipeline...", expanded=True) as status:
        st.write("🧠 Interpreting Intent...")
        result = orchestrator.run(request)
        
        if result.get("errors"):
            status.update(label="Pipeline Failed", state="error")
            st.error("Errors encountered during generation:")
            for err in result["errors"]:
                st.error(f"- {err}")
            return
            
        st.write("✅ Pipeline Complete!")
        status.update(label="Generation Successful", state="complete")

    # 1. Specification Tab
    with tab_spec:
        if result.get("yaml_output"):
            st.code(result["yaml_output"], language="yaml")
            st.download_button(
                "Download YAML", 
                result["yaml_output"], 
                file_name="data_product.yaml"
            )
        else:
            st.warning("No spec generated.")

    # 2. Data Preview Tab
    with tab_data:
        sql_code = result.get("transformation", {}).get("sql_code", "")
        if sql_code:
            try:
                success, data, msg = engine.execute_query(sql_code)
                if success:
                    df = pd.DataFrame(data)
                    st.success(f"Execution Successful! Returned {len(df)} rows.")
                    st.dataframe(df.head(100), use_container_width=True)
                else:
                    st.error(f"Execution Failed: {msg}")
            except Exception as e:
                st.error(f"Preview Error: {e}")
        else:
            st.info("No SQL to execute.")

    # 3. Transformation Tab
    with tab_sql:
        sql_code = result.get("transformation", {}).get("sql_code", "")
        st.code(sql_code, language="sql")
        if result.get("transformation", {}).get("explanation"):
            st.info(result["transformation"]["explanation"])

    # 4. Quality Tab
    with tab_quality:
        checks = result.get("quality_checks", {}).get("quality_checks", [])
        if checks:
            st.table(pd.DataFrame(checks))
        else:
            st.info("No quality checks generated.")

    # 5. Logs Tab
    with tab_logs:
        st.json(result)

if __name__ == "__main__":
    main()
