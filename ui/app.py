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
from src.utils.visualization import recommend_chart
from src.services.catalog import DataProductCatalog

# Page Config
st.set_page_config(
    page_title="Agentic Data Product Builder",
    page_icon="🤖",
    layout="wide"
)

# Load Env
load_dotenv(os.path.join(project_root, ".env"))

@st.cache_resource
def get_catalog():
    """Available globally cached catalog instance"""
    cat = DataProductCatalog()
    cat.index()
    return cat

@st.cache_resource
def get_engine():
    """Available globally cached execution engine"""
    return ExecutionEngine()

def main():
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "📂 Product Catalog", 
        "➕ New Data Product",
        "💾 Raw Data Explorer",
        "⚙️ Generated Workflows" # NEW
    ])
    
    if page == "📂 Product Catalog":
        render_catalog()
    elif page == "➕ New Data Product":
        render_builder()
    elif page == "💾 Raw Data Explorer":
        render_explorer()
    elif page == "⚙️ Generated Workflows":
        render_workflows()

def render_explorer():
    st.title("💾 Bronze Data Explorer")
    st.markdown("Inspect the raw source datasets available in the **Bronze Layer**.")
    
    engine = get_engine()
    
    # 1. Get Table List
    # DuckDB specific query to list tables
    try:
        tables_df = engine.conn.execute("SHOW TABLES").fetchdf()
        table_names = tables_df['name'].tolist()
    except Exception as e:
        st.error(f"Could not fetch tables: {e}")
        return

    # 2. Select Table
    selected_table = st.selectbox("Select Dataset", table_names)
    
    if selected_table:
        col1, col2 = st.columns(2)
        
        # 3. Show Schema
        with col1:
            st.subheader("📋 Schema")
            try:
                schema_df = engine.conn.execute(f"DESCRIBE {selected_table}").fetchdf()
                st.dataframe(schema_df[['column_name', 'column_type']], use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Error fetching schema: {e}")

        # 4. Show Sample Data
        with col2:
            st.subheader("👀 Sample Data (Top 50)")
            try:
                sample_df = engine.conn.execute(f"SELECT * FROM {selected_table} LIMIT 50").fetchdf()
                st.dataframe(sample_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error fetching sample: {e}")
                
        # 5. Stats
        st.divider()
        st.subheader("📊 Quick Stats")
        try:
            count = engine.conn.execute(f"SELECT COUNT(*) FROM {selected_table}").fetchone()[0]
            st.info(f"Total Rows: **{count:,}**")
        except:
            pass

def render_catalog():
    st.title("📂 Data Product Catalog")
    st.markdown("Browse and search existing data products.")
    
    catalog = get_catalog()
    
    # Search Bar
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("🔍 Search products (semantic search enabled)...", placeholder="E.g., 'money reports', 'customer help'")
    with col2:
        if st.button("🔄 Refresh Catalog"):
            catalog.index(force_refresh=True)
            st.rerun()
            
    # Get Results
    if search_query:
        results = catalog.search(search_query, top_k=6)
        st.subheader(f"Search Results for '{search_query}'")
    else:
        results = catalog.list_all()
        st.subheader("All Data Products")
        
    if not results:
        st.info("No data products found. Go to 'New Data Product' to create one!")
        return

    # Display Cards in Grid
    cols = st.columns(3)
    for idx, product in enumerate(results):
        with cols[idx % 3]:
            meta = product.get("metadata", {})
            name = meta.get("name", "Untitled")
            desc = meta.get("description", "No description")
            score = product.get("relevance_score", None)
            
            with st.container(border=True):
                st.subheader(name)
                st.caption(desc)
                if score:
                    st.progress(score, text=f"Match: {int(score*100)}%")
                
                # Tags
                tags = meta.get("tags", [])
                if tags:
                    st.write(" ".join([f"`#{t}`" for t in tags]))
                
                if st.button("👁️ View Product", key=f"view_{name}_{idx}"):
                    st.session_state.selected_product = product
                    st.rerun()

    # Viewer (Modal)
    if "selected_product" in st.session_state and st.session_state.selected_product:
        with st.expander("📖 Product Details", expanded=True):
            p = st.session_state.selected_product
            st.write(f"**Name:** {p['metadata']['name']}")
            st.code(yaml.dump(p), language="yaml")
            if st.button("Close Viewer"):
                del st.session_state.selected_product
                st.rerun()

def render_workflows():
    """Display generated Airflow DAGs and cron jobs"""
    st.title("⚙️ Generated Workflows")
    st.markdown("View all generated Airflow DAGs and cron job scripts.")
    
    from pathlib import Path
    
    # Tabs for DAGs and Cron
    tab_dags, tab_cron = st.tabs(["🚀 Airflow DAGs", "⏰ Cron Jobs"])
    
    with tab_dags:
        st.subheader("Airflow DAG Files")
        dag_dir = Path(project_root) / "generated_workflows" / "dags"
        
        if dag_dir.exists():
            dag_files = list(dag_dir.glob("*.py"))
            
            if dag_files:
                st.info(f"Found {len(dag_files)} DAG files")
                
                # Display each DAG
                for dag_file in sorted(dag_files, key=lambda x: x.stat().st_mtime, reverse=True):
                    with st.expander(f"📄 {dag_file.name}", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("File Size", f"{dag_file.stat().st_size / 1024:.1f} KB")
                        with col2:
                            import datetime
                            mtime = datetime.datetime.fromtimestamp(dag_file.stat().st_mtime)
                            st.metric("Modified", mtime.strftime("%Y-%m-%d %H:%M"))
                        with col3:
                            # Extract schedule from file
                            try:
                                content = dag_file.read_text()
                                import re
                                schedule_match = re.search(r'SCHEDULE_INTERVAL = "(.+?)"', content)
                                if schedule_match:
                                    st.metric("Schedule", schedule_match.group(1))
                            except:
                                pass
                        
                        # Show code
                        st.code(dag_file.read_text(), language="python")
                        
                        # Download button
                        st.download_button(
                            "⬇️ Download DAG",
                            dag_file.read_text(),
                            file_name=dag_file.name,
                            mime="text/x-python"
                        )
            else:
                st.warning("No DAG files found. Create a data product to generate workflows!")
        else:
            st.warning("DAG directory not found. Workflows will be created here after generating data products.")
    
    with tab_cron:
        st.subheader("Cron Job Scripts")
        cron_dir = Path(project_root) / "generated_workflows" / "cron"
        
        if cron_dir.exists():
            cron_files = list(cron_dir.glob("*.sh"))
            
            if cron_files:
                st.info(f"Found {len(cron_files)} cron scripts")
                
                # Display each cron script
                for cron_file in sorted(cron_files, key=lambda x: x.stat().st_mtime, reverse=True):
                    with st.expander(f"⏰ {cron_file.name}", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("File Size", f"{cron_file.stat().st_size / 1024:.1f} KB")
                        with col2:
                            import datetime
                            mtime = datetime.datetime.fromtimestamp(cron_file.stat().st_mtime)
                            st.metric("Modified", mtime.strftime("%Y-%m-%d %H:%M"))
                        
                        # Show code
                        st.code(cron_file.read_text(), language="bash")
                        
                        # Download button
                        st.download_button(
                            "⬇️ Download Script",
                            cron_file.read_text(),
                            file_name=cron_file.name,
                            mime="text/x-shellscript"
                        )
                        
                        # Installation instructions
                        st.info(f"""**To install this cron job:**
```bash
chmod +x {cron_file}
crontab -e
# Add the schedule line from the script
```""")
            else:
                st.warning("No cron scripts found. Create a data product to generate workflows!")
        else:
            st.warning("Cron directory not found. Workflows will be created here after generating data products.")

def render_builder():
    st.title("➕ New Data Product")
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

    # Tabs - Added Workflow tab
    tab_spec, tab_data, tab_sql, tab_quality, tab_workflow, tab_logs = st.tabs([
        "📄 Specification", "📊 Data Preview", "🛠 Transformation", "✅ Quality", "⚙️ Workflow", "📝 Logs"
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

    # 2. Data Preview Tab / ML Result
    with tab_data:
        ml_result = result.get("ml_result")
        
        if ml_result:
            st.subheader("🧠 Machine Learning Output")
            st.info(ml_result.get("explanation", ""))
            
            # Show Plot if available
            plot_b64 = ml_result.get("plot")
            if plot_b64:
                try:
                    import base64
                    st.image(base64.b64decode(plot_b64), caption="Generated Plot")
                except Exception as e:
                    st.error(f"Failed to render plot: {e}")
            else:
                st.warning("No visualization generated.")
                
            # Show Text Output
            output_summary = ml_result.get("output_summary")
            if output_summary:
                st.text_area("Execution Output", output_summary, height=150)
                
        else:
            # Standard SQL Data Preview
            sql_code = result.get("transformation", {}).get("sql_code", "")
            if sql_code:
                try:
                    success, data, msg = engine.execute_query(sql_code)
                    if success:
                        df = pd.DataFrame(data)
                        st.success(f"Execution Successful! Returned {len(df)} rows.")

                        # Smart Visualization
                        try:
                            viz_config = recommend_chart(df)
                            if viz_config["type"] != "table":
                                st.subheader(f"📈 {viz_config['title']}")
                                
                                if viz_config["type"] == "line":
                                    st.line_chart(df, x=viz_config["dimensions"][0], y=viz_config["primary_metric"])
                                elif viz_config["type"] == "bar":
                                    st.bar_chart(df, x=viz_config["dimensions"][0], y=viz_config["primary_metric"])
                                elif viz_config["type"] == "scatter":
                                    st.scatter_chart(df, x=viz_config["dimensions"][0], y=viz_config["primary_metric"])
                        except Exception as e:
                            st.warning(f"Could not generate chart: {e}")

                        st.dataframe(df.head(100), use_container_width=True)
                    else:
                        st.error(f"Execution Failed: {msg}")
                except Exception as e:
                    st.error(f"Preview Error: {e}")
            else:
                st.info("No SQL to execute.")

    # 3. Transformation Tab
    with tab_sql:
        if result.get("ml_result"):
            st.subheader("🐍 Generated Python Code")
            st.code(result["ml_result"]["code"], language="python")
        else:
            sql_code = result.get("transformation", {}).get("sql_code", "")
            st.code(sql_code, language="sql")
            if result.get("transformation", {}).get("explanation"):
                st.info(result["transformation"]["explanation"])

    # 4. Quality Tab
    with tab_quality:
        checks = (result.get("quality_checks") or {}).get("quality_checks", [])
        if checks:
            st.table(pd.DataFrame(checks))
        else:
            st.info("No quality checks generated.")

    # 5. Workflow Tab (NEW)
    with tab_workflow:
        workflow_result = result.get("workflow_result")
        
        if workflow_result:
            st.success("✅ Workflow generation successful!")
            
            # Display metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Data Product ID", workflow_result.get("data_product_id", "N/A"))
            with col2:
                st.metric("Schedule", workflow_result.get("schedule", "N/A"))
            with col3:
                st.metric("Workflow Type", "Airflow + Cron")
            
            st.divider()
            
            # Show generated files
            st.subheader("📁 Generated Files")
            dag_file = workflow_result.get("dag_file", "")
            cron_file = workflow_result.get("cron_file", "")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Airflow DAG:** `{dag_file}`")
            with col2:
                st.info(f"**Cron Script:** `{cron_file}`")
            
            st.divider()
            
            # Display DAG code
            dag_code = workflow_result.get("dag_code", "")
            if dag_code:
                with st.expander("🚀 View Airflow DAG Code", expanded=True):
                    st.code(dag_code, language="python")
                    st.download_button(
                        "⬇️ Download DAG",
                        dag_code,
                        file_name=f"{workflow_result.get('data_product_id', 'dag')}.py",
                        mime="text/x-python"
                    )
            
            # Display Cron code
            cron_code = workflow_result.get("cron_code", "")
            if cron_code:
                with st.expander("⏰ View Cron Job Script", expanded=False):
                    st.code(cron_code, language="bash")
                    st.download_button(
                        "⬇️ Download Cron Script",
                        cron_code,
                        file_name=f"{workflow_result.get('data_product_id', 'cron')}.sh",
                        mime="text/x-shellscript"
                    )
                    
                    # Installation instructions
                    st.info("""**To install this cron job:**
```bash
chmod +x script.sh
crontab -e
# Add: {} script.sh
```""".format(workflow_result.get("schedule", "0 6 * * *")))
            
            # Next steps
            st.divider()
            st.subheader("🎯 Next Steps")
            st.markdown("""
            1. **Deploy DAG**: Copy the DAG file to your Airflow `dags/` folder
            2. **Or use Cron**: Install the cron script on your server
            3. **Monitor**: Check execution logs and data quality
            4. **Iterate**: Refine based on results
            """)
        else:
            st.warning("No workflow generated. This might be an ML task or an error occurred.")
            if result.get("ml_result"):
                st.info("This is an ML task. Workflows are generated for analytics/reporting tasks.")
    
    # 6. Logs Tab
    with tab_logs:
        st.json(result)

if __name__ == "__main__":
    main()
