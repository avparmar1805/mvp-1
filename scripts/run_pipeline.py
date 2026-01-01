import os
import sys
import datetime
from dotenv import load_dotenv

# Ensure we can import from src (go up one level from scripts)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.agents.orchestrator import OrchestratorAgent

# Setup output file
os.makedirs(os.path.join(project_root, "logs"), exist_ok=True)
LOG_FILE = os.path.join(project_root, "logs", "pipeline_runs.log")

def log_output(message):
    """Print to console and append to log file"""
    print(message)
    with open(LOG_FILE, "a") as f:
        f.write(message + "\n")

def main():
    # Clear previous log for this run (optional, or keep appending)
    with open(LOG_FILE, "a") as f:
        f.write(f"\n\n{'='*80}\nRUN TIMESTAMP: {datetime.datetime.now()}\n{'='*80}\n")

    # Load environment variables (expecting OPENAI_API_KEY)
    load_dotenv(os.path.join(project_root, ".env"))
    
    if not os.getenv("OPENAI_API_KEY"):
        log_output("Error: OPENAI_API_KEY not found in environment variables.")
        return

    log_output("Initializing Orchestrator Agent...")
    try:
        # This will initialize all sub-agents and the KG client
        orchestrator = OrchestratorAgent()
    except Exception as e:
        log_output(f"Failed to initialize orchestrator: {e}")
        return

    # Define test scenarios
    test_inputs = [
        "Predict daily revenue for the next 7 days.",
        "Segment customers into VIP, Loyal, and At-Risk groups."
    ]

    for i, user_request in enumerate(test_inputs, 1):
        log_output(f"\n{'='*20} SCENARIO {i}: {user_request} {'='*20}")
        log_output("-" * 80)
    
        try:
            # Run the workflow
            result = orchestrator.run(user_request)
            
            # Display Results
            if result.get("errors"):
                log_output("\n❌ Errors encountered:")
                for error in result["errors"]:
                    log_output(f"- {error}")
            else:
                log_output("\n✅ Success!")
                
                log_output("\n1. Interpreted Intent:")
                intent = result.get("intent") or {}
                log_output(f"   Metrics: {intent.get('business_metrics')}")
                log_output(f"   Dimensions: {intent.get('dimensions')}")
                log_output(f"   Granularity: {intent.get('temporal_granularity')}")
                
                log_output("\n2. Discovered Data:")
                discovery = result.get("discovery_result") or {}
                log_output(f"   Selected Datasets: {discovery.get('selected_datasets')}")
                
                log_output("\n3. Generated Data Model:")
                model = result.get("data_model") or {}
                log_output(f"   Target Table: {model.get('target_table')}")
                log_output(f"   Grain: {model.get('grain')}")
                log_output(f"   Schema Columns: {len(model.get('schema', []))}")
                for col in model.get('schema', [])[:5]: # Show first 5 columns
                    log_output(f"     - {col.get('name')} ({col.get('type')})")
                if len(model.get('schema', [])) > 5:
                    log_output("     ... (more columns)")

                log_output("\n4. Transformation Logic:")
                transform = result.get("transformation") or {}
                sql = transform.get("sql_code", "N/A")
                log_output(f"   SQL Code: {sql}")
                
                log_output("\n5. Quality Checks:")
                quality = result.get("quality_checks") or {}
                checks = quality.get("quality_checks", [])
                log_output(f"   Generated Checks: {len(checks)}")
                for check in checks:
                    log_output(f"     - {check.get('check_type')} on {check.get('column')}")

                log_output("\n6. Packaging:")
                spec = result.get("data_product_spec") or {}
                yaml_out = result.get("yaml_output", "")
                log_output(f"   Spec generated: {spec.get('metadata', {}).get('name')}")
                log_output(f"   YAML Size: {len(yaml_out)} chars")
                log_output("\n" + "-"*40 + " YAML PREVIEW " + "-"*40)
                log_output(yaml_out[:500] + "..." if len(yaml_out) > 500 else yaml_out)
                log_output("-" * 94)

                # Save spec to file
                sanitized_name = spec.get("metadata", {}).get("name", "data_product").lower().replace(" ", "_")
                filename = os.path.join(project_root, "output", "specs", f"{sanitized_name}.yaml")
                with open(filename, "w") as f:
                    f.write(yaml_out)
                log_output(f"\n📁 Spec saved to: {filename}")

        except Exception as e:
            log_output(f"\n❌ Runtime Error: {e}")
            import traceback
            traceback.print_exc(file=sys.stdout) # Print trace to console only typically, or use traceback.format_exc to log
            
        log_output("\n" + "="*80 + "\n")
    
    log_output(f"Run complete. Logs saved to: {LOG_FILE}")

if __name__ == "__main__":
    main()
