import sys
import os
import pandas as pd
import duckdb

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.orchestrator import OrchestratorAgent
from src.utils.execution_engine import ExecutionEngine

def verify_fixes():
    orchestrator = OrchestratorAgent()
    engine = ExecutionEngine()
    
    scenarios = [
        {
            "name": "Clustering (Plot Check)",
            "intent": "Cluster orders into 3 groups based on total_amount and order_date",
            "type": "ml",
            "check": lambda res: res.get("ml_result", {}).get("plot_base64") is not None
        },
        {
            "name": "Forecasting (NameError Check)",
            "intent": "Predict daily revenue for the next 7 days",
            "type": "ml",
            "check": lambda res: "NameError" not in res.get("ml_result", {}).get("logs", "") and res.get("ml_result", {}).get("success") is True
        },
        {
            "name": "Segmentation (Plot Check)",
            "intent": "Segment customers into VIP, Loyal, and At-Risk groups",
            "type": "ml",
            "check": lambda res: res.get("ml_result", {}).get("plot_base64") is not None
        },
        {
            "name": "SQL Date Format (DuckDB Check)",
            "intent": "Show total sales by month",
            "type": "sql",
            "check": lambda res: "DATE_FORMAT" not in res.get("transformation", {}).get("sql_code", "").upper() and res.get("transformation", {}).get("sql_code")
        }
    ]
    
    print("--- STARTING VERIFICATION ---")
    
    for sc in scenarios:
        print(f"\nRunning: {sc['name']}...")
        try:
            result = orchestrator.run(sc['intent'])
            
            if sc["type"] == "ml":
                ml_res = result.get("ml_result") or {}
                # In Orchestrator, success is implied by existence of ml_result (errors are in global list)
                success = bool(ml_res)
                plot = ml_res.get("plot") # Key is 'plot', not 'plot_base64'
                logs = ml_res.get("output_summary", "") # Logs are not stored, but output_summary is

                print(f"  Success: {success}")
                print(f"  Plot Present: {bool(plot)}")
                print(f"  Global Pipeline Errors: {result.get('errors', [])}")
                print(f"  Intent: {result.get('intent', {}).get('task_type')}")

                if not success:
                     pass # Errors already printed
                elif not plot:
                    print(f"  WARNING: Success but NO PLOT.")
                    
                if sc["check"](result):
                    print(f"  ✅ {sc['name']} PASSED")
                else:
                    print(f"  ❌ {sc['name']} FAILED")
                    
            elif sc["type"] == "sql":
                sql = result.get("transformation", {}).get("sql_code", "")
                print(f"  SQL: {sql}")
                
                # Try executing
                if sql:
                    try:
                        engine.conn.execute(sql)
                        print("  Execution: Success")
                        if sc["check"](result):
                            print(f"  ✅ {sc['name']} PASSED")
                        else:
                            print(f"  ❌ {sc['name']} FAILED (Invalid SQL or Logic)")
                    except Exception as e:
                        print(f"  Execution Error: {e}")
                        print(f"  ❌ {sc['name']} FAILED")
                else:
                    print(f"  ❌ {sc['name']} FAILED (No SQL)")
                    
        except Exception as e:
            print(f"  CRITICAL ERROR: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    verify_fixes()
