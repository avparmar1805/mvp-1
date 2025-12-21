import os
import sys
import pandas as pd
from dotenv import load_dotenv

# Ensure we can import from src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.agents.orchestrator import OrchestratorAgent
from src.utils.execution_engine import ExecutionEngine

def main():
    print("="*80)
    print("PHASE 4: USE CASE VERIFICATION")
    print("="*80 + "\n")
    
    # 1. Setup
    load_dotenv(os.path.join(project_root, ".env"))
    
    try:
        engine = ExecutionEngine()
        orchestrator = OrchestratorAgent()
    except Exception as e:
        print(f"❌ Initialization Failed: {e}")
        return

    # 2. Define Scenarios
    scenarios = [
        {
            "name": "A1: Daily Sales",
            "request": "I need a daily sales report showing total revenue and order count by region for the last month."
        },
        {
            "name": "A2: Marketing Performance",
            "request": "Create a weekly marketing report with CTR, CVR, CPA, and ROAS"
        },
        {
            "name": "B3: Product Recommendations",
            "request": "Build a feature table for product recommendations based on user interaction signals including views, cart adds, purchases, and ratings."
        },
        {
            "name": "C2: Customer 360",
            "request": "Create a unified customer 360 view combining profile, transaction history, loyalty status, and support interactions."
        }
    ]

    # 3. Execution Loop
    for scenario in scenarios:
        print(f"\n⚡ Verifying: {scenario['name']}")
        print(f"   Request: \"{scenario['request']}\"")
        print("-" * 60)
        
        # A. Run Agents
        print("   > Running Agents...")
        result = orchestrator.run(scenario["request"])
        
        if result.get("errors"):
            print(f"   ❌ Agent Error: {result['errors']}")
            continue
            
        sql_code = result.get("transformation", {}).get("sql_code", "")
        if not sql_code:
            print("   ❌ No SQL generated")
            continue
            
        print("   > SQL Generated. Executing on Data...")
        
        # B. Execute SQL
        success, data, msg = engine.execute_query(sql_code)
        
        if success:
            row_count = len(data)
            print(f"   ✅ Execution Successful! Rows returned: {row_count}")
            
            if row_count > 0:
                print("   > Sample Output:")
                df = pd.DataFrame(data)
                print(df.head(3).to_string(index=False))
            else:
                print("   ⚠️ Query returned 0 rows. (Check temporal filters vs data range)")
        else:
            print(f"   ❌ SQL Execution Failed: {msg}")
            print(f"   > Offending SQL: {sql_code}")
            
        print("-" * 60)

if __name__ == "__main__":
    main()
