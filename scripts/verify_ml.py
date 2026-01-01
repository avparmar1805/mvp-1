import sys
import os
import pandas as pd
from loguru import logger

# Add project root to path
sys.path.append(os.getcwd())

from src.utils.execution_engine import ExecutionEngine
from src.agents.ml_agent import MachineLearningAgent

def main():
    logger.info("Initializing Engine & ML Agent...")
    engine = ExecutionEngine()
    ml_agent = MachineLearningAgent()
    
    # 1. Fetch Data
    logger.info("Fetching data from 'orders' table...")
    success, df, msg = engine.execute_query_df("SELECT order_date, total_amount FROM orders LIMIT 100")
    
    if not success:
        logger.error(f"Failed to fetch data: {msg}")
        return

    logger.info(f"Fetched DataFrame shape: {df.shape}")
    
    # 2. Generate Code
    intent = "Cluster orders into 3 groups based on total_amount and visualize them"
    logger.info(f"Generating ML code for intent: '{intent}'")
    
    code_result = ml_agent.generate_code(intent, df)
    
    if not code_result or "python_code" not in code_result:
        logger.error("Failed to generate code.")
        return
        
    python_code = code_result["python_code"]
    logger.info("\nGenerated Python Code:\n" + "-"*40 + "\n" + python_code + "\n" + "-"*40)
    
    # 3. Execute Code
    logger.info("Executing code...")
    exec_result = ml_agent.execute_code(python_code, df)
    
    if exec_result["success"]:
        logger.info("✅ Execution Successful!")
        logger.info(f"Result type: {type(exec_result.get('result'))}")
        if exec_result.get("plot_base64"):
            logger.info("✅ Plot generated (base64 length: " + str(len(exec_result["plot_base64"])) + ")")
    else:
        logger.error(f"❌ Execution Failed: {exec_result.get('error')}")
        logger.error(f"Logs: {exec_result.get('logs')}")

if __name__ == "__main__":
    main()
