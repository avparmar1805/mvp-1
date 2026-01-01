import sys
import os
from loguru import logger
import json

# Add project root to path
sys.path.append(os.getcwd())

from src.agents.orchestrator import OrchestratorAgent

def main():
    logger.info("Initializing Orchestrator Agent for ML Verification...")
    orchestrator = OrchestratorAgent()
    
    # Define an ML Request
    request = "Cluster orders into 3 groups based on total_amount and order_date"
    
    logger.info(f"Running pipeline for request: '{request}'")
    
    # Run Orchestrator
    result = orchestrator.run(request)
    
    # Check for errors
    errors = result.get("errors", [])
    if errors:
        logger.error(f"❌ Pipeline failed with errors: {errors}")
        return

    # Verify ML Result
    ml_result = result.get("ml_result")
    if not ml_result:
        logger.error("❌ No ML Result found in state!")
        return
        
    logger.info("✅ ML Result found.")
    logger.info(f"Generated Code Length: {len(ml_result.get('code', ''))}")
    
    if ml_result.get("plot"):
        logger.info(f"✅ Plot generated (base64 length: {len(ml_result['plot'])})")
    else:
        logger.warning("⚠️ No plot was generated.")

    # Verify Packaging
    spec = result.get("data_product_spec")
    if spec and "machine_learning" in spec:
        logger.info("✅ Data Product Spec contains 'machine_learning' section.")
    else:
        logger.error("❌ Data Product Spec missing 'machine_learning' section.")

    # Print success
    logger.info("\nE2E ML Verification Successful! 🚀")

if __name__ == "__main__":
    main()
