"""
Debug script to trace the end-to-end flow of a user request.
Usage: python scripts/debug_flow.py
"""

import os
import sys
import json
from pprint import pprint

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.agents.intent_agent import IntentAgent
from src.agents.discovery_agent import DiscoveryAgent
from src.agents.modeling_agent import ModelingAgent
from src.agents.transformation_agent import TransformationAgent
from src.agents.quality_agent import QualityAgent
from src.knowledge_graph.queries import create_query_service
from src.utils.llm_client import LLMClient

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def debug_flow():
    """Debug the complete flow step by step"""
    
    # User request
    user_request = "Show me daily revenue by region for high-value customers"
    
    print_section("USER REQUEST")
    print(f"Request: {user_request}")
    input("\n>>> Press Enter to start Intent Agent...")
    
    # ========================================================================
    # STEP 1: INTENT AGENT
    # ========================================================================
    print_section("STEP 1: INTENT AGENT")
    print("Initializing Intent Agent...")
    
    llm_client = LLMClient()
    print(f"LLM Provider: {llm_client.provider}")
    print(f"Model: {llm_client.model}")
    
    intent_agent = IntentAgent(llm_client)
    
    print(f"\nCalling: intent_agent.analyze('{user_request}')")
    intent = intent_agent.analyze(user_request)
    
    print("\n--- Intent Agent Output ---")
    pprint(intent)
    
    print("\nKey Observations:")
    print(f"  - Task Type: {intent.get('task_type')}")
    print(f"  - Business Metrics: {intent.get('business_metrics')}")
    print(f"  - Dimensions: {intent.get('dimensions')}")
    print(f"  - Filters: {intent.get('filters')}")
    
    input("\n>>> Press Enter to continue to Discovery Agent...")
    
    # ========================================================================
    # STEP 2: DISCOVERY AGENT
    # ========================================================================
    print_section("STEP 2: DISCOVERY AGENT")
    print("Initializing Knowledge Graph...")
    
    kg_service = create_query_service()
    print(f"KG loaded from: data/knowledge_graph.json")
    
    discovery_agent = DiscoveryAgent(kg_service)
    
    print(f"\nCalling: discovery_agent.discover(intent)")
    print(f"  Input metrics: {intent.get('business_metrics')}")
    print(f"  Input dimensions: {intent.get('dimensions')}")
    
    discovery_result = discovery_agent.discover(intent)
    
    print("\n--- Discovery Agent Output ---")
    print(f"Candidate Datasets ({len(discovery_result['candidate_datasets'])}):")
    for ds in discovery_result['candidate_datasets']:
        print(f"  - {ds['name']} (relevance: {ds.get('relevance_score', 'N/A')})")
        print(f"    Columns: {', '.join(ds.get('columns', [])[:5])}...")
    
    print(f"\nSelected Datasets: {discovery_result['selected_datasets']}")
    
    print("\nKey Observations:")
    print("  - How were datasets found?")
    print("    1. Exact term match in KG")
    print("    2. Semantic search (if no exact match)")
    print("    3. Heuristic rules (keyword matching)")
    
    input("\n>>> Press Enter to continue to Modeling Agent...")
    
    # ========================================================================
    # STEP 3: MODELING AGENT
    # ========================================================================
    print_section("STEP 3: MODELING AGENT")
    print("Initializing Modeling Agent...")
    
    modeling_agent = ModelingAgent(llm_client)
    
    print(f"\nCalling: modeling_agent.design_schema(...)")
    print(f"  Available datasets: {[ds['name'] for ds in discovery_result['candidate_datasets']]}")
    
    modeling_result = modeling_agent.design_schema(
        intent=intent,
        available_datasets=discovery_result['candidate_datasets']
    )
    
    print("\n--- Modeling Agent Output ---")
    print(f"Target Table: {modeling_result.get('target_table')}")
    print(f"Grain: {modeling_result.get('grain')}")
    print(f"\nColumns ({len(modeling_result.get('schema', []))}):")
    for col in modeling_result.get('schema', []):
        print(f"  - {col.get('name')} ({col.get('type')})")
        if col.get('description'):
            print(f"    Description: {col['description']}")
    
    print(f"\nPrimary Keys: {modeling_result.get('primary_keys', [])}")
    
    print("\nKey Observations:")
    print("  - LLM designed a star schema")
    print("  - Mapped business terms to physical columns")
    print("  - Identified necessary joins")
    
    input("\n>>> Press Enter to continue to Transformation Agent...")
    
    # ========================================================================
    # STEP 4: TRANSFORMATION AGENT
    # ========================================================================
    print_section("STEP 4: TRANSFORMATION AGENT")
    print("Initializing Transformation Agent...")
    
    transformation_agent = TransformationAgent(llm_client)
    
    print(f"\nCalling: transformation_agent.generate_logic(...)")
    
    transformation_result = transformation_agent.generate_logic(
        data_model=modeling_result,
        source_datasets=discovery_result['candidate_datasets']
    )
    
    print("\n--- Transformation Agent Output ---")
    print("Generated SQL:")
    print("-" * 80)
    print(transformation_result.get('sql_code', 'No SQL generated'))
    print("-" * 80)
    
    print("\nKey Observations:")
    print("  - LLM generated DuckDB-compatible SQL")
    print("  - Used strftime() for date formatting")
    print("  - Applied table aliases to avoid ambiguity")
    
    input("\n>>> Press Enter to continue to Quality Agent...")
    
    # ========================================================================
    # STEP 5: QUALITY AGENT
    # ========================================================================
    print_section("STEP 5: QUALITY AGENT")
    print("Initializing Quality Agent...")
    
    quality_agent = QualityAgent(llm_client)
    
    print(f"\nCalling: quality_agent.generate_checks(...)")
    
    quality_result = quality_agent.generate_checks(
        data_model=modeling_result
    )
    
    print("\n--- Quality Agent Output ---")
    print(f"Quality Checks ({len(quality_result.get('quality_checks', []))}):")
    for check in quality_result.get('quality_checks', []):
        print(f"  - {check.get('check_type')}: {check.get('column')}")
        if check.get('description'):
            print(f"    Description: {check['description']}")
    
    print("\nKey Observations:")
    print("  - LLM generated data quality rules")
    print("  - Checks include: not_null, positive_values, etc.")
    
    input("\n>>> Press Enter to see final summary...")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print_section("FINAL SUMMARY: COMPLETE PIPELINE")
    
    print("User Request:")
    print(f"  '{user_request}'")
    
    print("\nAgent Flow:")
    print("  1. Intent Agent")
    print(f"     → Extracted: {intent.get('business_metrics')} + {intent.get('dimensions')}")
    
    print("\n  2. Discovery Agent")
    print(f"     → Found datasets: {discovery_result['selected_datasets']}")
    
    print("\n  3. Modeling Agent")
    print(f"     → Designed table: {modeling_result.get('target_table_name')}")
    print(f"     → Columns: {len(modeling_result.get('columns', []))}")
    
    print("\n  4. Transformation Agent")
    print(f"     → Generated SQL ({len(transformation_result.get('sql_code', ''))} chars)")
    
    print("\n  5. Quality Agent")
    print(f"     → Created {len(quality_result.get('quality_checks', []))} quality checks")
    
    print("\n" + "="*80)
    print("  DEBUG SESSION COMPLETE")
    print("="*80)
    
    # Save results to file for inspection
    debug_output = {
        "user_request": user_request,
        "intent": intent,
        "discovery": discovery_result,
        "modeling": modeling_result,
        "transformation": transformation_result,
        "quality": quality_result
    }
    
    output_file = "debug_output.json"
    with open(output_file, "w") as f:
        json.dump(debug_output, f, indent=2)
    
    print(f"\nFull debug output saved to: {output_file}")
    print("\nTo inspect specific steps, set breakpoints in:")
    print("  - src/agents/intent_agent.py")
    print("  - src/agents/discovery_agent.py")
    print("  - src/knowledge_graph/queries.py")

if __name__ == "__main__":
    try:
        debug_flow()
    except KeyboardInterrupt:
        print("\n\nDebug session interrupted by user.")
    except Exception as e:
        print(f"\n\nError during debug: {e}")
        import traceback
        traceback.print_exc()
