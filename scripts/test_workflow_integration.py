"""
Test script to demonstrate WorkflowAgent integration with Orchestrator

This script shows the complete end-to-end flow:
1. User request → Orchestrator
2. All agents process (Intent, Discovery, Modeling, Transform, Quality, Packaging)
3. WorkflowAgent generates Airflow DAG and cron job
4. Display results
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.orchestrator import OrchestratorAgent
import json


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_subsection(title: str):
    """Print a formatted subsection header"""
    print(f"\n--- {title} ---")


def main():
    """Run the integrated workflow test"""
    
    print_section("🚀 WorkflowAgent Integration Test")
    
    # Sample user requests
    test_requests = [
        "Show me daily sales by region and category",
        "Create a weekly marketing performance report with CTR and conversion rate",
    ]
    
    # Initialize orchestrator (includes WorkflowAgent)
    print("\n📦 Initializing Orchestrator with all agents...")
    orchestrator = OrchestratorAgent()
    print("   ✅ Orchestrator initialized")
    print("   ✅ WorkflowAgent integrated")
    
    # Process first request
    user_request = test_requests[0]
    
    print_section(f"💬 Processing Request")
    print(f"\nUser Request: \"{user_request}\"")
    
    print("\n🔄 Running complete pipeline...")
    print("   → Intent Agent")
    print("   → Discovery Agent")
    print("   → Modeling Agent")
    print("   → Transformation Agent")
    print("   → Quality Agent")
    print("   → Packaging Agent")
    print("   → Workflow Agent (NEW!)")
    
    # Run orchestrator
    result = orchestrator.run(user_request)
    
    # Check for errors
    if result.get("errors"):
        print_section("❌ Errors Encountered")
        for error in result["errors"]:
            print(f"   • {error}")
        return
    
    # Display Intent
    print_section("🎯 Intent Analysis")
    intent = result.get("intent", {})
    print(f"   Task Type: {intent.get('task_type', 'N/A')}")
    print(f"   Metrics: {', '.join(intent.get('business_metrics', []))}")
    print(f"   Dimensions: {', '.join(intent.get('dimensions', []))}")
    print(f"   Granularity: {intent.get('temporal_granularity', 'N/A')}")
    
    # Display Discovery
    print_section("🔍 Dataset Discovery")
    discovery = result.get("discovery_result", {})
    datasets = discovery.get("selected_datasets", [])
    print(f"   Found {len(datasets)} relevant datasets:")
    for ds in datasets:
        # Handle both string and dict formats
        if isinstance(ds, str):
            print(f"   • {ds}")
        else:
            print(f"   • {ds.get('name', 'Unknown')}")
    
    # Display Data Model
    print_section("📊 Data Model")
    data_model = result.get("data_model", {})
    print(f"   Target Table: {data_model.get('target_table', 'N/A')}")
    print(f"   Grain: {data_model.get('grain', 'N/A')}")
    print(f"   Schema: {len(data_model.get('schema', []))} columns")
    
    # Display Transformation
    print_section("⚙️ Transformation Logic")
    transformation = result.get("transformation", {})
    sql_code = transformation.get("sql_code", "")
    if sql_code:
        print("\nGenerated SQL:")
        print("-" * 80)
        print(sql_code[:500] + ("..." if len(sql_code) > 500 else ""))
        print("-" * 80)
    
    # Display Quality Checks
    print_section("✅ Quality Checks")
    quality = result.get("quality_checks", {})
    checks = quality.get("quality_checks", [])
    print(f"   Generated {len(checks)} quality rules:")
    for i, check in enumerate(checks[:5], 1):  # Show first 5
        print(f"   {i}. {check.get('rule_id', 'N/A')}: {check.get('rule_type', 'N/A')}")
    
    # Display Data Product Spec
    print_section("📦 Data Product Specification")
    spec = result.get("data_product_spec", {})
    metadata = spec.get("metadata", {})
    print(f"   Name: {metadata.get('name', 'N/A')}")
    print(f"   Version: {metadata.get('version', 'N/A')}")
    print(f"   Description: {metadata.get('description', 'N/A')}")
    
    # Display SLA
    sla = spec.get("sla", {})
    if sla:
        print(f"\n   SLA:")
        print(f"   • Freshness: {sla.get('freshness', 'N/A')}")
        print(f"   • Latency: {sla.get('latency', 'N/A')}")
        print(f"   • Completeness: {sla.get('completeness', 'N/A')}")
    
    # Display Workflow Results (NEW!)
    print_section("🔄 Workflow Generation (NEW!)")
    workflow = result.get("workflow_result", {})
    
    if workflow:
        print(f"   ✅ Workflow generation successful!")
        print(f"\n   Data Product ID: {workflow.get('data_product_id', 'N/A')}")
        print(f"   Schedule: {workflow.get('schedule', 'N/A')}")
        print(f"\n   Generated Files:")
        print(f"   • Airflow DAG: {workflow.get('dag_file', 'N/A')}")
        print(f"   • Cron Script: {workflow.get('cron_file', 'N/A')}")
        
        # Show DAG code preview
        dag_code = workflow.get("dag_code", "")
        if dag_code:
            print(f"\n   DAG Code Preview (first 500 chars):")
            print("   " + "-" * 76)
            preview = dag_code[:500].replace("\n", "\n   ")
            print(f"   {preview}...")
            print("   " + "-" * 76)
        
        # Show cron code preview
        cron_code = workflow.get("cron_code", "")
        if cron_code:
            print(f"\n   Cron Script Preview (first 500 chars):")
            print("   " + "-" * 76)
            preview = cron_code[:500].replace("\n", "\n   ")
            print(f"   {preview}...")
            print("   " + "-" * 76)
    else:
        print("   ⚠️ No workflow generated")
    
    # Summary
    print_section("📈 Pipeline Summary")
    print(f"   ✅ Intent Analysis: Complete")
    print(f"   ✅ Dataset Discovery: {len(datasets)} datasets found")
    print(f"   ✅ Data Modeling: {len(data_model.get('schema', []))} columns")
    print(f"   ✅ Transformation: SQL generated")
    print(f"   ✅ Quality Checks: {len(checks)} rules")
    print(f"   ✅ Packaging: Specification created")
    print(f"   ✅ Workflow: {'DAG & Cron generated' if workflow else 'Not generated'}")
    
    # Save full result to file
    output_file = Path("output/workflow_integration_test_result.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"\n   💾 Full result saved to: {output_file}")
    
    print_section("✨ Integration Test Complete!")
    print("\nNext Steps:")
    print("   1. Check generated_workflows/dags/ for Airflow DAG")
    print("   2. Check generated_workflows/cron/ for cron script")
    print("   3. Review output/workflow_integration_test_result.json for full details")
    print()


if __name__ == "__main__":
    main()
