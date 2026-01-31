"""
Demonstration script for WorkflowAgent

Shows how to generate Airflow DAGs and cron jobs from data product specifications.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.workflow_agent import WorkflowAgent


def main():
    """Demonstrate WorkflowAgent capabilities"""
    
    print("=" * 70)
    print("WorkflowAgent Demonstration")
    print("=" * 70)
    print()
    
    # Sample data product specification
    sample_spec = {
        "metadata": {
            "name": "daily_sales_analytics",
            "version": "1.0.0",
            "description": "Daily sales analytics by region and category",
            "owner": "data_team"
        },
        "sla": {
            "freshness": "Daily at 6:00 AM UTC",
            "latency": "< 1 hour"
        },
        "source_datasets": [
            {"name": "bronze.orders", "type": "table"},
            {"name": "bronze.products", "type": "table"}
        ],
        "data_model": {
            "target_table": "gold.daily_sales_analytics",
            "grain": "Daily, by region and category",
            "schema": [
                {"name": "date", "type": "DATE", "nullable": False},
                {"name": "region", "type": "VARCHAR(50)", "nullable": False},
                {"name": "category", "type": "VARCHAR(100)", "nullable": False},
                {"name": "total_revenue", "type": "DECIMAL(18,2)", "nullable": False},
                {"name": "order_count", "type": "INTEGER", "nullable": False}
            ]
        },
        "transformations": {
            "language": "SQL",
            "code": """
                SELECT 
                    DATE(o.order_date) AS date,
                    o.region,
                    p.category,
                    SUM(o.total_amount) AS total_revenue,
                    COUNT(DISTINCT o.order_id) AS order_count
                FROM bronze.orders o
                JOIN bronze.products p ON o.product_id = p.product_id
                WHERE o.status = 'completed'
                GROUP BY DATE(o.order_date), o.region, p.category
            """
        },
        "quality_rules": [
            {
                "rule_id": "no_nulls_in_date",
                "rule_type": "not_null",
                "column": "date",
                "severity": "critical"
            },
            {
                "rule_id": "positive_revenue",
                "rule_type": "expression",
                "expression": "total_revenue >= 0",
                "severity": "critical"
            }
        ]
    }
    
    # Initialize WorkflowAgent
    print("📦 Initializing WorkflowAgent...")
    agent = WorkflowAgent()
    print(f"   Template directory: {agent.template_dir}")
    print(f"   Output directory: {agent.output_dir}")
    print()
    
    # Generate Airflow DAG
    print("🚀 Generating Airflow DAG...")
    dag_code = agent.generate_airflow_dag(
        data_product_spec=sample_spec,
        data_product_id="dp_demo123"
    )
    print(f"   ✅ DAG generated ({len(dag_code)} characters)")
    print()
    
    # Generate cron job
    print("⏰ Generating cron job script...")
    script_code = agent.generate_cron_job(
        data_product_spec=sample_spec,
        data_product_id="dp_demo123"
    )
    print(f"   ✅ Cron script generated ({len(script_code)} characters)")
    print()
    
    # Show schedule extraction examples
    print("📅 Schedule Extraction Examples:")
    print("-" * 70)
    
    test_schedules = [
        {"freshness": "Daily at 6:00 AM UTC"},
        {"freshness": "Updated hourly"},
        {"freshness": "Weekly on Monday"},
        {"freshness": "Monthly on the 1st"},
        {"freshness": "Every 15 minutes"}
    ]
    
    for sla in test_schedules:
        schedule = agent._extract_schedule(sla)
        print(f"   {sla['freshness']:30} → {schedule}")
    print()
    
    # Validate cron schedules
    print("✅ Cron Schedule Validation:")
    print("-" * 70)
    
    test_crons = [
        "0 6 * * *",      # Daily at 6 AM
        "*/15 * * * *",   # Every 15 minutes
        "@hourly",        # Hourly
        "invalid_cron"    # Invalid
    ]
    
    for cron in test_crons:
        is_valid = agent.validate_cron_schedule(cron)
        status = "✅ Valid" if is_valid else "❌ Invalid"
        print(f"   {cron:20} → {status}")
    print()
    
    # Show next execution times
    print("🕐 Next 5 Execution Times (Daily at 6 AM):")
    print("-" * 70)
    
    next_times = agent.get_next_execution_times("0 6 * * *", count=5)
    for i, time in enumerate(next_times, 1):
        print(f"   {i}. {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Summary
    print("=" * 70)
    print("✅ Demonstration Complete!")
    print("=" * 70)
    print()
    print("Generated Files:")
    print(f"   - Airflow DAG: {agent.output_dir}/dags/daily_sales_analytics_v1_0_0.py")
    print(f"   - Cron Script: {agent.output_dir}/cron/daily_sales_analytics.sh")
    print()
    print("Key Features:")
    print("   ✅ Automatic schedule extraction from SLA")
    print("   ✅ 4-step workflow (validate, transform, quality check, update registry)")
    print("   ✅ Support for SQL and PySpark transformations")
    print("   ✅ Cron schedule validation")
    print("   ✅ Next execution time calculation")
    print()


if __name__ == "__main__":
    main()
