"""
Unit tests for WorkflowAgent
"""

import pytest
import os
from pathlib import Path
from src.agents.workflow_agent import WorkflowAgent


@pytest.fixture
def workflow_agent(tmp_path):
    """Create WorkflowAgent instance with temporary output directory"""
    template_dir = str(Path(__file__).parent.parent / "templates")
    output_dir = str(tmp_path / "workflows")
    return WorkflowAgent(template_dir=template_dir, output_dir=output_dir)


@pytest.fixture
def sample_data_product_spec():
    """Sample data product specification"""
    return {
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
                {"name": "date", "type": "DATE"},
                {"name": "region", "type": "VARCHAR(50)"},
                {"name": "total_revenue", "type": "DECIMAL(18,2)"}
            ]
        },
        "transformations": {
            "language": "SQL",
            "code": "SELECT DATE(order_date) AS date, region, SUM(total_amount) AS total_revenue FROM bronze.orders GROUP BY 1, 2"
        },
        "quality_rules": [
            {
                "rule_id": "no_nulls_in_date",
                "rule_type": "not_null",
                "column": "date"
            }
        ]
    }


class TestWorkflowAgent:
    """Test suite for WorkflowAgent"""
    
    def test_initialization(self, workflow_agent):
        """Test WorkflowAgent initialization"""
        assert workflow_agent is not None
        assert Path(workflow_agent.output_dir).exists()
    
    def test_generate_airflow_dag(self, workflow_agent, sample_data_product_spec):
        """Test Airflow DAG generation"""
        dag_code = workflow_agent.generate_airflow_dag(
            data_product_spec=sample_data_product_spec,
            data_product_id="dp_test123"
        )
        
        # Check that DAG code was generated
        assert dag_code is not None
        assert len(dag_code) > 0
        
        # Check for key components
        assert "from airflow import DAG" in dag_code
        assert "daily_sales_analytics" in dag_code
        assert "validate_source_freshness" in dag_code
        assert "execute_transformation" in dag_code
        assert "run_quality_checks" in dag_code
        assert "update_registry_metadata" in dag_code
        
        # Check task dependencies
        assert "validate_sources >> transform >> quality_check >> update_metadata" in dag_code
    
    def test_generate_cron_job(self, workflow_agent, sample_data_product_spec):
        """Test cron job generation"""
        script_code = workflow_agent.generate_cron_job(
            data_product_spec=sample_data_product_spec,
            data_product_id="dp_test123"
        )
        
        # Check that script was generated
        assert script_code is not None
        assert len(script_code) > 0
        
        # Check for key components
        assert "#!/bin/bash" in script_code
        assert "daily_sales_analytics" in script_code
        assert "Validating source data freshness" in script_code
        assert "Executing transformation" in script_code
        assert "Running quality checks" in script_code
        assert "Updating registry" in script_code
    
    def test_extract_schedule_daily(self, workflow_agent):
        """Test schedule extraction for daily frequency"""
        sla = {"freshness": "Daily at 6:00 AM UTC"}
        schedule = workflow_agent._extract_schedule(sla)
        assert schedule == "00 6 * * *"
    
    def test_extract_schedule_hourly(self, workflow_agent):
        """Test schedule extraction for hourly frequency"""
        sla = {"freshness": "Updated hourly"}
        schedule = workflow_agent._extract_schedule(sla)
        assert schedule == "@hourly"
    
    def test_extract_schedule_weekly(self, workflow_agent):
        """Test schedule extraction for weekly frequency"""
        sla = {"freshness": "Weekly on Monday"}
        schedule = workflow_agent._extract_schedule(sla)
        assert schedule == "0 6 * * 1"
    
    def test_extract_schedule_monthly(self, workflow_agent):
        """Test schedule extraction for monthly frequency"""
        sla = {"freshness": "Monthly on the 1st"}
        schedule = workflow_agent._extract_schedule(sla)
        assert schedule == "0 6 1 * *"
    
    def test_sanitize_dag_id(self, workflow_agent):
        """Test DAG ID sanitization"""
        dag_id = workflow_agent._sanitize_dag_id("daily_sales_analytics", "1.0.0")
        assert dag_id == "daily_sales_analytics_v1_0_0"
        
        # Test with special characters
        dag_id = workflow_agent._sanitize_dag_id("sales-report@2024", "2.1.0")
        assert dag_id == "sales_report_2024_v2_1_0"
    
    def test_validate_cron_schedule(self, workflow_agent):
        """Test cron schedule validation"""
        # Valid schedules
        assert workflow_agent.validate_cron_schedule("0 6 * * *") is True
        assert workflow_agent.validate_cron_schedule("*/15 * * * *") is True
        assert workflow_agent.validate_cron_schedule("@daily") is True
        
        # Invalid schedules
        assert workflow_agent.validate_cron_schedule("invalid") is False
        assert workflow_agent.validate_cron_schedule("60 25 * * *") is False
    
    def test_get_next_execution_times(self, workflow_agent):
        """Test getting next execution times"""
        cron_expression = "0 6 * * *"  # Daily at 6 AM
        next_times = workflow_agent.get_next_execution_times(cron_expression, count=3)
        
        assert len(next_times) == 3
        
        # Check that times are in the future
        from datetime import datetime
        for time in next_times:
            assert time > datetime.now()
        
        # Check that times are in order
        assert next_times[0] < next_times[1] < next_times[2]
    
    def test_dag_file_creation(self, workflow_agent, sample_data_product_spec, tmp_path):
        """Test that DAG file is actually created"""
        workflow_agent.generate_airflow_dag(
            data_product_spec=sample_data_product_spec,
            data_product_id="dp_test123"
        )
        
        # Check that file was created
        dag_files = list(Path(workflow_agent.output_dir).glob("dags/*.py"))
        assert len(dag_files) > 0
        
        # Check file content
        dag_file = dag_files[0]
        content = dag_file.read_text()
        assert "from airflow import DAG" in content
    
    def test_cron_script_creation(self, workflow_agent, sample_data_product_spec, tmp_path):
        """Test that cron script file is actually created"""
        workflow_agent.generate_cron_job(
            data_product_spec=sample_data_product_spec,
            data_product_id="dp_test123"
        )
        
        # Check that file was created
        cron_files = list(Path(workflow_agent.output_dir).glob("cron/*.sh"))
        assert len(cron_files) > 0
        
        # Check file is executable
        cron_file = cron_files[0]
        assert os.access(cron_file, os.X_OK)
        
        # Check file content
        content = cron_file.read_text()
        assert "#!/bin/bash" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
