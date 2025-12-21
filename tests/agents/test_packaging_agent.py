import pytest
import yaml
from src.agents.packaging_agent import PackagingAgent

@pytest.fixture
def packaging_agent():
    return PackagingAgent()

@pytest.fixture
def sample_state():
    return {
        "user_request": "Daily sales report",
        "intent": {
            "business_metrics": ["revenue"],
            "dimensions": ["region"],
            "temporal_granularity": "daily"
        },
        "discovery_result": {
            "selected_datasets": ["orders"]
        },
        "data_model": {
            "target_table": "gold.daily_sales",
            "grain": "daily by region",
            "schema": [{"name": "revenue", "type": "decimal"}],
            "primary_keys": ["date", "region"]
        },
        "transformation": {
            "sql_code": "SELECT * FROM orders",
            "explanation": "Simple select"
        },
        "quality_checks": {
            "quality_checks": [{"check_type": "not_null", "column": "revenue"}]
        }
    }

def test_package_structure(packaging_agent, sample_state):
    result = packaging_agent.package(sample_state)
    
    assert "data_product_spec" in result
    assert "yaml_output" in result
    
    spec = result["data_product_spec"]
    assert spec["metadata"]["name"] == "Daily Sales"
    assert spec["business_context"]["intent"]["metrics"] == ["revenue"]
    assert spec["data_model"]["target_table"] == "gold.daily_sales"
    assert spec["source_data"]["datasets"] == ["orders"]
    assert spec["transformation"]["code"] == "SELECT * FROM orders"
    assert len(spec["quality_assurance"]["rules"]) == 1

def test_yaml_generation(packaging_agent, sample_state):
    result = packaging_agent.package(sample_state)
    yaml_str = result["yaml_output"]
    
    # Verify it's valid YAML
    parsed = yaml.safe_load(yaml_str)
    assert parsed["metadata"]["name"] == "Daily Sales"
    assert parsed["data_model"]["grain"] == "daily by region"
