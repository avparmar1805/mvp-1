import pytest
from unittest.mock import MagicMock
from src.agents.modeling_agent import ModelingAgent
from src.utils.llm_client import LLMClient

@pytest.fixture
def mock_llm_client():
    return MagicMock(spec=LLMClient)

def test_design_schema_sales(mock_llm_client):
    # Setup mock response
    expected_schema = {
        "target_table": "gold.daily_sales",
        "grain": "daily by region",
        "schema": [
            {"name": "date", "type": "DATE", "primary_key": True},
            {"name": "region", "type": "VARCHAR", "primary_key": True},
            {"name": "total_revenue", "type": "DECIMAL"}
        ],
        "primary_keys": ["date", "region"]
    }
    mock_llm_client.generate_structured_output.return_value = expected_schema
    
    agent = ModelingAgent(llm_client=mock_llm_client)
    
    intent = {
        "business_metrics": ["revenue"],
        "dimensions": ["region"],
        "temporal_granularity": "daily"
    }
    
    datasets = [
        {"name": "bronze.orders", "columns": ["order_date", "total_amount", "region"]}
    ]
    
    result = agent.design_schema(intent, datasets)
    
    assert result["target_table"] == "gold.daily_sales"
    assert len(result["schema"]) == 3
    assert "date" in result["primary_keys"]
