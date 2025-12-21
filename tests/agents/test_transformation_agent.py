import pytest
from unittest.mock import MagicMock
from src.agents.transformation_agent import TransformationAgent
from src.utils.llm_client import LLMClient

@pytest.fixture
def mock_llm_client():
    return MagicMock(spec=LLMClient)

def test_generate_logic_basic(mock_llm_client):
    # Setup mock response
    expected_response = {
        "sql_code": "SELECT region, SUM(amount) as total_revenue FROM bronze.orders GROUP BY region",
        "explanation": "Aggregates revenue by region."
    }
    mock_llm_client.generate_structured_output.return_value = expected_response
    
    agent = TransformationAgent(llm_client=mock_llm_client)
    
    data_model = {
        "target_table": "gold.sales_by_region",
        "grain": "region",
        "schema": [{"name": "region"}, {"name": "total_revenue"}]
    }
    
    source_datasets = [
        {"name": "bronze.orders", "columns": ["order_id", "amount", "region"]}
    ]
    
    result = agent.generate_logic(data_model, source_datasets)
    
    assert result["sql_code"] == expected_response["sql_code"]
    assert "explanation" in result
    
    # Verify prompt construction implicitly by checking if llm was called
    mock_llm_client.generate_structured_output.assert_called_once()
