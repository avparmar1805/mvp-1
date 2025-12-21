import pytest
from unittest.mock import MagicMock
from src.agents.quality_agent import QualityAgent
from src.utils.llm_client import LLMClient

@pytest.fixture
def mock_llm_client():
    return MagicMock(spec=LLMClient)

def test_generate_checks_basic(mock_llm_client):
    # Setup mock response
    expected_response = {
        "quality_checks": [
            {"check_type": "expect_column_values_to_be_unique", "column": "id", "description": "PK must be unique"},
            {"check_type": "expect_column_values_to_be_between", "column": "revenue", "description": "Revenue must be positive"}
        ]
    }
    mock_llm_client.generate_structured_output.return_value = expected_response
    
    agent = QualityAgent(llm_client=mock_llm_client)
    
    data_model = {
        "target_table": "gold.sales",
        "schema": [
            {"name": "id", "type": "INTEGER", "primary_key": True},
            {"name": "revenue", "type": "DECIMAL"}
        ]
    }
    
    result = agent.generate_checks(data_model)
    
    assert len(result["quality_checks"]) == 2
    assert result["quality_checks"][0]["column"] == "id"
    
    # Verify interaction
    mock_llm_client.generate_structured_output.assert_called_once()
