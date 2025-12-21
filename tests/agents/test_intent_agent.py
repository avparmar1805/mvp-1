import pytest
from unittest.mock import MagicMock
from src.agents.intent_agent import IntentAgent
from src.utils.llm_client import LLMClient

@pytest.fixture
def mock_llm_client():
    client = MagicMock(spec=LLMClient)
    return client

def test_analyze_daily_sales(mock_llm_client):
    # Setup mock response
    mock_llm_client.generate_structured_output.return_value = {
        "business_metrics": ["revenue", "order_count"],
        "dimensions": ["region", "category"],
        "temporal_granularity": "daily",
        "filters": []
    }
    
    agent = IntentAgent(llm_client=mock_llm_client)
    request = "I need daily sales analytics showing revenue and order count by region and category"
    
    result = agent.analyze(request)
    
    assert result["business_metrics"] == ["revenue", "order_count"]
    assert result["dimensions"] == ["region", "category"]
    assert result["temporal_granularity"] == "daily"
    assert result["confidence_score"] == 1.0

def test_analyze_missing_metrics(mock_llm_client):
    # Setup mock response with no metrics
    mock_llm_client.generate_structured_output.return_value = {
        "business_metrics": [],
        "dimensions": ["region"],
        "temporal_granularity": "weekly",
        "filters": []
    }
    
    agent = IntentAgent(llm_client=mock_llm_client)
    request = "Show me something by region"
    
    result = agent.analyze(request)
    
    assert result["business_metrics"] == []
    # Should have low confidence because no metrics found
    assert result["confidence_score"] <= 0.5
