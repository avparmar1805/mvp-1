import pytest
from unittest.mock import MagicMock
from src.agents.discovery_agent import DiscoveryAgent
from src.knowledge_graph.queries import KnowledgeGraphQueryService

@pytest.fixture
def mock_kg_service():
    return MagicMock(spec=KnowledgeGraphQueryService)

def test_discover_basic(mock_kg_service):
    # Setup mock response
    mock_datasets = [
        {
            "name": "bronze.orders",
            "relevance_score": 0.9,
            "matched_terms": ["revenue"],
            "matched_columns": ["total_amount"]
        },
        {
            "name": "bronze.products",
            "relevance_score": 0.5,
            "matched_terms": [],
            "matched_columns": []
        }
    ]
    mock_kg_service.find_datasets_for_metrics.return_value = mock_datasets
    
    agent = DiscoveryAgent(kg_service=mock_kg_service)
    
    intent = {
        "business_metrics": ["revenue"],
        "dimensions": ["region"]
    }
    
    result = agent.discover(intent)
    
    # Verify interaction with KG service
    mock_kg_service.find_datasets_for_metrics.assert_called_once_with(
        metrics=["revenue"],
        dimensions=["region"]
    )
    
    # Verify result structure
    assert "candidate_datasets" in result
    assert len(result["candidate_datasets"]) == 2
    assert "selected_datasets" in result
    assert "bronze.orders" in result["selected_datasets"]

def test_discover_no_results(mock_kg_service):
    mock_kg_service.find_datasets_for_metrics.return_value = []
    
    agent = DiscoveryAgent(kg_service=mock_kg_service)
    intent = {"business_metrics": ["unknown_metric"]}
    
    result = agent.discover(intent)
    
    assert result["candidate_datasets"] == []
    assert result["selected_datasets"] == []
