import pytest
from unittest.mock import MagicMock, patch
from src.agents.orchestrator import OrchestratorAgent

@pytest.fixture
def mock_agents():
    with patch("src.agents.orchestrator.IntentAgent") as MockIntent, \
         patch("src.agents.orchestrator.DiscoveryAgent") as MockDiscovery, \
         patch("src.agents.orchestrator.ModelingAgent") as MockModeling, \
         patch("src.agents.orchestrator.TransformationAgent") as MockTransformation, \
         patch("src.agents.orchestrator.QualityAgent") as MockQuality, \
         patch("src.agents.orchestrator.create_query_service"), \
         patch("src.agents.orchestrator.LLMClient"):
        
        # Setup specific mock instances
        intent_instance = MockIntent.return_value
        intent_instance.analyze.return_value = {"business_metrics": ["revenue"]}
        
        discovery_instance = MockDiscovery.return_value
        discovery_instance.discover.return_value = {"candidate_datasets": [{"name": "orders"}]}
        
        modeling_instance = MockModeling.return_value
        modeling_instance.design_schema.return_value = {"target_table": "gold.sales"}
        
        transformation_instance = MockTransformation.return_value
        transformation_instance.generate_logic.return_value = {"sql_code": "SELECT *"}
        
        quality_instance = MockQuality.return_value
        quality_instance.generate_checks.return_value = {"quality_checks": []}
        
        yield {
            "intent": intent_instance,
            "discovery": discovery_instance,
            "modeling": modeling_instance,
            "transformation": transformation_instance,
            "quality": quality_instance
        }

def test_orchestrator_run_success(mock_agents):
    orchestrator = OrchestratorAgent()
    result = orchestrator.run("calc revenue")
    
    # Assert flow execution
    assert result["intent"] == {"business_metrics": ["revenue"]}
    assert result["discovery_result"] == {"candidate_datasets": [{"name": "orders"}]}
    assert result["data_model"] == {"target_table": "gold.sales"}
    assert result["transformation"] == {"sql_code": "SELECT *"}
    assert result["quality_checks"] == {"quality_checks": []}
    assert result["errors"] == [] 
    
    # Assert agent calls
    mock_agents["intent"].analyze.assert_called_once()
    mock_agents["discovery"].discover.assert_called_once()
    mock_agents["modeling"].design_schema.assert_called_once()
    mock_agents["transformation"].generate_logic.assert_called_once()
    mock_agents["quality"].generate_checks.assert_called_once()

def test_orchestrator_handle_error(mock_agents):
    # Simulate error in intent agent
    mock_agents["intent"].analyze.side_effect = Exception("API Error")
    
    orchestrator = OrchestratorAgent()
    result = orchestrator.run("break me")
    
    assert "Intent Error: API Error" in result["errors"][0]
    # Subsequent steps might run or fail depending on logic, but error should be captured
