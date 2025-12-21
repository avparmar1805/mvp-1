from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from src.agents.intent_agent import IntentAgent
from src.agents.discovery_agent import DiscoveryAgent
from src.agents.modeling_agent import ModelingAgent
from src.agents.transformation_agent import TransformationAgent
from src.agents.quality_agent import QualityAgent
from src.agents.packaging_agent import PackagingAgent
from src.utils.llm_client import LLMClient
from src.knowledge_graph.queries import KnowledgeGraphQueryService, create_query_service

# Define the state of the data product generation workflow
class DataProductState(TypedDict):
    user_request: str
    intent: Optional[Dict[str, Any]]
    discovery_result: Optional[Dict[str, Any]]
    data_model: Optional[Dict[str, Any]]
    transformation: Optional[Dict[str, Any]]
    quality_checks: Optional[Dict[str, Any]]
    data_product_spec: Optional[Dict[str, Any]]
    yaml_output: Optional[str]
    errors: List[str]

class OrchestratorAgent:
    """
    Coordinator agent that manages the workflow between Intent, Discovery, Modeling, Transformation, and Quality agents.
    """
    
    def __init__(self):
        # Initialize dependencies
        self.llm_client = LLMClient()
        self.kg_service = create_query_service()  # Assumes default path or env var
        
        # Initialize sub-agents
        self.intent_agent = IntentAgent(self.llm_client, self.kg_service)
        self.discovery_agent = DiscoveryAgent(self.kg_service)
        self.modeling_agent = ModelingAgent(self.llm_client)
        self.transformation_agent = TransformationAgent(self.llm_client)
        self.quality_agent = QualityAgent(self.llm_client)
        self.packaging_agent = PackagingAgent()
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        
    def _build_workflow(self):
        workflow = StateGraph(DataProductState)
        
        # Add nodes
        workflow.add_node("process_intent", self._run_intent)
        workflow.add_node("process_discovery", self._run_discovery)
        workflow.add_node("process_modeling", self._run_modeling)
        workflow.add_node("process_transformation", self._run_transformation)
        workflow.add_node("process_quality", self._run_quality)
        workflow.add_node("process_packaging", self._run_packaging)
        
        # Define edges
        workflow.set_entry_point("process_intent")
        workflow.add_edge("process_intent", "process_discovery")
        workflow.add_edge("process_discovery", "process_modeling")
        workflow.add_edge("process_modeling", "process_transformation")
        workflow.add_edge("process_transformation", "process_quality")
        workflow.add_edge("process_quality", "process_packaging")
        workflow.add_edge("process_packaging", END)
        
        return workflow.compile()

    def run(self, user_request: str) -> Dict[str, Any]:
        """
        Run the full orchestration flow.
        """
        initial_state = DataProductState(
            user_request=user_request,
            intent=None,
            discovery_result=None,
            data_model=None,
            transformation=None,
            quality_checks=None,
            data_product_spec=None,
            yaml_output=None,
            errors=[]
        )
        
        result = self.workflow.invoke(initial_state)
        return result

    # --- Node Implementations ---
    
    def _run_intent(self, state: DataProductState) -> Dict[str, Any]:
        try:
            intent = self.intent_agent.analyze(state["user_request"])
            return {"intent": intent}
        except Exception as e:
            return {"errors": state["errors"] + [f"Intent Error: {str(e)}"]}

    def _run_discovery(self, state: DataProductState) -> Dict[str, Any]:
        try:
            if not state.get("intent"):
                return {"errors": state["errors"] + ["Skipping discovery: No intent found"]}
            
            discovery_result = self.discovery_agent.discover(state["intent"])
            return {"discovery_result": discovery_result}
        except Exception as e:
            return {"errors": state["errors"] + [f"Discovery Error: {str(e)}"]}

    def _run_modeling(self, state: DataProductState) -> Dict[str, Any]:
        try:
            if not state.get("intent") or not state.get("discovery_result"):
                 return {"errors": state["errors"] + ["Skipping modeling: Missing predecessors"]}

            data_model = self.modeling_agent.design_schema(
                state["intent"], 
                state["discovery_result"].get("candidate_datasets", [])
            )
            return {"data_model": data_model}
        except Exception as e:
             return {"errors": state["errors"] + [f"Modeling Error: {str(e)}"]}

    def _run_transformation(self, state: DataProductState) -> Dict[str, Any]:
        try:
            if not state.get("data_model") or not state.get("discovery_result"):
                return {"errors": state["errors"] + ["Skipping transformation: Missing predecessors"]}
                
            transformation = self.transformation_agent.generate_logic(
                state["data_model"],
                state["discovery_result"].get("candidate_datasets", [])
            )
            return {"transformation": transformation}
        except Exception as e:
            return {"errors": state["errors"] + [f"Transformation Error: {str(e)}"]}

    def _run_quality(self, state: DataProductState) -> Dict[str, Any]:
        try:
            if not state.get("data_model"):
                return {"errors": state["errors"] + ["Skipping quality: Missing data model"]}
                
            checks = self.quality_agent.generate_checks(state["data_model"])
            return {"quality_checks": checks}
        except Exception as e:
            return {"errors": state["errors"] + [f"Quality Error: {str(e)}"]}

    def _run_packaging(self, state: DataProductState) -> Dict[str, Any]:
        try:
            # We can package even with partial results, but ideally we have at least intent and metadata
            result = self.packaging_agent.package(state)
            return result
        except Exception as e:
            return {"errors": state["errors"] + [f"Packaging Error: {str(e)}"]}

