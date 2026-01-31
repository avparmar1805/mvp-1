from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from src.agents.intent_agent import IntentAgent
from src.agents.discovery_agent import DiscoveryAgent
from src.agents.modeling_agent import ModelingAgent
from src.agents.transformation_agent import TransformationAgent
from src.agents.quality_agent import QualityAgent
from src.agents.packaging_agent import PackagingAgent
from src.agents.workflow_agent import WorkflowAgent
from src.utils.llm_client import LLMClient
from src.knowledge_graph.queries import KnowledgeGraphQueryService, create_query_service
from src.agents.ml_agent import MachineLearningAgent
from src.utils.execution_engine import ExecutionEngine
from src.knowledge_graph.usage_analytics import UsageAnalyticsService
from src.knowledge_graph.learning_engine import LearningEngine

# Define the state of the data product generation workflow
class DataProductState(TypedDict):
    user_request: str
    intent: Optional[Dict[str, Any]]
    discovery_result: Optional[Dict[str, Any]]
    data_model: Optional[Dict[str, Any]]
    transformation: Optional[Dict[str, Any]]
    quality_checks: Optional[Dict[str, Any]]
    ml_result: Optional[Dict[str, Any]]  # New: Store ML results
    data_product_spec: Optional[Dict[str, Any]]
    yaml_output: Optional[str]
    workflow_result: Optional[Dict[str, Any]]  # New: Store workflow generation results
    errors: List[str]

class OrchestratorAgent:
    """
    Coordinator agent that manages the workflow between Intent, Discovery, Modeling, Transformation, Quality, and ML agents.
    """
    
    def __init__(self):
        # Initialize dependencies
        self.llm_client = LLMClient()
        self.kg_service = create_query_service()
        
        # Initialize sub-agents
        self.intent_agent = IntentAgent(self.llm_client, self.kg_service)
        self.discovery_agent = DiscoveryAgent(self.kg_service)
        self.modeling_agent = ModelingAgent(self.llm_client)
        self.transformation_agent = TransformationAgent(self.llm_client)
        self.quality_agent = QualityAgent(self.llm_client)
        self.ml_agent = MachineLearningAgent(self.llm_client) # New
        self.packaging_agent = PackagingAgent()
        self.workflow_agent = WorkflowAgent()  # New: Workflow generation
        
        # Initialize Execution Engine for ML data fetching
        self.execution_engine = ExecutionEngine()
        
        # Initialize Learning components
        self.usage_analytics = UsageAnalyticsService()
        self.learning_engine = LearningEngine(self.usage_analytics, self.kg_service.kg)
        
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
        workflow.add_node("process_ml", self._run_ml) # New node
        workflow.add_node("process_packaging", self._run_packaging)
        workflow.add_node("process_workflow", self._run_workflow)  # New: Workflow generation
        
        # Define edges
        workflow.set_entry_point("process_intent")
        workflow.add_edge("process_intent", "process_discovery")
        workflow.add_edge("process_discovery", "process_modeling")
        workflow.add_edge("process_modeling", "process_transformation")
        
        # Conditional branching based on task_type
        def route_after_transform(state: DataProductState):
            task_type = state.get("intent", {}).get("task_type", "ANALYTICS")
            if task_type == "ML":
                return "process_ml"
            return "process_quality"

        workflow.add_conditional_edges(
            "process_transformation",
            route_after_transform,
            {
                "process_ml": "process_ml",
                "process_quality": "process_quality"
            }
        )
        
        workflow.add_edge("process_quality", "process_packaging")
        workflow.add_edge("process_ml", "process_packaging") # ML also goes to packaging
        workflow.add_edge("process_packaging", "process_workflow")  # Packaging -> Workflow
        workflow.add_edge("process_workflow", END)  # Workflow is the final step
        
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
            ml_result=None,
            data_product_spec=None,
            yaml_output=None,
            workflow_result=None,
            errors=[]
        )
        
        result = self.workflow.invoke(initial_state)
        
        # Record usage for learning (if successful)
        if not result.get("errors") and result.get("discovery_result"):
            self._record_usage(user_request, result)
        
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
            
    def _run_ml(self, state: DataProductState) -> Dict[str, Any]:
        """
        Execute ML workflow: Fetch Data -> Generate Code -> Execute Code.
        """
        try:
            # 1. Fetch Data using Execution Engine
            sql = state["transformation"]["sql_code"]
            success, df, msg = self.execution_engine.execute_query_df(sql)
            
            if not success or df is None or df.empty:
               return {"errors": state["errors"] + [f"ML Data Fetch Failed: {msg}"]}
               
            # 2. Generate Python Code
            intent_str = f"{state['user_request']} (Parameters: {state['intent'].get('ml_parameters')})"
            code_result = self.ml_agent.generate_code(intent_str, df)
            
            # 3. Execute Python Code
            exec_result = self.ml_agent.execute_code(code_result["python_code"], df)
            
            if not exec_result["success"]:
                 return {"errors": state["errors"] + [f"ML Execution Failed: {exec_result.get('error')}"]}
            
            return {
                "ml_result": {
                    "code": code_result["python_code"],
                    "explanation": code_result["explanation"],
                    "plot": exec_result.get("plot_base64"),
                    "output_summary": str(exec_result.get("result"))
                }
            }
        except Exception as e:
            return {"errors": state["errors"] + [f"ML Error: {str(e)}"]}

    def _run_packaging(self, state: DataProductState) -> Dict[str, Any]:
        try:
            # We can package even with partial results, but ideally we have at least intent and metadata
            result = self.packaging_agent.package(state)
            return result
        except Exception as e:
            return {"errors": state["errors"] + [f"Packaging Error: {str(e)}"]}
    
    def _run_workflow(self, state: DataProductState) -> Dict[str, Any]:
        """
        Generate workflow orchestration (Airflow DAG and cron job) from data product spec.
        """
        try:
            spec = state.get("data_product_spec")
            if not spec:
                return {"errors": state["errors"] + ["Skipping workflow: No data product spec found"]}
            
            # Generate a unique data product ID from metadata
            metadata = spec.get("metadata", {})
            name = metadata.get("name", "unnamed_product")
            version = metadata.get("version", "1.0.0")
            data_product_id = f"{name}_{version}".replace(" ", "_").replace(".", "_")
            
            # Generate Airflow DAG
            dag_code = self.workflow_agent.generate_airflow_dag(
                data_product_spec=spec,
                data_product_id=data_product_id
            )
            
            # Generate cron job script
            cron_code = self.workflow_agent.generate_cron_job(
                data_product_spec=spec,
                data_product_id=data_product_id
            )
            
            # Extract schedule information
            sla = spec.get("sla", {})
            schedule = self.workflow_agent._extract_schedule(sla)
            
            return {
                "workflow_result": {
                    "dag_code": dag_code,
                    "cron_code": cron_code,
                    "schedule": schedule,
                    "data_product_id": data_product_id,
                    "dag_file": f"generated_workflows/dags/{self.workflow_agent._sanitize_dag_id(name, version)}.py",
                    "cron_file": f"generated_workflows/cron/{name.replace(' ', '_')}.sh"
                }
            }
        except Exception as e:
            return {"errors": state["errors"] + [f"Workflow Error: {str(e)}"]}
    
    def _record_usage(self, user_request: str, result: Dict[str, Any]):
        """
        Record usage event for Knowledge Graph learning
        
        Args:
            user_request: Original user query
            result: Pipeline execution result
        """
        try:
            # Extract datasets from discovery result
            discovery = result.get("discovery_result", {})
            datasets = discovery.get("selected_datasets", [])
            
            # Handle both string and dict formats
            dataset_names = []
            for ds in datasets:
                if isinstance(ds, str):
                    dataset_names.append(ds)
                elif isinstance(ds, dict):
                    dataset_names.append(ds.get("name", ""))
            
            # Get data product ID if available
            spec = result.get("data_product_spec", {})
            metadata = spec.get("metadata", {})
            data_product_id = metadata.get("name", "").replace(" ", "_")
            
            # Record usage event
            if dataset_names:
                self.learning_engine.process_usage_event(
                    query=user_request,
                    selected_datasets=dataset_names,
                    data_product_id=data_product_id
                )
        except Exception as e:
            # Don't fail the main pipeline if usage tracking fails
            import logging
            logging.warning(f"Failed to record usage: {e}")



