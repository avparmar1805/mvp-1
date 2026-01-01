from typing import Dict, List, Any, Optional
from src.utils.llm_client import LLMClient
# from src.knowledge_graph.queries import KnowledgeGraphClient 
# TODO: Import KG client when available

class IntentAgent:
    """
    Agent responsible for parsing natural language requests into structured business requirements.
    """
    
    def __init__(self, llm_client: LLMClient, kg_client: Any = None):
        self.llm = llm_client
        self.kg = kg_client
    
    def analyze(self, user_request: str) -> Dict[str, Any]:
        """
        Analyze the user request to extract business metrics, dimensions, and other intent details.
        
        Args:
            user_request (str): The natural language request from the user.
            
        Returns:
            Dict[str, Any]: Structured intent dictionary.
        """
        # Step 1: Extract structured intent using LLM
        prompt = self._build_prompt(user_request)
        
        # Define expected schema for LLM output
        response_schema = {
            "type": "object",
            "properties": {
                "task_type": {
                    "type": "string",
                    "enum": ["ANALYTICS", "ML"],
                    "description": "Type of task: 'ANALYTICS' for SQL reports, 'ML' for predictive/prescriptive tasks."
                },
                "business_metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of business metrics to calculate (e.g. revenue, sales)"
                },
                "dimensions": {
                    "type": "array",
                    "items": {"type": "string"}, 
                    "description": "List of dimensions to group by (e.g. region, category)"
                },
                "temporal_granularity": {
                    "type": "string",
                    "description": "Time granularity (daily, weekly, monthly, etc.)"
                },
                "ml_parameters": {
                    "type": "object",
                    "properties": {
                        "target_variable": {"type": "string"},
                        "model_type": {"type": "string", "enum": ["FORECASTING", "CLUSTERING", "SENTIMENT"]},
                        "horizon": {"type": "string"}
                    },
                    "description": "Parameters specific to ML tasks"
                },
                "filters": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of filter conditions"
                }
            },
            "required": ["task_type", "business_metrics", "dimensions", "temporal_granularity"]
        }
        
        llm_output = self.llm.generate_structured_output(
            prompt,
            response_schema=response_schema,
            system_prompt="You are a business expert. Classify the request and extract requirements."
        )
        
        # Default empty values if LLM fails
        if not llm_output:
            llm_output = {
                "task_type": "ANALYTICS",
                "business_metrics": [],
                "dimensions": [],
                "temporal_granularity": None,
                "filters": []
            }

        # Step 2: Map to business terms (Placeholder)
        business_terms = [] 

        # Step 3: Calculate confidence
        confidence = self._calculate_confidence(llm_output, business_terms)
        
        return {
            "task_type": llm_output.get("task_type", "ANALYTICS"),
            "business_metrics": llm_output.get("business_metrics", []),
            "dimensions": llm_output.get("dimensions", []),
            "temporal_granularity": llm_output.get("temporal_granularity"),
            "ml_parameters": llm_output.get("ml_parameters", {}),
            "filters": llm_output.get("filters", []),
            "business_terms": business_terms,
            "confidence_score": confidence
        }
    
    def _build_prompt(self, user_request: str) -> str:
        return f"""
        Analyze the following data request:
        Request: "{user_request}"
        
        Tasks:
        1. Classify as "ANALYTICS" (historical reporting) or "ML" (future prediction, segmentation, sentiment).
           - CRITICAL RULE: If the user asks to PREDICT, FORECAST, CLUSTER, SEGMENT, or RECOMMEND, you MUST set task_type="ML".
        2. Extract metrics, dimensions, and filters.
        3. If ML, extract target variable and model type.
           - "Predict revenue" -> FORECASTING
           - "Cluster/Segment customers" -> CLUSTERING
           - "Analyze sentiment" -> SENTIMENT
        
        Respond in valid JSON.
        """

    def _calculate_confidence(self, llm_output: Dict, business_terms: List) -> float:
        """
        Calculate a confidence score based on the completeness of extraction.
        """
        score = 1.0
        
        # Penalty if no metrics found
        if not llm_output.get("business_metrics"):
            score -= 0.5
            
        # Penalty if granularity is missing when it likely should be there (heuristic)
        # For now, just a small penalty
        if not llm_output.get("temporal_granularity"):
            score -= 0.1
            
        return max(0.0, score)
