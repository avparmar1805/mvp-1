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
                "filters": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of filter conditions"
                }
            },
            "required": ["business_metrics", "dimensions", "temporal_granularity"]
        }
        
        llm_output = self.llm.generate_structured_output(
            prompt,
            response_schema=response_schema,
            system_prompt="You are a business analyst expert. Extract structured requirements from the user request."
        )
        
        # Default empty values if LLM fails
        if not llm_output:
            llm_output = {
                "business_metrics": [],
                "dimensions": [],
                "temporal_granularity": None,
                "filters": []
            }

        # Step 2: Map to business terms in KG (Placeholder)
        # business_terms = self.kg.find_business_terms(llm_output.get("business_metrics", [])) if self.kg else []
        business_terms = [] # Placeholder until KG is integrated

        # Step 3: Calculate confidence (Placeholder logic)
        confidence = self._calculate_confidence(llm_output, business_terms)
        
        return {
            "business_metrics": llm_output.get("business_metrics", []),
            "dimensions": llm_output.get("dimensions", []),
            "temporal_granularity": llm_output.get("temporal_granularity"),
            "filters": llm_output.get("filters", []),
            "business_terms": business_terms,
            "confidence_score": confidence
        }
    
    def _build_prompt(self, user_request: str) -> str:
        return f"""
        Analyze the following data request:

        Request: "{user_request}"

        Extract:
        1. Business metrics to calculate (e.g., revenue, count, average)
        2. Dimensions for grouping (e.g., region, category, date)
        3. Time granularity (hourly, daily, weekly, monthly, none)
        4. Any specific filters or conditions

        Respond in valid JSON format matching the schema.
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
