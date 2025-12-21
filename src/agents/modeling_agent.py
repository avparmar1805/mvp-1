from typing import Any, Dict, List
from src.utils.llm_client import LLMClient

class ModelingAgent:
    """
    Agent responsible for designing the target data schema.
    """
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    def design_schema(self, intent: Dict[str, Any], available_datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Design the target schema based on intent and available datasets.
        
        Args:
            intent: Structured intent dictionary.
            available_datasets: List of dataset dictionaries containing schema info.
            
        Returns:
            Dictionary defining the target data model.
        """
        prompt = self._build_prompt(intent, available_datasets)
        
        response_schema = {
            "type": "object",
            "properties": {
                "target_table": {"type": "string"},
                "grain": {"type": "string"},
                "schema": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string"},
                            "description": {"type": "string"},
                            "nullable": {"type": "boolean"},
                            "primary_key": {"type": "boolean"}
                        },
                        "required": ["name", "type"]
                    }
                },
                "primary_keys": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["target_table", "grain", "schema"]
        }
        
        result = self.llm.generate_structured_output(
            prompt, 
            response_schema=response_schema,
            system_prompt="You are a data architect expert. Design a target schema optimized for analytics."
        )
        
        return result or {}

    def _build_prompt(self, intent: Dict, datasets: List[Dict]) -> str:
        # Format dataset schemas for context
        datasets_str = ""
        for ds in datasets:
            name = ds.get("name", "unknown")
            # Handle different formats of column info if necessary
            columns = ds.get("columns", [])
            datasets_str += f"- Dataset: {name}\n  Columns: {columns}\n"

        return f"""
        Design a target data model for the following request:
        
        Business Metrics: {intent.get('business_metrics')}
        Dimensions: {intent.get('dimensions')}
        Target Granularity: {intent.get('temporal_granularity')}
        
        Available Source Data:
        {datasets_str}
        
        Requirements:
        1. Define a target table name (start with 'gold.')
        2. Define the grain of the table
        3. List all columns with data types (use standard SQL types like VARCHAR, DECIMAL, INTEGER, DATE)
        4. Identify primary keys based on the grain
        
        Respond in valid JSON matching the schema.
        """
