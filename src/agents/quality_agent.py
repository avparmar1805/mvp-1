from typing import Any, Dict, List
from src.utils.llm_client import LLMClient

class QualityAgent:
    """
    Agent responsible for generating data quality checks (Expectations).
    """
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    def generate_checks(self, data_model: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate data quality checks based on the target schema.
        
        Args:
            data_model: Dictionary defining the target schema (output of ModelingAgent).
            
        Returns:
            Dictionary containing a list of 'quality_checks'.
        """
        prompt = self._build_prompt(data_model)
        
        response_schema = {
            "type": "object",
            "properties": {
                "quality_checks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "check_type": {"type": "string"},
                            "column": {"type": "string"},
                            "description": {"type": "string"}
                        },
                        "required": ["check_type", "column"]
                    },
                    "description": "List of data quality expectations."
                }
            },
            "required": ["quality_checks"]
        }
        
        result = self.llm.generate_structured_output(
            prompt,
            response_schema=response_schema,
            system_prompt="You are a Data Quality engineer. Suggest automated checks for data reliability."
        )
        
        return result or {"quality_checks": []}
    
    def _build_prompt(self, data_model: Dict) -> str:
        schema_str = ""
        for col in data_model.get("schema", []):
            name = col.get("name", "unknown")
            dtype = col.get("type", "unknown")
            constraints = []
            if col.get("primary_key"):
                constraints.append("Primary Key")
            if not col.get("nullable", True):
                constraints.append("Not Null")
            
            schema_str += f"- Column: {name} ({dtype}) [{', '.join(constraints)}]\n"

        return f"""
        Suggest data quality checks (Expectations) for the following table schema.
        
        Table: {data_model.get('target_table')}
        
        Schema:
        {schema_str}
        
        Instructions:
        1. Suggest standard checks like 'not_null', 'unique', 'min_value', 'accepted_values'.
        2. Ensure Primary Keys are unique and not null.
        3. Suggest range checks for numeric metrics (e.g. revenue >= 0).
        4. Suggest valid value checks for categorical dimensions if identifiable.
        
        Respond in valid JSON matching the schema.
        """
