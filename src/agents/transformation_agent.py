from typing import Any, Dict, List
from src.utils.llm_client import LLMClient

class TransformationAgent:
    """
    Agent responsible for generating the transformation logic (SQL code).
    """
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    def generate_logic(self, data_model: Dict[str, Any], source_datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate transformation logic (SQL) to populate the target data model.
        
        Args:
            data_model: Dictionary defining the target schema (output of ModelingAgent).
            source_datasets: List of source dataset schemas (output of DiscoveryAgent).
            
        Returns:
            Dictionary containing 'sql_code' and 'explanation'.
        """
        prompt = self._build_prompt(data_model, source_datasets)
        
        response_schema = {
            "type": "object",
            "properties": {
                "sql_code": {
                    "type": "string",
                    "description": "Valid SQL query to generate the target table."
                },
                "explanation": {
                    "type": "string",
                    "description": "Brief explanation of the transformation logic."
                }
            },
            "required": ["sql_code"]
        }
        
        result = self.llm.generate_structured_output(
            prompt,
            response_schema=response_schema,
            system_prompt="You are a SQL expert. Write efficient SQL to transform data."
        )
        
        return result or {"sql_code": "", "explanation": "Failed to generate code."}
    
    def _build_prompt(self, data_model: Dict, source_datasets: List[Dict]) -> str:
        sources_str = ""
        for ds in source_datasets:
            name = ds.get("name", "unknown")
            columns = ds.get("columns", [])
            sources_str += f"- Table: {name}\n  Columns: {columns}\n"
            
        target_str = f"Target Table: {data_model.get('target_table')}\n"
        target_str += f"Target Schema: {data_model.get('schema', [])}\n"
        target_str += f"Grain: {data_model.get('grain')}\n"

        return f"""
        Write a SQL query to populate the target table using the source tables.
        
        Source Data:
        {sources_str}
        
        Target Requirement:
        {target_str}
        
        Instructions:
        1. CRITICAL: Select the most appropriate source tables from the provided list.
           - If the request mentions "orders", "sales", or "revenue", prefer the 'orders' table if available.
           - 'marketing_events' tracks ad spend, NOT product sales revenue.
           - If metrics come from different entities (e.g., Revenue from Orders, Cost from Products), you MUST JOIN them.
        2. Select columns from the chosen source tables.
        3. Perform necessary joins (e.g., orders.product_id = products.product_id).
        4. Apply aggregations to match the target grain.
        5. Alias columns to match the target schema exactly.
        6. Use standard ANSI SQL (DuckDB compatible).
           - CRITICAL: For date formatting, use `strftime(date_column, '%Y-%m')`.
           - DO NOT use `DATE_FORMAT` (it is not supported in DuckDB).
        6. CRITICAL: Use table aliases (e.g., t1.col) for EVERY column to prevent ambiguity errors.
           - NEVER use reserved keywords as aliases (e.g., 'is', 'as', 'on', 'user', 'from', 'to').
           - Use simple aliases like 't1', 't2', 't3' or descriptive abbreviations (e.g., 'cust', 'prod').
        7. CRITICAL: Return ONLY a SELECT statement (or CTEs + SELECT).
        8. DO NOT use INSERT, UPDATE, DELETE, or CREATE TABLE statements.
        
        Respond in valid JSON matching the schema.
        """
