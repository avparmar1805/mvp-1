from typing import Dict, Any, List
import yaml
import datetime

class PackagingAgent:
    """
    Agent responsible for packaging the final Data Product Specification.
    It aggregates outputs from all previous agents and formats them into a standardized spec.
    """
    
    def __init__(self):
        # In the future, we could inject a schema validator here
        pass
        
    def package(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compile the final Data Product Specification from the pipeline state.
        
        Args:
            state: The current state of the Orchestrator, containing outputs from all agents.
            
        Returns:
            Dictionary containing the structured 'data_product_spec' and 'yaml_output'.
        """
        intent = state.get("intent", {})
        discovery = state.get("discovery_result", {})
        data_model = state.get("data_model", {})
        transformation = state.get("transformation", {})
        quality = state.get("quality_checks", {})
        
        # 1. Generate Metadata
        metadata = self._generate_metadata(intent, data_model)
        
        # 2. Build Specification Structure based on Roadmap JSON Schema
        spec = {
            "metadata": metadata,
            "business_context": {
                "request": state.get("user_request", ""),
                "intent": {
                    "metrics": intent.get("business_metrics", []),
                    "dimensions": intent.get("dimensions", []),
                    "granularity": intent.get("temporal_granularity")
                }
            },
            "data_model": {
                "target_table": data_model.get("target_table"),
                "grain": data_model.get("grain"),
                "schema": data_model.get("schema", []),
                "primary_keys": data_model.get("primary_keys", [])
            },
            "source_data": {
                "datasets": discovery.get("selected_datasets", [])
            },
            "transformation": {
                "type": "sql",
                "code": transformation.get("sql_code", ""),
                "explanation": transformation.get("explanation", "")
            },
            "quality_assurance": {
                "rules": quality.get("quality_checks", [])
            }
        }
        
        # 3. generate YAML
        yaml_output = yaml.dump(spec, sort_keys=False, default_flow_style=False)
        
        return {
            "data_product_spec": spec,
            "yaml_output": yaml_output
        }
        
    def _generate_metadata(self, intent: Dict, data_model: Dict) -> Dict[str, Any]:
        """Generate metadata for the data product."""
        # Clean table name to create a display name title
        table_name = data_model.get("target_table", "data_product")
        # Remove 'gold.' prefix if exists and replace underscores with spaces
        name = table_name.replace("gold.", "").replace("_", " ").title()
        
        return {
            "name": name,
            "version": "1.0.0",
            "description": f"Data product for {intent.get('temporal_granularity', 'ad-hoc')} analysis of {', '.join(intent.get('business_metrics', []))}.",
            "owner": "Agentic Builder",
            "created_at": datetime.datetime.now().isoformat()
        }
