from typing import Any, Dict, List
from src.knowledge_graph.queries import KnowledgeGraphQueryService

class DiscoveryAgent:
    """
    Agent responsible for finding relevant datasets using the Knowledge Graph.
    """
    
    def __init__(self, kg_service: KnowledgeGraphQueryService):
        self.kg_service = kg_service
    
    def discover(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Discover datasets based on structured intent.
        
        Args:
            intent: Structured intent dictionary containing business_metrics and dimensions.
            
        Returns:
            Dictionary containing candidate and selected datasets.
        """
        metrics = intent.get("business_metrics", [])
        dimensions = intent.get("dimensions", [])
        
        
        # Add ML parameters to search scope if present
        ml_params = intent.get("ml_parameters", {})
        if ml_params and ml_params.get("target_variable"):
            metrics.append(ml_params.get("target_variable"))
            
        # Step 1: Query Knowledge Graph
        # We pass both metrics and dimensions to find datasets that cover as many terms as possible
        candidate_datasets = self.kg_service.find_datasets_for_metrics(
            metrics=metrics,
            dimensions=dimensions
        )
        
        # --- Heuristics: Force-include core datasets based on keywords ---
        # This fixes recall issues where embeddings might miss the "orders" table for "revenue"
        all_text = " ".join(metrics + dimensions).lower()
        forced_additions = []
        
        rule_map = {
            "orders": ["order", "revenue", "sales", "transaction"],
            "products": ["product", "item", "sku", "category", "cost", "price"],
            "customers": ["customer", "user", "client", "buyer"],
            "marketing_campaigns": ["campaign", "ad", "budget"],
        }
        
        existing_names = {ds.get("name") for ds in candidate_datasets}
        
        for dataset_name, keywords in rule_map.items():
            if any(k in all_text for k in keywords):
                if dataset_name not in existing_names:
                    # Fetch dataset schema explicitly
                    ds_schema = self.kg_service.get_dataset_schema(dataset_name)
                    if ds_schema:
                        ds_schema["relevance_score"] = 0.95 # High score for heuristic match
                        candidate_datasets.append(ds_schema)
                        existing_names.add(dataset_name)
        
        # Step 2: Select top candidates (Simple logic for now: take top 5)
        # In a real system, we might apply access control or more complex filtering here
        selected_datasets = [ds["name"] for ds in candidate_datasets[:5]]
        
        return {
            "candidate_datasets": candidate_datasets,
            "selected_datasets": selected_datasets
        }
