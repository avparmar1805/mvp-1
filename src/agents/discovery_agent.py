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
        
        # Step 1: Query Knowledge Graph
        # We pass both metrics and dimensions to find datasets that cover as many terms as possible
        candidate_datasets = self.kg_service.find_datasets_for_metrics(
            metrics=metrics,
            dimensions=dimensions
        )
        
        # Step 2: Select top candidates (Simple logic for now: take top 5)
        # In a real system, we might apply access control or more complex filtering here
        selected_datasets = [ds["name"] for ds in candidate_datasets[:5]]
        
        return {
            "candidate_datasets": candidate_datasets,
            "selected_datasets": selected_datasets
        }
