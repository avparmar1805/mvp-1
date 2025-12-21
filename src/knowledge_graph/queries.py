"""
Knowledge Graph Query Functions

Provides high-level query functions for common operations
used by the discovery and modeling agents.
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from src.knowledge_graph.client import NetworkXKnowledgeGraph
from src.knowledge_graph.embeddings import EmbeddingGenerator


class KnowledgeGraphQueryService:
    """
    High-level query service for the knowledge graph.
    
    Provides methods for dataset discovery, column search,
    and business term lookups.
    """
    
    def __init__(self, kg: NetworkXKnowledgeGraph):
        """
        Initialize the query service.
        
        Args:
            kg: Knowledge graph client
        """
        self.kg = kg
        self.embedding_generator = EmbeddingGenerator()
    
    def find_datasets_for_metrics(
        self,
        metrics: List[str],
        dimensions: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find datasets that can provide the requested metrics and dimensions.
        
        Args:
            metrics: List of business metrics (e.g., ["revenue", "order_count"])
            dimensions: List of dimensions (e.g., ["region", "category"])
            
        Returns:
            List of datasets with relevance scores
        """
        all_terms = metrics + (dimensions or [])
        
        logger.info(f"Finding datasets for: {all_terms}")
        
        # Use the KG's term-based search
        results = self.kg.find_datasets_by_terms(all_terms)
        
        # Fallback: If no results found, try semantic search for each term
        if not results:
            logger.info("No exact matches found. Trying semantic search...")
            found_datasets = {}
            
            for term in all_terms:
                semantic_results = self.semantic_column_search(term, top_k=3)
                for res in semantic_results:
                    # 'res' is a column node dict, needs to be mapped to dataset
                    ds_name = res.get("dataset_name")
                    if ds_name and ds_name not in found_datasets:
                        # fetch dataset node
                        ds_node = self.kg.get_dataset(ds_name)
                        if ds_node:
                            # Add a relevance score based on semantic similarity
                            ds_node["relevance_score"] = res.get("similarity", 0.5)
                            found_datasets[ds_name] = ds_node
            
            results = list(found_datasets.values())

        # Enhance results with column details
        for result in results:
            ds_name = result.get("name")
            if ds_name:
                columns = self.kg.get_dataset_columns(ds_name)
                result["columns"] = [c.get("name") for c in columns]
                result["column_count"] = len(columns)
        
        logger.info(f"Found {len(results)} relevant datasets")
        
        return results
    
    def get_columns_for_term(self, term: str) -> List[Dict[str, Any]]:
        """
        Get all columns that map to a business term.
        
        Args:
            term: Business term (e.g., "revenue")
            
        Returns:
            List of column details
        """
        return self.kg.get_columns_by_term(term)
    
    def semantic_column_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search for columns using semantic similarity.
        
        Args:
            query: Natural language query
            top_k: Number of results to return
            
        Returns:
            List of columns with similarity scores
        """
        return self.kg.search_by_query(query, top_k)
    
    def get_dataset_schema(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get full schema information for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dataset details with column schemas
        """
        dataset = self.kg.get_dataset(dataset_name)
        if not dataset:
            return {}
        
        columns = self.kg.get_dataset_columns(dataset_name)
        
        return {
            **dataset,
            "columns": columns,
        }
    
    def find_join_paths(
        self,
        source_dataset: str,
        target_dataset: str,
    ) -> List[Dict[str, Any]]:
        """
        Find possible join paths between two datasets.
        
        Args:
            source_dataset: Source dataset name
            target_dataset: Target dataset name
            
        Returns:
            List of possible join paths
        """
        paths = []
        
        source_cols = self.kg.get_dataset_columns(source_dataset)
        target_cols = self.kg.get_dataset_columns(target_dataset)
        
        # Look for FK relationships
        for col in source_cols:
            if col.get("is_foreign_key") and col.get("references"):
                ref_parts = col.get("references", "").split(".")
                if len(ref_parts) == 2:
                    ref_table = ref_parts[0]
                    ref_col = ref_parts[1]
                    
                    if ref_table == target_dataset:
                        paths.append({
                            "type": "direct",
                            "source_column": col["name"],
                            "target_column": ref_col,
                            "join_type": "inner",
                        })
        
        # Look for common column names (potential join keys)
        source_col_names = {c["name"] for c in source_cols}
        target_col_names = {c["name"] for c in target_cols}
        
        common_cols = source_col_names & target_col_names
        for col_name in common_cols:
            if col_name.endswith("_id"):
                paths.append({
                    "type": "common_key",
                    "column": col_name,
                    "join_type": "inner",
                })
        
        return paths
    
    def get_related_datasets(
        self,
        dataset_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Find datasets related to a given dataset via FK relationships.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            List of related datasets with relationship info
        """
        related = []
        columns = self.kg.get_dataset_columns(dataset_name)
        
        for col in columns:
            if col.get("is_foreign_key") and col.get("references"):
                ref_parts = col.get("references", "").split(".")
                if len(ref_parts) == 2:
                    ref_table = ref_parts[0]
                    
                    ref_dataset = self.kg.get_dataset(ref_table)
                    if ref_dataset:
                        related.append({
                            "dataset": ref_table,
                            "relationship": "references",
                            "via_column": col["name"],
                            "referenced_column": ref_parts[1],
                        })
        
        return related
    
    def get_business_terms(self) -> List[Dict[str, Any]]:
        """
        Get all business terms in the knowledge graph.
        
        Returns:
            List of business term details
        """
        terms = []
        for term_name, term_obj in self.kg._business_terms.items():
            terms.append(term_obj.to_dict())
        return terms
    
    def explain_column(self, column_full_name: str) -> Dict[str, Any]:
        """
        Get detailed explanation of a column including mapped business terms.
        
        Args:
            column_full_name: Full column name (dataset.column)
            
        Returns:
            Column details with business context
        """
        if column_full_name not in self.kg._columns:
            return {}
        
        column = self.kg._columns[column_full_name]
        result = column.to_dict()
        
        # Find mapped business terms
        mapped_terms = []
        for term_name, cols in self.kg._term_to_columns.items():
            if column_full_name in cols:
                term = self.kg._business_terms.get(term_name)
                if term:
                    mapped_terms.append({
                        "term": term.term,
                        "definition": term.definition,
                        "domain": term.domain,
                    })
        
        result["business_terms"] = mapped_terms
        
        return result
    
    def suggest_metrics_for_dataset(
        self,
        dataset_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Suggest possible business metrics based on dataset columns.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            List of suggested metrics with columns
        """
        suggestions = []
        columns = self.kg.get_dataset_columns(dataset_name)
        
        # Numeric columns can be aggregated
        for col in columns:
            dtype = col.get("data_type", "").lower()
            col_name = col.get("name", "")
            
            if "int" in dtype or "float" in dtype or "decimal" in dtype:
                # Skip ID columns
                if col_name.endswith("_id"):
                    continue
                
                suggestions.append({
                    "column": col_name,
                    "suggested_metrics": [
                        f"SUM({col_name})",
                        f"AVG({col_name})",
                        f"COUNT({col_name})",
                    ],
                    "data_type": dtype,
                })
        
        return suggestions


def create_query_service(
    kg_path: str = "data/knowledge_graph.json",
) -> KnowledgeGraphQueryService:
    """
    Create a query service instance.
    
    Args:
        kg_path: Path to knowledge graph file
        
    Returns:
        KnowledgeGraphQueryService instance
    """
    from src.knowledge_graph.client import create_knowledge_graph_client
    
    kg = create_knowledge_graph_client(
        backend="networkx",
        persist_path=kg_path,
    )
    
    return KnowledgeGraphQueryService(kg)

