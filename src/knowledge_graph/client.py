"""
Knowledge Graph Client

Provides a unified interface for interacting with the knowledge graph,
supporting both Neo4j and NetworkX backends.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
from loguru import logger

from src.knowledge_graph.schema import (
    NodeType,
    RelationshipType,
    DatasetNode,
    ColumnNode,
    BusinessTermNode,
    DomainNode,
    QualityRuleNode,
)
from src.knowledge_graph.embeddings import (
    EmbeddingGenerator,
    cosine_similarity,
    find_similar_embeddings,
)


class KnowledgeGraphClient(ABC):
    """Abstract base class for knowledge graph clients."""
    
    @abstractmethod
    def add_dataset(self, dataset: DatasetNode) -> None:
        """Add a dataset node to the graph."""
        pass
    
    @abstractmethod
    def add_column(self, column: ColumnNode) -> None:
        """Add a column node to the graph."""
        pass
    
    @abstractmethod
    def add_business_term(self, term: BusinessTermNode) -> None:
        """Add a business term node to the graph."""
        pass
    
    @abstractmethod
    def add_relationship(
        self,
        source_type: NodeType,
        source_id: str,
        target_type: NodeType,
        target_id: str,
        relationship_type: RelationshipType,
        properties: Dict[str, Any] = None,
    ) -> None:
        """Add a relationship between nodes."""
        pass
    
    @abstractmethod
    def get_datasets(self) -> List[Dict[str, Any]]:
        """Get all datasets."""
        pass
    
    @abstractmethod
    def get_dataset_columns(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Get columns for a dataset."""
        pass
    
    @abstractmethod
    def find_datasets_by_terms(self, terms: List[str]) -> List[Dict[str, Any]]:
        """Find datasets that contain columns mapped to given business terms."""
        pass
    
    @abstractmethod
    def find_similar_columns(
        self,
        query_embedding: List[float],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find columns with similar embeddings."""
        pass


class NetworkXKnowledgeGraph(KnowledgeGraphClient):
    """
    In-memory Knowledge Graph implementation using NetworkX.
    
    This is a lightweight alternative to Neo4j for development and testing.
    """
    
    def __init__(self, persist_path: Optional[str] = None):
        """
        Initialize the NetworkX knowledge graph.
        
        Args:
            persist_path: Optional path to persist/load the graph
        """
        self.graph = nx.DiGraph()
        self.persist_path = Path(persist_path) if persist_path else None
        self.embedding_generator = EmbeddingGenerator()
        
        # Index structures for fast lookup
        self._datasets: Dict[str, DatasetNode] = {}
        self._columns: Dict[str, ColumnNode] = {}
        self._business_terms: Dict[str, BusinessTermNode] = {}
        self._column_embeddings: List[Tuple[str, List[float]]] = []
        self._term_to_columns: Dict[str, Set[str]] = {}
        
        # Load from disk if path exists
        if self.persist_path and self.persist_path.exists():
            self._load()
        
        logger.info("NetworkX Knowledge Graph initialized")
    
    def add_dataset(self, dataset: DatasetNode) -> None:
        """Add a dataset node to the graph."""
        node_id = f"dataset:{dataset.name}"
        self.graph.add_node(
            node_id,
            node_type=NodeType.DATASET.value,
            **dataset.to_dict(),
        )
        self._datasets[dataset.name] = dataset
        logger.debug(f"Added dataset: {dataset.name}")
    
    def add_column(self, column: ColumnNode) -> None:
        """Add a column node to the graph."""
        node_id = f"column:{column.full_name}"
        self.graph.add_node(
            node_id,
            node_type=NodeType.COLUMN.value,
            **column.to_dict(),
        )
        self._columns[column.full_name] = column
        
        # Add relationship to dataset
        dataset_id = f"dataset:{column.dataset_name}"
        if self.graph.has_node(dataset_id):
            self.graph.add_edge(
                dataset_id,
                node_id,
                relationship_type=RelationshipType.HAS_COLUMN.value,
            )
        
        # Store embedding if available
        if column.embedding:
            self._column_embeddings.append((column.full_name, column.embedding))
        
        logger.debug(f"Added column: {column.full_name}")
    
    def add_business_term(self, term: BusinessTermNode) -> None:
        """Add a business term node to the graph."""
        node_id = f"term:{term.term}"
        self.graph.add_node(
            node_id,
            node_type=NodeType.BUSINESS_TERM.value,
            **term.to_dict(),
        )
        self._business_terms[term.term] = term
        
        # Initialize term-to-columns mapping
        if term.term not in self._term_to_columns:
            self._term_to_columns[term.term] = set()
        
        logger.debug(f"Added business term: {term.term}")
    
    def add_domain(self, domain: DomainNode) -> None:
        """Add a domain node to the graph."""
        node_id = f"domain:{domain.name}"
        self.graph.add_node(
            node_id,
            node_type=NodeType.DOMAIN.value,
            **domain.to_dict(),
        )
        logger.debug(f"Added domain: {domain.name}")
    
    def add_relationship(
        self,
        source_type: NodeType,
        source_id: str,
        target_type: NodeType,
        target_id: str,
        relationship_type: RelationshipType,
        properties: Dict[str, Any] = None,
    ) -> None:
        """Add a relationship between nodes."""
        # Map node types to their prefix format
        type_prefix = {
            NodeType.DATASET: "dataset",
            NodeType.COLUMN: "column",
            NodeType.BUSINESS_TERM: "term",
            NodeType.DOMAIN: "domain",
            NodeType.QUALITY_RULE: "rule",
            NodeType.DATA_PRODUCT: "product",
        }
        
        source_prefix = type_prefix.get(source_type, source_type.value.lower())
        target_prefix = type_prefix.get(target_type, target_type.value.lower())
        
        source_node = f"{source_prefix}:{source_id}"
        target_node = f"{target_prefix}:{target_id}"
        
        if not self.graph.has_node(source_node):
            logger.warning(f"Source node not found: {source_node}")
            return
        
        if not self.graph.has_node(target_node):
            logger.warning(f"Target node not found: {target_node}")
            return
        
        edge_props = {"relationship_type": relationship_type.value}
        if properties:
            edge_props.update(properties)
        
        self.graph.add_edge(source_node, target_node, **edge_props)
        
        # Update term-to-columns mapping for MAPS_TO relationships
        if relationship_type == RelationshipType.MAPS_TO:
            if target_id in self._term_to_columns:
                self._term_to_columns[target_id].add(source_id)
        
        logger.debug(f"Added relationship: {source_node} -[{relationship_type.value}]-> {target_node}")
    
    def map_column_to_term(self, column_full_name: str, term: str) -> None:
        """Map a column to a business term."""
        self.add_relationship(
            NodeType.COLUMN,
            column_full_name,
            NodeType.BUSINESS_TERM,
            term,
            RelationshipType.MAPS_TO,
        )
        
        if term in self._term_to_columns:
            self._term_to_columns[term].add(column_full_name)
    
    def get_datasets(self) -> List[Dict[str, Any]]:
        """Get all datasets."""
        datasets = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("node_type") == NodeType.DATASET.value:
                datasets.append(data)
        return datasets
    
    def get_dataset(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a specific dataset by name."""
        node_id = f"dataset:{name}"
        if self.graph.has_node(node_id):
            return dict(self.graph.nodes[node_id])
        return None
    
    def get_dataset_columns(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Get columns for a dataset."""
        columns = []
        dataset_id = f"dataset:{dataset_name}"
        
        if not self.graph.has_node(dataset_id):
            return columns
        
        for _, target, data in self.graph.out_edges(dataset_id, data=True):
            if data.get("relationship_type") == RelationshipType.HAS_COLUMN.value:
                column_data = dict(self.graph.nodes[target])
                columns.append(column_data)
        
        return columns
    
    def get_columns_by_term(self, term: str) -> List[Dict[str, Any]]:
        """Get columns mapped to a business term."""
        columns = []
        
        # Check exact term match
        if term in self._term_to_columns:
            for col_name in self._term_to_columns[term]:
                if col_name in self._columns:
                    columns.append(self._columns[col_name].to_dict())
        
        # Check synonyms
        for bt in self._business_terms.values():
            if term.lower() in [s.lower() for s in bt.synonyms]:
                if bt.term in self._term_to_columns:
                    for col_name in self._term_to_columns[bt.term]:
                        if col_name in self._columns:
                            col_dict = self._columns[col_name].to_dict()
                            if col_dict not in columns:
                                columns.append(col_dict)
        
        return columns
    
    def find_datasets_by_terms(self, terms: List[str]) -> List[Dict[str, Any]]:
        """Find datasets that contain columns mapped to given business terms."""
        dataset_scores: Dict[str, Dict[str, Any]] = {}
        
        for term in terms:
            term_lower = term.lower()
            
            # Find columns for this term
            columns = self.get_columns_by_term(term_lower)
            
            # Also search by synonym
            for bt in self._business_terms.values():
                if term_lower in [s.lower() for s in bt.synonyms] or term_lower == bt.term.lower():
                    columns.extend(self.get_columns_by_term(bt.term))
            
            # Aggregate by dataset
            for col in columns:
                ds_name = col.get("dataset_name")
                if ds_name:
                    if ds_name not in dataset_scores:
                        ds_data = self.get_dataset(ds_name)
                        if ds_data:
                            dataset_scores[ds_name] = {
                                **ds_data,
                                "matched_columns": [],
                                "matched_terms": set(),
                                "relevance_score": 0.0,
                            }
                    
                    if ds_name in dataset_scores:
                        dataset_scores[ds_name]["matched_columns"].append(col["name"])
                        dataset_scores[ds_name]["matched_terms"].add(term)
        
        # Calculate relevance scores
        for ds_name, ds_data in dataset_scores.items():
            n_matched_terms = len(ds_data["matched_terms"])
            n_matched_cols = len(set(ds_data["matched_columns"]))
            quality_score = ds_data.get("quality_score", 1.0)
            
            # Score = term coverage (40%) + column count (30%) + quality (30%)
            term_coverage = n_matched_terms / len(terms) if terms else 0
            col_score = min(n_matched_cols / 5, 1.0)  # Cap at 5 columns
            
            ds_data["relevance_score"] = (
                term_coverage * 0.4 +
                col_score * 0.3 +
                quality_score * 0.3
            )
            ds_data["matched_terms"] = list(ds_data["matched_terms"])
        
        # Sort by relevance
        results = sorted(
            dataset_scores.values(),
            key=lambda x: x["relevance_score"],
            reverse=True,
        )
        
        return results
    
    def find_similar_columns(
        self,
        query_embedding: List[float],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find columns with similar embeddings."""
        if not self._column_embeddings:
            logger.warning("No column embeddings available")
            return []
        
        similar = find_similar_embeddings(
            query_embedding,
            self._column_embeddings,
            top_k=top_k,
        )
        
        results = []
        for col_name, score in similar:
            if col_name in self._columns:
                col_data = self._columns[col_name].to_dict()
                col_data["similarity_score"] = score
                results.append(col_data)
        
        return results
    
    def search_by_query(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant columns using semantic search.
        
        Args:
            query: Natural language query
            top_k: Number of results to return
            
        Returns:
            List of relevant columns with similarity scores
        """
        if not self.embedding_generator.is_available:
            logger.warning("Embeddings not available, falling back to keyword search")
            return self._keyword_search(query, top_k)
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        if query_embedding is None:
            return self._keyword_search(query, top_k)
        
        return self.find_similar_columns(query_embedding, top_k)
    
    def _keyword_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Fallback keyword-based search."""
        query_terms = query.lower().split()
        results = []
        
        for col_name, column in self._columns.items():
            score = 0
            for term in query_terms:
                if term in col_name.lower():
                    score += 2
                if term in column.description.lower():
                    score += 1
            
            if score > 0:
                col_data = column.to_dict()
                col_data["relevance_score"] = score
                results.append(col_data)
        
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:top_k]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "datasets": len(self._datasets),
            "columns": len(self._columns),
            "business_terms": len(self._business_terms),
            "columns_with_embeddings": len(self._column_embeddings),
        }
    
    def save(self, path: Optional[str] = None) -> None:
        """Save the graph to disk."""
        save_path = Path(path) if path else self.persist_path
        if not save_path:
            logger.warning("No persist path specified")
            return
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save graph structure
        graph_data = nx.node_link_data(self.graph)
        
        # Save index structures
        data = {
            "graph": graph_data,
            "datasets": {k: v.to_dict() for k, v in self._datasets.items()},
            "columns": {k: v.to_dict() for k, v in self._columns.items()},
            "business_terms": {k: v.to_dict() for k, v in self._business_terms.items()},
            "column_embeddings": self._column_embeddings,
            "term_to_columns": {k: list(v) for k, v in self._term_to_columns.items()},
        }
        
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Saved knowledge graph to {save_path}")
    
    def _load(self) -> None:
        """Load the graph from disk."""
        if not self.persist_path or not self.persist_path.exists():
            return
        
        with open(self.persist_path, "r") as f:
            data = json.load(f)
        
        # Restore graph structure
        self.graph = nx.node_link_graph(data["graph"])
        
        # Restore index structures
        for name, ds_dict in data.get("datasets", {}).items():
            self._datasets[name] = DatasetNode(**ds_dict)
        
        for name, col_dict in data.get("columns", {}).items():
            # Remove computed properties that shouldn't be passed to __init__
            col_dict_clean = {k: v for k, v in col_dict.items() if k != "full_name"}
            self._columns[name] = ColumnNode(**col_dict_clean)
        
        for term, bt_dict in data.get("business_terms", {}).items():
            self._business_terms[term] = BusinessTermNode(**bt_dict)
        
        self._column_embeddings = [
            (item[0], item[1]) for item in data.get("column_embeddings", [])
        ]
        
        self._term_to_columns = {
            k: set(v) for k, v in data.get("term_to_columns", {}).items()
        }
        
        logger.info(f"Loaded knowledge graph from {self.persist_path}")


def create_knowledge_graph_client(
    backend: str = "networkx",
    **kwargs,
) -> KnowledgeGraphClient:
    """
    Factory function to create a knowledge graph client.
    
    Args:
        backend: Backend type ("networkx" or "neo4j")
        **kwargs: Backend-specific arguments
        
    Returns:
        KnowledgeGraphClient instance
    """
    if backend == "networkx":
        return NetworkXKnowledgeGraph(**kwargs)
    elif backend == "neo4j":
        raise NotImplementedError("Neo4j backend not yet implemented")
    else:
        raise ValueError(f"Unknown backend: {backend}")

