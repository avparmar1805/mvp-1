"""Knowledge Graph module for the Agentic Data Product Builder."""

from src.knowledge_graph.schema import (
    NodeType,
    RelationshipType,
    DatasetNode,
    ColumnNode,
    BusinessTermNode,
    DomainNode,
    QualityRuleNode,
    BUSINESS_TERMS_CATALOG,
    COLUMN_TERM_MAPPINGS,
)

from src.knowledge_graph.client import (
    KnowledgeGraphClient,
    NetworkXKnowledgeGraph,
    create_knowledge_graph_client,
)

from src.knowledge_graph.embeddings import (
    EmbeddingGenerator,
    cosine_similarity,
    find_similar_embeddings,
)

from src.knowledge_graph.queries import (
    KnowledgeGraphQueryService,
    create_query_service,
)

__all__ = [
    # Schema
    "NodeType",
    "RelationshipType",
    "DatasetNode",
    "ColumnNode",
    "BusinessTermNode",
    "DomainNode",
    "QualityRuleNode",
    "BUSINESS_TERMS_CATALOG",
    "COLUMN_TERM_MAPPINGS",
    # Client
    "KnowledgeGraphClient",
    "NetworkXKnowledgeGraph",
    "create_knowledge_graph_client",
    # Embeddings
    "EmbeddingGenerator",
    "cosine_similarity",
    "find_similar_embeddings",
    # Queries
    "KnowledgeGraphQueryService",
    "create_query_service",
]

