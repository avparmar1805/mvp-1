#!/usr/bin/env python3
"""
Knowledge Graph Population Script

Reads bronze layer metadata and populates the knowledge graph with:
- Dataset nodes
- Column nodes with embeddings
- Business term nodes
- Column-to-term mappings

Usage:
    python -m src.knowledge_graph.populate --data-dir data/bronze
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from src.knowledge_graph.schema import (
    NodeType,
    RelationshipType,
    DatasetNode,
    ColumnNode,
    BusinessTermNode,
    DomainNode,
    BUSINESS_TERMS_CATALOG,
    COLUMN_TERM_MAPPINGS,
)
from src.knowledge_graph.client import (
    NetworkXKnowledgeGraph,
    create_knowledge_graph_client,
)
from src.knowledge_graph.embeddings import EmbeddingGenerator


def load_dataset_metadata(metadata_path: Path) -> Dict[str, Any]:
    """Load metadata from JSON file."""
    with open(metadata_path, "r") as f:
        return json.load(f)


def discover_bronze_datasets(data_dir: Path) -> List[Dict[str, Any]]:
    """
    Discover all bronze datasets by scanning the directory.
    
    Args:
        data_dir: Path to the bronze data directory
        
    Returns:
        List of dataset metadata dictionaries
    """
    datasets = []
    
    # Expected subdirectories
    expected_dirs = [
        "customers",
        "products", 
        "orders",
        "marketing_campaigns",
        "marketing_events",
        "user_interactions",
        "support_tickets",
    ]
    
    for subdir in expected_dirs:
        subdir_path = data_dir / subdir
        
        if not subdir_path.exists():
            logger.warning(f"Expected directory not found: {subdir_path}")
            continue
        
        # Look for metadata JSON files
        json_files = list(subdir_path.glob("*.json"))
        
        for json_file in json_files:
            try:
                metadata = load_dataset_metadata(json_file)
                
                # Find corresponding parquet file
                parquet_file = json_file.with_suffix(".parquet")
                if parquet_file.exists():
                    metadata["path"] = str(parquet_file)
                
                datasets.append(metadata)
                logger.debug(f"Discovered dataset: {metadata.get('dataset_name', subdir)}")
                
            except Exception as e:
                logger.error(f"Failed to load metadata from {json_file}: {e}")
    
    return datasets


def create_dataset_node(metadata: Dict[str, Any]) -> DatasetNode:
    """Create a DatasetNode from metadata dictionary."""
    name = metadata.get("dataset_name", "").replace("bronze.", "")
    
    return DatasetNode(
        name=name,
        description=metadata.get("description", ""),
        layer="bronze",
        format="parquet",
        path=metadata.get("path", ""),
        row_count=metadata.get("row_count", 0),
        size_bytes=metadata.get("size_bytes", 0),
        created_at=metadata.get("created_at", ""),
        owner="data_engineering",
        tags=_infer_tags(name),
        quality_score=_calculate_quality_score(metadata),
        freshness="daily",
    )


def create_column_nodes(
    dataset_name: str,
    schema: List[Dict[str, Any]],
    embedding_generator: Optional[EmbeddingGenerator] = None,
    generate_embeddings: bool = True,
) -> List[ColumnNode]:
    """
    Create ColumnNode objects from schema information.
    
    Args:
        dataset_name: Name of the parent dataset
        schema: List of column schema dictionaries
        embedding_generator: Optional embedding generator
        generate_embeddings: Whether to generate embeddings
        
    Returns:
        List of ColumnNode objects
    """
    columns = []
    
    for col_info in schema:
        col_name = col_info.get("name", "")
        
        # Determine if this is a primary or foreign key
        is_pk = col_name.endswith("_id") and col_name == f"{dataset_name.rstrip('s')}_id"
        is_fk = col_name.endswith("_id") and not is_pk
        
        # Infer references for foreign keys
        references = None
        if is_fk:
            ref_table = col_name.replace("_id", "") + "s"
            references = f"{ref_table}.{col_name}"
        
        # Extract sample values
        sample_values = []
        top_values = col_info.get("top_values", {})
        if top_values:
            sample_values = list(top_values.keys())[:5]
        
        # Extract statistics
        statistics = col_info.get("statistics", {})
        
        column = ColumnNode(
            name=col_name,
            dataset_name=dataset_name,
            data_type=col_info.get("type", "unknown"),
            description=_generate_column_description(col_name, dataset_name),
            nullable=col_info.get("nullable", True),
            is_primary_key=is_pk,
            is_foreign_key=is_fk,
            references=references,
            null_count=col_info.get("null_count", 0),
            unique_count=col_info.get("unique_count", 0),
            sample_values=sample_values,
            statistics=statistics,
        )
        
        # Generate embedding if available
        if generate_embeddings and embedding_generator and embedding_generator.is_available:
            embedding = embedding_generator.generate_column_embedding(
                column_name=col_name,
                dataset_name=dataset_name,
                data_type=column.data_type,
                description=column.description,
                sample_values=sample_values,
            )
            column.embedding = embedding
        
        columns.append(column)
    
    return columns


def _infer_tags(dataset_name: str) -> List[str]:
    """Infer tags from dataset name."""
    tags = ["bronze"]
    
    if "customer" in dataset_name:
        tags.extend(["customer", "master_data"])
    if "product" in dataset_name:
        tags.extend(["product", "master_data"])
    if "order" in dataset_name:
        tags.extend(["order", "transaction", "sales"])
    if "marketing" in dataset_name:
        tags.extend(["marketing"])
    if "campaign" in dataset_name:
        tags.append("campaign")
    if "event" in dataset_name:
        tags.extend(["event", "streaming"])
    if "interaction" in dataset_name:
        tags.extend(["interaction", "behavior", "recommendation"])
    if "ticket" in dataset_name or "support" in dataset_name:
        tags.extend(["support", "ticket"])
    
    return tags


def _calculate_quality_score(metadata: Dict[str, Any]) -> float:
    """Calculate quality score based on metadata."""
    score = 1.0
    
    # Check for quality issues
    quality_issues = metadata.get("quality_issues", [])
    if quality_issues:
        # Reduce score based on number of issues
        score -= len(quality_issues) * 0.05
    
    # Check for null percentages in schema
    schema = metadata.get("schema", [])
    for col in schema:
        null_count = col.get("null_count", 0)
        row_count = metadata.get("row_count", 1)
        if row_count > 0:
            null_pct = null_count / row_count
            if null_pct > 0.1:  # >10% nulls
                score -= 0.02
    
    return max(score, 0.5)  # Minimum score of 0.5


def _generate_column_description(col_name: str, dataset_name: str) -> str:
    """Generate a description for a column based on its name."""
    descriptions = {
        "customer_id": "Unique identifier for customer",
        "product_id": "Unique identifier for product",
        "order_id": "Unique identifier for order",
        "campaign_id": "Unique identifier for marketing campaign",
        "event_id": "Unique identifier for marketing event",
        "interaction_id": "Unique identifier for user interaction",
        "ticket_id": "Unique identifier for support ticket",
        "name": "Name or title",
        "email": "Email address",
        "phone": "Phone number",
        "total_amount": "Total monetary amount",
        "quantity": "Number of items",
        "price": "Price per unit",
        "category": "Category classification",
        "status": "Current status",
        "created_at": "Creation timestamp",
        "order_date": "Date of order",
        "signup_date": "Date of signup",
        "loyalty_tier": "Customer loyalty tier (bronze/silver/gold/platinum)",
        "total_lifetime_value": "Total customer lifetime value",
        "segment": "Customer segment classification",
        "region": "Geographic region",
        "rating_avg": "Average rating score",
        "review_count": "Number of reviews",
        "budget": "Allocated budget",
        "channel": "Marketing or support channel",
        "event_type": "Type of event (impression/click/conversion)",
        "cost": "Cost amount",
        "revenue": "Revenue amount",
        "interaction_type": "Type of user interaction",
        "rating": "User rating score",
        "session_id": "User session identifier",
        "duration_seconds": "Duration in seconds",
        "priority": "Priority level",
        "satisfaction_score": "Customer satisfaction score",
        "resolution_time_hours": "Time to resolve in hours",
    }
    
    return descriptions.get(col_name, f"Column {col_name} in {dataset_name}")


def populate_knowledge_graph(
    kg: NetworkXKnowledgeGraph,
    data_dir: str = "data/bronze",
    generate_embeddings: bool = True,
) -> Dict[str, Any]:
    """
    Populate the knowledge graph with bronze layer metadata.
    
    Args:
        kg: Knowledge graph client
        data_dir: Path to bronze data directory
        generate_embeddings: Whether to generate embeddings for columns
        
    Returns:
        Dictionary with population statistics
    """
    start_time = time.time()
    data_path = Path(data_dir)
    
    logger.info("=" * 60)
    logger.info("Knowledge Graph Population")
    logger.info("=" * 60)
    logger.info(f"Data directory: {data_path.absolute()}")
    logger.info(f"Generate embeddings: {generate_embeddings}")
    
    stats = {
        "datasets": 0,
        "columns": 0,
        "business_terms": 0,
        "mappings": 0,
        "embeddings_generated": 0,
    }
    
    # Initialize embedding generator
    embedding_generator = None
    if generate_embeddings:
        embedding_generator = EmbeddingGenerator()
        if not embedding_generator.is_available:
            logger.warning("Embeddings not available - continuing without embeddings")
            embedding_generator = None
    
    # =========================================================================
    # Step 1: Add Business Domains
    # =========================================================================
    logger.info("\n[1/4] Adding business domains...")
    
    domains = ["Sales", "Customer", "Product", "Marketing", "Support", "Geography"]
    for domain_name in domains:
        kg.add_domain(DomainNode(
            name=domain_name,
            description=f"{domain_name} domain",
        ))
    
    # =========================================================================
    # Step 2: Add Business Terms
    # =========================================================================
    logger.info("\n[2/4] Adding business terms...")
    
    for term in BUSINESS_TERMS_CATALOG:
        # Generate embedding for term if available
        if embedding_generator and embedding_generator.is_available:
            term.embedding = embedding_generator.generate_business_term_embedding(
                term=term.term,
                definition=term.definition,
                synonyms=term.synonyms,
                domain=term.domain,
            )
        
        kg.add_business_term(term)
        
        # Add domain relationship
        if term.domain:
            kg.add_relationship(
                NodeType.BUSINESS_TERM,
                term.term,
                NodeType.DOMAIN,
                term.domain,
                RelationshipType.BELONGS_TO_DOMAIN,
            )
        
        stats["business_terms"] += 1
    
    logger.info(f"  Added {stats['business_terms']} business terms")
    
    # =========================================================================
    # Step 3: Discover and Add Datasets
    # =========================================================================
    logger.info("\n[3/4] Adding datasets and columns...")
    
    datasets = discover_bronze_datasets(data_path)
    
    for ds_metadata in datasets:
        # Create dataset node
        dataset = create_dataset_node(ds_metadata)
        kg.add_dataset(dataset)
        stats["datasets"] += 1
        
        # Create column nodes
        schema = ds_metadata.get("schema", [])
        columns = create_column_nodes(
            dataset_name=dataset.name,
            schema=schema,
            embedding_generator=embedding_generator,
            generate_embeddings=generate_embeddings,
        )
        
        for column in columns:
            kg.add_column(column)
            stats["columns"] += 1
            
            if column.embedding:
                stats["embeddings_generated"] += 1
        
        logger.info(f"  Added dataset: {dataset.name} ({len(columns)} columns)")
    
    # =========================================================================
    # Step 4: Add Column-to-Term Mappings
    # =========================================================================
    logger.info("\n[4/4] Adding column-to-term mappings...")
    
    for col_full_name, terms in COLUMN_TERM_MAPPINGS.items():
        for term in terms:
            kg.map_column_to_term(col_full_name, term)
            stats["mappings"] += 1
    
    logger.info(f"  Added {stats['mappings']} column-term mappings")
    
    # =========================================================================
    # Summary
    # =========================================================================
    elapsed_time = time.time() - start_time
    
    logger.info("\n" + "=" * 60)
    logger.info("Population Summary")
    logger.info("=" * 60)
    logger.info(f"  Datasets: {stats['datasets']}")
    logger.info(f"  Columns: {stats['columns']}")
    logger.info(f"  Business terms: {stats['business_terms']}")
    logger.info(f"  Column-term mappings: {stats['mappings']}")
    logger.info(f"  Embeddings generated: {stats['embeddings_generated']}")
    logger.info(f"  Time: {elapsed_time:.2f} seconds")
    logger.info("=" * 60)
    
    # Get graph statistics
    graph_stats = kg.get_statistics()
    logger.info(f"\nGraph Statistics:")
    logger.info(f"  Total nodes: {graph_stats['total_nodes']}")
    logger.info(f"  Total edges: {graph_stats['total_edges']}")
    
    stats["elapsed_time"] = elapsed_time
    stats["graph_stats"] = graph_stats
    
    return stats


def main(
    data_dir: str = "data/bronze",
    output_path: str = "data/knowledge_graph.json",
    generate_embeddings: bool = True,
) -> Dict[str, Any]:
    """
    Main function to populate and save the knowledge graph.
    
    Args:
        data_dir: Path to bronze data directory
        output_path: Path to save the knowledge graph
        generate_embeddings: Whether to generate embeddings
        
    Returns:
        Population statistics
    """
    # Create knowledge graph
    kg = create_knowledge_graph_client(
        backend="networkx",
        persist_path=output_path,
    )
    
    # Populate
    stats = populate_knowledge_graph(
        kg=kg,
        data_dir=data_dir,
        generate_embeddings=generate_embeddings,
    )
    
    # Save
    kg.save(output_path)
    logger.info(f"\n✅ Knowledge graph saved to {output_path}")
    
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Populate knowledge graph from bronze layer metadata"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/bronze",
        help="Path to bronze data directory (default: data/bronze)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/knowledge_graph.json",
        help="Output path for knowledge graph (default: data/knowledge_graph.json)"
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip embedding generation"
    )
    
    args = parser.parse_args()
    
    try:
        stats = main(
            data_dir=args.data_dir,
            output_path=args.output,
            generate_embeddings=not args.no_embeddings,
        )
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"❌ Population failed: {e}")
        raise

