"""
Knowledge Graph Learning Engine

Processes usage data to improve the Knowledge Graph:
- Updates co-occurrence relationships
- Adjusts relevance scores
- Extracts and maps business terms
"""

import re
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import Counter
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BusinessTermExtractor:
    """
    Extracts business terms from user queries
    """
    
    def __init__(self):
        """Initialize term extractor"""
        # Common business/data terms to look for
        self.business_keywords = {
            'revenue', 'sales', 'profit', 'cost', 'margin', 'price',
            'customer', 'user', 'order', 'product', 'category',
            'region', 'country', 'city', 'location',
            'daily', 'weekly', 'monthly', 'quarterly', 'yearly',
            'total', 'average', 'count', 'sum', 'max', 'min',
            'conversion', 'retention', 'churn', 'engagement',
            'campaign', 'marketing', 'advertising', 'promotion',
            'inventory', 'stock', 'warehouse', 'shipment',
            'transaction', 'payment', 'refund', 'discount'
        }
        
        # Stop words to filter out
        self.stop_words = {
            'show', 'me', 'get', 'find', 'create', 'build', 'make',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was'
        }
    
    def extract_terms(self, query: str) -> List[str]:
        """
        Extract business terms from query
        
        Args:
            query: User query text
            
        Returns:
            List of extracted terms
        """
        # Lowercase and tokenize
        query_lower = query.lower()
        
        # Extract words (alphanumeric)
        words = re.findall(r'\b[a-z]+\b', query_lower)
        
        # Filter terms
        terms = []
        for word in words:
            if word in self.business_keywords and word not in self.stop_words:
                terms.append(word)
        
        # Extract multi-word phrases (simple approach)
        # Look for common patterns like "X by Y"
        phrases = re.findall(r'(\w+)\s+by\s+(\w+)', query_lower)
        for metric, dimension in phrases:
            if metric in self.business_keywords:
                terms.append(metric)
            if dimension in self.business_keywords:
                terms.append(dimension)
        
        return list(set(terms))  # Remove duplicates
    
    def identify_new_terms(
        self,
        terms: List[str],
        existing_terms: Set[str]
    ) -> List[str]:
        """
        Identify terms not in existing glossary
        
        Args:
            terms: Extracted terms
            existing_terms: Set of existing glossary terms
            
        Returns:
            List of new terms
        """
        return [term for term in terms if term not in existing_terms]


class LearningEngine:
    """
    Main learning engine that updates the Knowledge Graph
    """
    
    def __init__(self, usage_analytics, kg_client):
        """
        Initialize learning engine
        
        Args:
            usage_analytics: UsageAnalyticsService instance
            kg_client: Knowledge Graph client
        """
        self.usage_analytics = usage_analytics
        self.kg_client = kg_client
        self.term_extractor = BusinessTermExtractor()
        logger.info("Learning Engine initialized")
    
    def process_usage_event(
        self,
        query: str,
        selected_datasets: List[str],
        data_product_id: Optional[str] = None
    ):
        """
        Process a usage event and update KG
        
        Args:
            query: User query
            selected_datasets: Datasets selected for data product
            data_product_id: Optional data product ID
        """
        # Extract terms from query
        terms = self.term_extractor.extract_terms(query)
        
        # Record usage event
        self.usage_analytics.record_dataset_selection(
            datasets=selected_datasets,
            query=query,
            data_product_id=data_product_id
        )
        
        # Process extracted terms
        if terms:
            self._process_extracted_terms(terms, selected_datasets)
        
        logger.info(f"Processed usage event: {len(selected_datasets)} datasets, {len(terms)} terms")
    
    def _process_extracted_terms(self, terms: List[str], datasets: List[str]):
        """
        Process extracted terms and update KG
        
        Args:
            terms: Extracted business terms
            datasets: Associated datasets
        """
        # Get existing terms from KG
        existing_terms = self._get_existing_terms()
        
        # Identify new terms
        new_terms = self.term_extractor.identify_new_terms(terms, existing_terms)
        
        # Update term-dataset mappings
        for term in terms:
            self._update_term_dataset_mapping(term, datasets)
        
        if new_terms:
            logger.info(f"Discovered {len(new_terms)} new terms: {new_terms}")
    
    def _get_existing_terms(self) -> Set[str]:
        """Get existing business terms from KG"""
        # Query KG for business term nodes
        try:
            terms = set()
            for node_id, node_data in self.kg_client.graph.nodes(data=True):
                if node_data.get('type') == 'business_term':
                    terms.add(node_data.get('name', '').lower())
            return terms
        except Exception as e:
            logger.warning(f"Error getting existing terms: {e}")
            return set()
    
    def _update_term_dataset_mapping(self, term: str, datasets: List[str]):
        """
        Update mapping between term and datasets
        
        Args:
            term: Business term
            datasets: Associated datasets
        """
        # This would update the KG with term-dataset relationships
        # For now, we'll track in analytics DB
        pass
    
    def get_enhanced_datasets(
        self,
        initial_datasets: List[str],
        top_k: int = 3
    ) -> List[str]:
        """
        Enhance dataset list with co-occurrence recommendations
        
        Args:
            initial_datasets: Initially discovered datasets
            top_k: Number of related datasets to add per dataset
            
        Returns:
            Enhanced list of datasets
        """
        enhanced = set(initial_datasets)
        
        for dataset in initial_datasets:
            # Get related datasets
            related = self.usage_analytics.get_related_datasets(dataset, top_k)
            
            # Add high-scoring related datasets
            for related_ds, score in related:
                if score > 0.3:  # Threshold
                    enhanced.add(related_ds)
        
        return list(enhanced)
    
    def rank_datasets_by_relevance(
        self,
        datasets: List[str],
        query: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Rank datasets by relevance scores
        
        Args:
            datasets: List of datasets to rank
            query: Optional query context
            
        Returns:
            List of (dataset, score) tuples, sorted by score
        """
        scored = []
        
        for dataset in datasets:
            # Get base relevance score
            score = self.usage_analytics.get_relevance_score(dataset)
            
            # Boost score if query contains dataset-related terms
            if query:
                # Simple boost based on name matching
                if dataset.lower() in query.lower():
                    score *= 1.2
            
            scored.append((dataset, score))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return scored
    
    def get_suggested_terms(self, min_frequency: int = 5) -> List[Dict[str, Any]]:
        """
        Get suggested new business terms to add to glossary
        
        Args:
            min_frequency: Minimum usage frequency to suggest
            
        Returns:
            List of term suggestions with metadata
        """
        # This would query the analytics DB for frequently used new terms
        # For now, return empty list
        return []
    
    def update_kg_from_usage(self):
        """
        Batch update KG based on accumulated usage data
        
        This should be run periodically (e.g., daily)
        """
        logger.info("Starting KG update from usage data...")
        
        # Get usage stats
        stats = self.usage_analytics.get_usage_stats(days=30)
        
        # Update dataset metadata in KG
        for dataset_info in stats['top_datasets']:
            self._update_dataset_metadata(
                dataset_info['name'],
                usage_count=dataset_info['count']
            )
        
        # Update co-occurrence edges in KG
        for cooccur in stats['top_cooccurrences']:
            self._update_cooccurrence_edge(
                cooccur['datasets'][0],
                cooccur['datasets'][1],
                cooccur['count']
            )
        
        logger.info("KG update complete")
    
    def _update_dataset_metadata(self, dataset: str, usage_count: int):
        """Update dataset node metadata in KG"""
        try:
            # Find dataset node
            for node_id, node_data in self.kg_client.graph.nodes(data=True):
                if node_data.get('name') == dataset and node_data.get('type') == 'dataset':
                    # Update metadata
                    self.kg_client.graph.nodes[node_id]['usage_count'] = usage_count
                    self.kg_client.graph.nodes[node_id]['last_updated'] = datetime.now().isoformat()
                    
                    # Get and update relevance score
                    relevance = self.usage_analytics.get_relevance_score(dataset)
                    self.kg_client.graph.nodes[node_id]['relevance_score'] = relevance
                    
                    logger.debug(f"Updated metadata for dataset: {dataset}")
                    break
        except Exception as e:
            logger.warning(f"Error updating dataset metadata: {e}")
    
    def _update_cooccurrence_edge(self, dataset_a: str, dataset_b: str, count: int):
        """Add or update co-occurrence edge in KG"""
        try:
            # Find dataset nodes
            node_a = None
            node_b = None
            
            for node_id, node_data in self.kg_client.graph.nodes(data=True):
                if node_data.get('name') == dataset_a and node_data.get('type') == 'dataset':
                    node_a = node_id
                if node_data.get('name') == dataset_b and node_data.get('type') == 'dataset':
                    node_b = node_id
            
            if node_a and node_b:
                # Calculate score
                score = min(1.0, count / 10.0)
                
                # Add or update edge
                if self.kg_client.graph.has_edge(node_a, node_b):
                    # Update existing edge
                    self.kg_client.graph[node_a][node_b]['cooccurrence_count'] = count
                    self.kg_client.graph[node_a][node_b]['cooccurrence_score'] = score
                else:
                    # Add new edge
                    self.kg_client.graph.add_edge(
                        node_a,
                        node_b,
                        relationship='FREQUENTLY_USED_WITH',
                        cooccurrence_count=count,
                        cooccurrence_score=score,
                        created_at=datetime.now().isoformat()
                    )
                
                logger.debug(f"Updated co-occurrence edge: {dataset_a} <-> {dataset_b}")
        except Exception as e:
            logger.warning(f"Error updating co-occurrence edge: {e}")
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the learning process
        
        Returns:
            Statistics dictionary
        """
        usage_stats = self.usage_analytics.get_usage_stats(days=30)
        
        return {
            "usage_stats": usage_stats,
            "total_cooccurrences": len(usage_stats['top_cooccurrences']),
            "datasets_with_scores": len(usage_stats['top_datasets']),
            "learning_enabled": True
        }
