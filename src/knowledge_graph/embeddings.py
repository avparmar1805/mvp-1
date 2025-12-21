"""
Embedding generation for Knowledge Graph nodes.

Uses OpenAI's embedding API to generate vector representations
for columns and business terms for semantic search.
"""

import os
from typing import List, Optional

from loguru import logger

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not installed. Embeddings will not be available.")


class EmbeddingGenerator:
    """Generates embeddings using OpenAI's API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
    ):
        """
        Initialize the embedding generator.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Embedding model to use
        """
        self.model = model
        self.client = None
        self._initialized = False
        
        # Load .env file explicitly
        from pathlib import Path
        from dotenv import load_dotenv
        
        # Try to load from project root
        project_root = Path(__file__).parent.parent.parent
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI package not available")
            return
            
        if not self.api_key or self.api_key.startswith("your_"):
            logger.warning("OpenAI API key not configured. Embeddings disabled.")
            return
        
        try:
            self.client = OpenAI(api_key=self.api_key)
            self._initialized = True
            logger.info(f"Embedding generator initialized with model: {model}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
    
    @property
    def is_available(self) -> bool:
        """Check if embedding generation is available."""
        return self._initialized and self.client is not None
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding, or None if unavailable
        """
        if not self.is_available:
            return None
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 100,
    ) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process per API call
            
        Returns:
            List of embeddings (None for failed items)
        """
        if not self.is_available:
            return [None] * len(texts)
        
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                logger.debug(f"Generated embeddings for batch {i//batch_size + 1}")
            except Exception as e:
                logger.error(f"Failed to generate batch embeddings: {e}")
                embeddings.extend([None] * len(batch))
        
        return embeddings
    
    def generate_column_embedding(
        self,
        column_name: str,
        dataset_name: str,
        data_type: str,
        description: str = "",
        sample_values: List[str] = None,
    ) -> Optional[List[float]]:
        """
        Generate embedding for a column based on its metadata.
        
        Args:
            column_name: Name of the column
            dataset_name: Name of the dataset
            data_type: Data type of the column
            description: Optional description
            sample_values: Optional sample values
            
        Returns:
            Embedding vector or None
        """
        # Create rich text representation of the column
        parts = [
            f"Column: {column_name}",
            f"Dataset: {dataset_name}",
            f"Type: {data_type}",
        ]
        
        if description:
            parts.append(f"Description: {description}")
        
        if sample_values:
            samples = ", ".join(str(v) for v in sample_values[:5])
            parts.append(f"Sample values: {samples}")
        
        text = " | ".join(parts)
        return self.generate_embedding(text)
    
    def generate_business_term_embedding(
        self,
        term: str,
        definition: str = "",
        synonyms: List[str] = None,
        domain: str = "",
    ) -> Optional[List[float]]:
        """
        Generate embedding for a business term.
        
        Args:
            term: The business term
            definition: Definition of the term
            synonyms: List of synonyms
            domain: Business domain
            
        Returns:
            Embedding vector or None
        """
        parts = [f"Term: {term}"]
        
        if definition:
            parts.append(f"Definition: {definition}")
        
        if synonyms:
            parts.append(f"Synonyms: {', '.join(synonyms)}")
        
        if domain:
            parts.append(f"Domain: {domain}")
        
        text = " | ".join(parts)
        return self.generate_embedding(text)


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score (0-1)
    """
    if vec1 is None or vec2 is None:
        return 0.0
    
    import numpy as np
    
    a = np.array(vec1)
    b = np.array(vec2)
    
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


def find_similar_embeddings(
    query_embedding: List[float],
    embeddings: List[tuple],  # List of (id, embedding) tuples
    top_k: int = 5,
) -> List[tuple]:
    """
    Find most similar embeddings to a query.
    
    Args:
        query_embedding: Query vector
        embeddings: List of (id, embedding) tuples
        top_k: Number of results to return
        
    Returns:
        List of (id, similarity_score) tuples, sorted by similarity
    """
    if query_embedding is None:
        return []
    
    similarities = []
    
    for item_id, embedding in embeddings:
        if embedding is not None:
            sim = cosine_similarity(query_embedding, embedding)
            similarities.append((item_id, sim))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]

