"""
Embedding generation for Knowledge Graph nodes.

Uses OpenAI or Gemini API to generate vector representations
for columns and business terms for semantic search.
"""

import os
from typing import List, Optional

from loguru import logger

# Try importing OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Try importing Google Generative AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class EmbeddingGenerator:
    """Generates embeddings using OpenAI or Gemini API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = None, 
    ):
        """
        Initialize the embedding generator.
        
        Args:
            api_key: API key
            model: Embedding model to use
        """
        self.provider = "openai"
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
            
        # 1. Check for Gemini Key first
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key and GEMINI_AVAILABLE:
            self.provider = "gemini"
            genai.configure(api_key=gemini_key)
            self.model = model or "models/embedding-001"
            self._initialized = True
            logger.info(f"Embedding generator initialized with provider: Gemini, model: {self.model}")
            return
        
        # 2. Fallback to OpenAI
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
             logger.warning("No API key configured (OpenAI or Gemini). Embeddings disabled.")
             return

        if OPENAI_AVAILABLE:
            try:
                self.client = OpenAI(api_key=self.api_key)
                self.model = model or "text-embedding-3-small"
                self._initialized = True
                logger.info(f"Embedding generator initialized with provider: OpenAI, model: {self.model}")
            except Exception as e:
                 logger.error(f"Failed to initialize OpenAI client: {e}")
        else:
             logger.warning("OpenAI package not available.")

    
    @property
    def is_available(self) -> bool:
        """Check if embedding generation is available."""
        return self._initialized
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text.
        """
        if not self.is_available:
            return None
            
        if self.provider == "gemini":
            try:
                result = genai.embed_content(
                    model=self.model,
                    content=text,
                    task_type="retrieval_document",
                    title="Embedding"
                )
                return result['embedding']
            except Exception as e:
                logger.error(f"Failed to generate Gemini embedding: {e}")
                return None
        
        # OpenAI Fallback
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate OpenAI embedding: {e}")
            return None
    
    def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 100,
    ) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts.
        """
        if not self.is_available:
            return [None] * len(texts)
        
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            if self.provider == "gemini":
                try:
                    # Gemini might not support batching in the same way, looping for safety or checking docs
                    # embed_content supports a list, but let's be safe with smaller batches or loops if needed.
                    # Actually genai.embed_content content argument can be a list.
                    # Note: "models/embedding-001" supports batching.
                    result = genai.embed_content(
                        model=self.model,
                        content=batch,
                        task_type="retrieval_document"
                    )
                    # Result is a dict with 'embedding' key which is a list of embeddings if input is list
                    batch_embeddings = result['embedding']
                    embeddings.extend(batch_embeddings)
                except Exception as e:
                    logger.error(f"Failed to generate Gemini batch embeddings: {e}")
                    embeddings.extend([None] * len(batch))
            
            else: # OpenAI
                try:
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=batch,
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    embeddings.extend(batch_embeddings)
                except Exception as e:
                    logger.error(f"Failed to generate OpenAI batch embeddings: {e}")
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

