import os
import yaml
import glob
from typing import List, Dict, Any, Optional
from loguru import logger
from src.knowledge_graph.embeddings import EmbeddingGenerator, find_similar_embeddings

class DataProductCatalog:
    """
    Catalog service for indexing and searching Data Products.
    """
    def __init__(self, specs_dir: str = "output/specs"):
        self.specs_dir = specs_dir
        self.embedding_generator = EmbeddingGenerator()
        
        # In-memory index
        self.products: Dict[str, Dict] = {}
        self.embeddings: List[tuple] = [] # (product_name, embedding)
        self._is_indexed = False
        
    def index(self, force_refresh: bool = False):
        """
        Load all YAML specs and generate embeddings.
        """
        if self._is_indexed and not force_refresh:
            return
            
        logger.info(f"Indexing Data Products from {self.specs_dir}...")
        
        yaml_files = glob.glob(os.path.join(self.specs_dir, "*.yaml"))
        
        for file_path in yaml_files:
            try:
                with open(file_path, 'r') as f:
                    spec = yaml.safe_load(f)
                    
                meta = spec.get("metadata", {})
                name = meta.get("name")
                
                if not name:
                    continue
                    
                self.products[name] = spec
                
                # Create rich text for embedding
                # Combine Name + Description + Business Intent + Metrics
                desc = meta.get("description", "")
                intent = spec.get("business_context", {}).get("request", "")
                metrics = spec.get("business_context", {}).get("intent", {}).get("metrics", [])
                
                text_to_embed = f"{name} | {desc} | {intent} | Metrics: {', '.join(metrics)}"
                
                # Generate embedding
                embedding = self.embedding_generator.generate_embedding(text_to_embed)
                if embedding:
                    self.embeddings.append((name, embedding))
                    
            except Exception as e:
                logger.error(f"Failed to index {file_path}: {e}")
                
        self._is_indexed = True
        logger.info(f"Indexed {len(self.products)} products.")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Semantic search for data products.
        """
        if not self._is_indexed:
            self.index()
            
        # 1. Generate query embedding
        query_vec = self.embedding_generator.generate_embedding(query)
        if not query_vec:
            logger.warning("Could not generate query embedding. Falling back to keyword search.")
            return self._keyword_search(query)
            
        # 2. Find similar products
        results = find_similar_embeddings(query_vec, self.embeddings, top_k=top_k)
        
        # 3. Format output
        output = []
        for name, score in results:
            product = self.products.get(name)
            if product:
                # Add score to metadata for display
                product_copy = product.copy()
                product_copy["relevance_score"] = score
                output.append(product_copy)
                
        return output

    def _keyword_search(self, query: str) -> List[Dict[str, Any]]:
        """Fallback simple keyword search."""
        query = query.lower()
        results = []
        for name, spec in self.products.items():
            text = str(spec).lower()
            if query in text:
                results.append(spec)
        return results

    def list_all(self) -> List[Dict[str, Any]]:
        """Return all indexed products."""
        if not self._is_indexed:
            self.index()
        return list(self.products.values())
