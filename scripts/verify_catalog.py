import sys
import os
from loguru import logger

# Add project root to path
sys.path.append(os.getcwd())

from src.services.catalog import DataProductCatalog

def main():
    logger.info("Initializing Catalog...")
    catalog = DataProductCatalog()
    
    # 1. Index Products
    catalog.index()
    
    products = catalog.list_all()
    logger.info(f"Loaded {len(products)} products.")
    for p in products:
        logger.info(f" - {p['metadata']['name']}")
        
    # 2. Test Semantic Search
    queries = [
        "money reports",       # Should match "Revenue" / "Sales"
        "customer help",       # Should match "Support"
        "shopping cart data",  # Should match "Product Recommendations" / "Business Metrics"
    ]
    
    for q in queries:
        logger.info(f"\n🔍 Searching for: '{q}'")
        results = catalog.search(q, top_k=2)
        
        for res in results:
            name = res['metadata']['name']
            score = res.get('relevance_score', 0)
            logger.info(f"   found: {name} (Score: {score:.4f})")
            
if __name__ == "__main__":
    main()
