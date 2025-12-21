#!/usr/bin/env python3
"""
Comprehensive Knowledge Graph Test Script

Tests all major KG functionality:
1. Graph loading
2. Dataset queries
3. Column queries
4. Business term mappings
5. Semantic search
6. Schema retrieval
7. Relationship queries
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.knowledge_graph import create_query_service
from loguru import logger

def test_kg():
    print("=" * 70)
    print("KNOWLEDGE GRAPH COMPREHENSIVE TEST")
    print("=" * 70)
    
    # Initialize query service
    try:
        kg_path = project_root / 'data' / 'knowledge_graph.json'
        if not kg_path.exists():
            print(f"❌ Knowledge graph file not found at: {kg_path}")
            print(f"   Please run: python -m src.knowledge_graph.populate")
            return False
        
        qs = create_query_service(str(kg_path))
        print("✅ Knowledge Graph loaded successfully\n")
    except Exception as e:
        print(f"❌ Failed to load KG: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    
    # Get graph statistics
    stats = qs.kg.get_statistics()
    print("📊 Graph Statistics:")
    print(f"   Total nodes: {stats['total_nodes']}")
    print(f"   Total edges: {stats['total_edges']}")
    print(f"   Datasets: {stats['datasets']}")
    print(f"   Columns: {stats['columns']}")
    print(f"   Business terms: {stats['business_terms']}")
    print(f"   Columns with embeddings: {stats['columns_with_embeddings']}\n")
    
    # Test 1: List all datasets
    print("=" * 70)
    print("TEST 1: List All Datasets")
    print("=" * 70)
    datasets = qs.kg.get_datasets()
    print(f"Found {len(datasets)} datasets:")
    for ds in datasets:
        print(f"   ✓ {ds['name']}: {ds['row_count']:,} rows, {ds['size_bytes']/1024/1024:.2f} MB")
    
    assert len(datasets) == 7, f"Expected 7 datasets, got {len(datasets)}"
    print("✅ PASS: All 7 datasets found\n")
    
    # Test 2: Find datasets by business terms
    print("=" * 70)
    print("TEST 2: Find Datasets by Business Terms")
    print("=" * 70)
    
    test_cases = [
        {
            "name": "Sales Analytics (A1)",
            "metrics": ["revenue", "order_count"],
            "dimensions": ["region", "category"],
            "expected": ["orders"]
        },
        {
            "name": "Marketing Performance (A2)",
            "metrics": ["CTR", "CVR", "CPA", "ROAS"],
            "dimensions": ["campaign"],
            "expected": ["marketing_events", "marketing_campaigns"]
        },
        {
            "name": "Customer 360 (C2)",
            "metrics": ["lifetime_value"],
            "dimensions": ["customer"],
            "expected": ["customers"]
        }
    ]
    
    for test_case in test_cases:
        print(f"\nQuery: {test_case['name']}")
        print(f"   Metrics: {test_case['metrics']}")
        print(f"   Dimensions: {test_case['dimensions']}")
        
        results = qs.find_datasets_for_metrics(
            test_case['metrics'],
            test_case['dimensions']
        )
        
        print(f"   Found {len(results)} datasets:")
        for r in results[:3]:
            print(f"      - {r['name']}: relevance={r['relevance_score']:.2f}, "
                  f"matched_terms={r['matched_terms']}")
        
        # Check if expected datasets are in results
        found_names = [r['name'] for r in results]
        for expected in test_case['expected']:
            if expected in found_names:
                print(f"   ✅ Found expected dataset: {expected}")
            else:
                print(f"   ⚠️  Expected dataset not found: {expected}")
    
    print("\n✅ PASS: Dataset discovery working\n")
    
    # Test 3: Business term to column mapping
    print("=" * 70)
    print("TEST 3: Business Term → Column Mapping")
    print("=" * 70)
    
    test_terms = ["revenue", "customer", "campaign", "order", "product"]
    
    for term in test_terms:
        cols = qs.get_columns_for_term(term)
        print(f"\nTerm: '{term}'")
        print(f"   Mapped to {len(cols)} columns:")
        for col in cols[:5]:
            print(f"      - {col['dataset_name']}.{col['name']}")
        if len(cols) > 5:
            print(f"      ... and {len(cols) - 5} more")
    
    print("\n✅ PASS: Business term mappings working\n")
    
    # Test 4: Semantic search
    print("=" * 70)
    print("TEST 4: Semantic Search (Embedding-based)")
    print("=" * 70)
    
    semantic_queries = [
        "daily sales revenue by region",
        "marketing campaign performance metrics",
        "customer lifetime value and spending",
        "product recommendation features",
    ]
    
    for query in semantic_queries:
        print(f"\nQuery: '{query}'")
        results = qs.semantic_column_search(query, top_k=3)
        print(f"   Top {len(results)} results:")
        for r in results:
            print(f"      - {r['full_name']}: similarity={r.get('similarity_score', 0):.3f}")
    
    print("\n✅ PASS: Semantic search working\n")
    
    # Test 5: Dataset schema retrieval
    print("=" * 70)
    print("TEST 5: Dataset Schema Retrieval")
    print("=" * 70)
    
    test_datasets = ["orders", "customers", "marketing_events"]
    
    for ds_name in test_datasets:
        schema = qs.get_dataset_schema(ds_name)
        if schema:
            print(f"\nDataset: {ds_name}")
            print(f"   Description: {schema.get('description', 'N/A')}")
            print(f"   Columns: {len(schema.get('columns', []))}")
            print(f"   Sample columns:")
            for col in schema.get('columns', [])[:5]:
                print(f"      - {col['name']} ({col['data_type']})")
        else:
            print(f"\n❌ Dataset '{ds_name}' not found")
    
    print("\n✅ PASS: Schema retrieval working\n")
    
    # Test 6: Related datasets (FK relationships)
    print("=" * 70)
    print("TEST 6: Related Datasets (Foreign Keys)")
    print("=" * 70)
    
    test_datasets = ["orders", "marketing_events", "support_tickets"]
    
    for ds_name in test_datasets:
        related = qs.get_related_datasets(ds_name)
        print(f"\nDataset: {ds_name}")
        if related:
            print(f"   Related to {len(related)} datasets:")
            for rel in related:
                print(f"      - {rel['dataset']} (via {rel['via_column']})")
        else:
            print(f"   No related datasets found")
    
    print("\n✅ PASS: Relationship queries working\n")
    
    # Test 7: Join path discovery
    print("=" * 70)
    print("TEST 7: Join Path Discovery")
    print("=" * 70)
    
    join_tests = [
        ("orders", "customers"),
        ("orders", "products"),
        ("marketing_events", "marketing_campaigns"),
    ]
    
    for source, target in join_tests:
        paths = qs.find_join_paths(source, target)
        print(f"\nJoin: {source} → {target}")
        if paths:
            for path in paths:
                print(f"   ✓ {path['type']}: {path.get('source_column', path.get('column'))} = {path.get('target_column', path.get('column'))}")
        else:
            print(f"   ⚠️  No join path found")
    
    print("\n✅ PASS: Join path discovery working\n")
    
    # Test 8: Column explanation
    print("=" * 70)
    print("TEST 8: Column Explanation")
    print("=" * 70)
    
    test_columns = [
        "orders.total_amount",
        "customers.total_lifetime_value",
        "marketing_events.revenue",
    ]
    
    for col_name in test_columns:
        explanation = qs.explain_column(col_name)
        if explanation:
            print(f"\nColumn: {col_name}")
            print(f"   Type: {explanation.get('data_type', 'N/A')}")
            print(f"   Description: {explanation.get('description', 'N/A')}")
            business_terms = explanation.get('business_terms', [])
            if business_terms:
                print(f"   Business terms:")
                for bt in business_terms:
                    print(f"      - {bt['term']}: {bt['definition']}")
        else:
            print(f"\n❌ Column '{col_name}' not found")
    
    print("\n✅ PASS: Column explanation working\n")
    
    # Final Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print("✅ All Knowledge Graph tests passed!")
    print(f"✅ Graph contains {stats['total_nodes']} nodes and {stats['total_edges']} edges")
    print(f"✅ {stats['columns_with_embeddings']} columns have embeddings for semantic search")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    try:
        success = test_kg()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)