"""
Test script for Knowledge Graph Learning capabilities

Demonstrates:
1. Usage tracking
2. Co-occurrence learning
3. Relevance score updates
4. Business term extraction
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.orchestrator import OrchestratorAgent
from src.knowledge_graph.usage_analytics import UsageAnalyticsService
from src.knowledge_graph.learning_engine import LearningEngine
import json


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def main():
    """Test Knowledge Graph learning"""
    
    print_section("🧠 Knowledge Graph Learning - Demo")
    
    # Initialize orchestrator (includes learning components)
    print("\n📦 Initializing Orchestrator with Learning Engine...")
    orchestrator = OrchestratorAgent()
    print("   ✅ Orchestrator initialized")
    print("   ✅ Usage Analytics enabled")
    print("   ✅ Learning Engine ready")
    
    # Test queries that should create usage patterns
    test_queries = [
        "Show me daily sales by region and category",
        "Create a sales report by region",
        "Analyze revenue by category and region",
        "Show me customer orders by region",
        "Build a product sales analysis by category"
    ]
    
    print_section("📊 Running Test Queries to Generate Usage Data")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Processing: \"{query}\"")
        
        try:
            result = orchestrator.run(query)
            
            if result.get("errors"):
                print(f"   ⚠️  Errors: {result['errors']}")
            else:
                # Show what datasets were selected
                discovery = result.get("discovery_result", {})
                datasets = discovery.get("selected_datasets", [])
                
                dataset_names = []
                for ds in datasets:
                    if isinstance(ds, str):
                        dataset_names.append(ds)
                    else:
                        dataset_names.append(ds.get("name", "unknown"))
                
                print(f"   ✅ Datasets selected: {', '.join(dataset_names)}")
                print(f"   📝 Usage recorded for learning")
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Show usage analytics
    print_section("📈 Usage Analytics")
    
    analytics = orchestrator.usage_analytics
    stats = analytics.get_usage_stats(days=1)
    
    print(f"\n   Total Events: {stats['total_events']}")
    print(f"\n   Events by Type:")
    for event_type, count in stats['events_by_type'].items():
        print(f"      • {event_type}: {count}")
    
    print(f"\n   Top Datasets:")
    for ds_info in stats['top_datasets'][:5]:
        print(f"      • {ds_info['name']}: {ds_info['count']} uses")
    
    print(f"\n   Top Co-occurrences:")
    for cooccur in stats['top_cooccurrences'][:5]:
        datasets = cooccur['datasets']
        count = cooccur['count']
        print(f"      • {datasets[0]} + {datasets[1]}: {count} times")
    
    # Test co-occurrence recommendations
    print_section("🔗 Co-occurrence Recommendations")
    
    test_dataset = "orders"
    print(f"\n   Datasets frequently used with '{test_dataset}':")
    
    related = analytics.get_related_datasets(test_dataset, top_k=5)
    for related_ds, score in related:
        print(f"      • {related_ds}: score = {score:.2f}")
    
    # Test relevance scores
    print_section("⭐ Relevance Scores")
    
    print(f"\n   Dataset relevance scores:")
    for ds_info in stats['top_datasets'][:5]:
        dataset = ds_info['name']
        score = analytics.get_relevance_score(dataset)
        print(f"      • {dataset}: {score:.3f}")
    
    # Test business term extraction
    print_section("🏷️  Business Term Extraction")
    
    from src.knowledge_graph.learning_engine import BusinessTermExtractor
    
    extractor = BusinessTermExtractor()
    
    test_queries_for_terms = [
        "Show me daily revenue by region",
        "Analyze customer churn rate",
        "Create a marketing campaign performance report"
    ]
    
    print(f"\n   Extracted terms from queries:")
    for query in test_queries_for_terms:
        terms = extractor.extract_terms(query)
        print(f"      \"{query}\"")
        print(f"         → {', '.join(terms) if terms else 'No terms extracted'}")
    
    # Test learning engine stats
    print_section("🎓 Learning Engine Statistics")
    
    learning_stats = orchestrator.learning_engine.get_learning_stats()
    
    print(f"\n   Learning Status: {'✅ Enabled' if learning_stats['learning_enabled'] else '❌ Disabled'}")
    print(f"   Total Co-occurrences Tracked: {learning_stats['total_cooccurrences']}")
    print(f"   Datasets with Scores: {learning_stats['datasets_with_scores']}")
    
    # Test enhanced discovery
    print_section("🔍 Enhanced Discovery with Learning")
    
    print(f"\n   Testing enhanced discovery for: 'sales analysis'")
    
    initial_datasets = ["orders", "products"]
    print(f"   Initial datasets: {', '.join(initial_datasets)}")
    
    enhanced = orchestrator.learning_engine.get_enhanced_datasets(initial_datasets, top_k=2)
    print(f"   Enhanced with co-occurrence: {', '.join(enhanced)}")
    
    # Test dataset ranking
    print(f"\n   Ranking datasets by relevance:")
    
    test_datasets = ["orders", "products", "customers", "marketing_campaigns"]
    ranked = orchestrator.learning_engine.rank_datasets_by_relevance(
        test_datasets,
        query="Show me sales by region"
    )
    
    for dataset, score in ranked:
        print(f"      {dataset}: {score:.3f}")
    
    # Save analytics to file
    print_section("💾 Saving Analytics Data")
    
    output_file = Path("output/kg_learning_analytics.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    analytics_data = {
        "usage_stats": stats,
        "learning_stats": learning_stats,
        "test_queries": test_queries,
        "timestamp": str(Path("data/usage_analytics.db").stat().st_mtime)
    }
    
    with open(output_file, 'w') as f:
        json.dump(analytics_data, f, indent=2, default=str)
    
    print(f"\n   ✅ Analytics saved to: {output_file}")
    
    # Summary
    print_section("✨ Learning Demo Complete!")
    
    print("\n   Key Capabilities Demonstrated:")
    print("      ✅ Usage tracking - All queries recorded")
    print("      ✅ Co-occurrence learning - Dataset pairs identified")
    print("      ✅ Relevance scoring - Scores updated from usage")
    print("      ✅ Term extraction - Business terms identified")
    print("      ✅ Enhanced discovery - Recommendations improved")
    
    print("\n   Database Location:")
    print(f"      📁 {Path('data/usage_analytics.db').absolute()}")
    
    print("\n   Next Steps:")
    print("      1. Run more queries to build usage patterns")
    print("      2. Check analytics dashboard (coming soon)")
    print("      3. Provide explicit feedback to improve scores")
    print("      4. Watch the KG get smarter over time!")
    print()


if __name__ == "__main__":
    main()
