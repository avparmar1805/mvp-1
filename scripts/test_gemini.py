import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.utils.llm_client import LLMClient
from src.knowledge_graph.embeddings import EmbeddingGenerator

def test_gemini_connection():
    print("--- Testing Gemini Connectivity ---")
    
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        print("❌ Error: GEMINI_API_KEY not found in environment variables.")
        print("Please add GEMINI_API_KEY=... to your .env file.")
        return

    print(f"✅ Found GEMINI_API_KEY (starts with {key[:4]}...)")

    # Test LLM Generation
    print("\n1. Testing Text Generation...")
    try:
        client = LLMClient()
        if client.provider != "gemini":
            print(f"⚠️  Client initialized with provider: {client.provider}")
            print("   (Did you set GEMINI_API_KEY correctly?)")
        else:
            response = client.generate_text("Hello, say 'Gemini is working'!")
            print(f"   Response: {response}")
            if "Gemini is working" in response:
                print("   ✅ Text Generation Success")
            else:
                print("   ⚠️  Unexpected response content")
    except Exception as e:
        print(f"   ❌ Text Generation Failed: {e}")

    # Test Embeddings
    print("\n2. Testing Embeddings...")
    try:
        embedder = EmbeddingGenerator()
        if embedder.provider != "gemini":
             print(f"⚠️  Embedder initialized with provider: {embedder.provider}")
        else:
            vector = embedder.generate_embedding("Test embedding")
            if vector and len(vector) > 0:
                print(f"   Received vector of length: {len(vector)}")
                print("   ✅ Embeddings Success")
            else:
                print("   ❌ Failed to generate embedding vector")
    except Exception as e:
        print(f"   ❌ Embeddings Failed: {e}")

if __name__ == "__main__":
    test_gemini_connection()
