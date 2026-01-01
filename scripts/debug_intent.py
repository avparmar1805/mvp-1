import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.intent_agent import IntentAgent
from src.utils.llm_client import LLMClient

def debug_intent():
    client = LLMClient()
    agent = IntentAgent(client)
    
    queries = [
        "Cluster orders into 3 groups based on total_amount and order_date",
        "Predict daily revenue for the next 7 days",
        "Segment customers into VIP, Loyal, and At-Risk groups",
        "Show me total sales by month"
    ]
    
    for q in queries:
        print(f"\nQuery: {q}")
        res = agent.analyze(q)
        print(f"Task Type: {res.get('task_type')}")
        print(f"Confidence: {res.get('confidence_score')}")
        # print(f"Full: {json.dumps(res, indent=2)}")

if __name__ == "__main__":
    debug_intent()
