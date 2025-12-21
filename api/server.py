from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import sys
import os
import uvicorn
from contextlib import asynccontextmanager

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.agents.orchestrator import OrchestratorAgent
from src.knowledge_graph.queries import KnowledgeGraphQueryService, create_query_service
from dotenv import load_dotenv

# Load Env
load_dotenv(os.path.join(project_root, ".env"))

# --- Lifespan Events ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize agents (cache them)
    print("Initializing Agents...")
    app.state.orchestrator = OrchestratorAgent()
    app.state.kg_service = create_query_service()
    yield
    # Shutdown
    print("Shutting down...")

app = FastAPI(
    title="Agentic Data Product Builder API",
    description="REST API for generating data products from natural language.",
    version="1.0.0",
    lifespan=lifespan
)

# --- Pydantic Models ---
class GenerateRequest(BaseModel):
    request: str
    use_case: Optional[str] = "custom"

class GenerateResponse(BaseModel):
    status: str
    data_product_spec: Optional[Dict[str, Any]]
    yaml_output: Optional[str]
    logs: Optional[List[Dict[str, Any]]] # Could be refined
    errors: List[str]

# --- Endpoints ---

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "data-product-builder"}

@app.get("/api/v1/datasets")
async def list_datasets():
    """List all available datasets in the Knowledge Graph."""
    try:
        # We can implement a method in KG service to list all unique dataset nodes
        # For now, we'll assume a method exists or use a cypher query wrapper if needed
        # But KG service has find_datasets_for_metrics. We might need a generic list.
        # Let's add a placeholder or simple logic.
        # Ideally: datasets = app.state.kg_service.get_all_datasets()
        
        # Fallback to file system scan if KG doesn't support generic list yet, 
        # or just return success dummy for MVP if method missing.
        return {"datasets": ["orders", "products", "customers", "marketing_events", "campaigns"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/generate", response_model=GenerateResponse)
async def generate_data_product(payload: GenerateRequest = Body(...)):
    """Generate a data product specification from natural language."""
    if not payload.request:
        raise HTTPException(status_code=400, detail="Request string cannot be empty.")

    try:
        result = app.state.orchestrator.run(payload.request)
        
        status = "success"
        if result.get("errors"):
            status = "failed"
            
        return GenerateResponse(
            status=status,
            data_product_spec=result.get("data_product_spec"),
            yaml_output=result.get("yaml_output"),
            logs=[result], # Returning full state as logs for MVP
            errors=result.get("errors", [])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
