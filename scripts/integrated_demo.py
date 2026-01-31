"""
Enhanced Integrated Demo with Data Product List and DAG Viewer

Adds:
1. View all registered data products
2. View generated Airflow DAGs
3. Better navigation
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# Import existing components
from src.agents.orchestrator import OrchestratorAgent
from src.agents.workflow_agent import WorkflowAgent
from src.registry.service import RegistryService
from src.registry.schemas import DataProductCreate
from scripts.setup_registry_db import get_session

# Create FastAPI app
app = FastAPI(
    title="Integrated Data Product Platform",
    description="End-to-end: Natural Language → Registry → Workflow",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
orchestrator = OrchestratorAgent()
workflow_agent = WorkflowAgent()


class QueryRequest(BaseModel):
    """Request model for natural language query"""
    query: str


@app.get("/", response_class=HTMLResponse)
def root():
    """Enhanced demo interface with navigation"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Integrated Data Product Platform</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            
            body {
                font-family: 'Inter', 'Segoe UI', sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            
            .container { max-width: 1600px; margin: 0 auto; }
            
            .header {
                background: white;
                border-radius: 16px;
                padding: 30px;
                margin-bottom: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .header h1 {
                font-size: 32px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            
            .nav-buttons {
                display: flex;
                gap: 10px;
            }
            
            .nav-btn {
                padding: 10px 20px;
                background: #667eea;
                color: white;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-size: 14px;
                transition: all 0.3s;
            }
            
            .nav-btn:hover {
                background: #5568d3;
                transform: translateY(-2px);
            }
            
            .nav-btn.active {
                background: #764ba2;
            }
            
            .view {
                display: none;
            }
            
            .view.active {
                display: block;
            }
            
            .panel {
                background: white;
                border-radius: 16px;
                padding: 30px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                margin-bottom: 20px;
            }
            
            .panel h2 {
                color: #667eea;
                margin-bottom: 20px;
            }
            
            table {
                width: 100%;
                border-collapse: collapse;
            }
            
            th, td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #e0e0e0;
            }
            
            th {
                background: #f8f9fa;
                font-weight: 600;
                color: #333;
            }
            
            .badge {
                display: inline-block;
                padding: 4px 12px;
                border-radius: 12px;
                font-size: 11px;
                font-weight: 600;
            }
            
            .badge-draft { background: #fbbf24; color: white; }
            .badge-active { background: #10b981; color: white; }
            
            .dag-list {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                gap: 15px;
            }
            
            .dag-card {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 12px;
                border: 2px solid #e0e0e0;
                transition: all 0.3s;
            }
            
            .dag-card:hover {
                border-color: #667eea;
                transform: translateY(-2px);
            }
            
            .dag-card h3 {
                color: #333;
                margin-bottom: 10px;
                font-size: 16px;
            }
            
            .dag-card p {
                color: #666;
                font-size: 13px;
                margin: 5px 0;
            }
            
            button {
                padding: 12px 24px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-size: 14px;
                transition: all 0.3s;
            }
            
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <!-- Header with Navigation -->
            <div class="header">
                <div>
                    <h1>🚀 Integrated Data Product Platform</h1>
                    <p style="color: #666; margin-top: 5px;">Natural Language → Registry → Workflow</p>
                </div>
                <div class="nav-buttons">
                    <button class="nav-btn active" onclick="showView('create')">Create New</button>
                    <button class="nav-btn" onclick="showView('products')">All Products</button>
                    <button class="nav-btn" onclick="showView('dags')">Generated DAGs</button>
                </div>
            </div>
            
            <!-- View 1: Create New Data Product -->
            <div id="create-view" class="view active">
                <div class="panel">
                    <h2>💬 Create New Data Product</h2>
                    <p style="color: #666; margin-bottom: 20px;">Enter a natural language query to generate a complete data product</p>
                    
                    <textarea id="query" style="width: 100%; padding: 14px; border: 2px solid #e0e0e0; border-radius: 8px; min-height: 100px; font-family: inherit;" placeholder="e.g., Show me daily sales by region and category">Show me daily sales by region and category</textarea>
                    
                    <button onclick="createProduct()" style="margin-top: 15px; width: 100%;">
                        🚀 Create Data Product
                    </button>
                    
                    <div id="create-result" style="margin-top: 20px;"></div>
                </div>
            </div>
            
            <!-- View 2: All Data Products -->
            <div id="products-view" class="view">
                <div class="panel">
                    <h2>📊 All Registered Data Products</h2>
                    <button onclick="loadProducts()" style="margin-bottom: 20px;">🔄 Refresh List</button>
                    <div id="products-list">Click "Refresh List" to load data products...</div>
                </div>
            </div>
            
            <!-- View 3: Generated DAGs -->
            <div id="dags-view" class="view">
                <div class="panel">
                    <h2>⚙️ Generated Airflow DAGs</h2>
                    <button onclick="loadDAGs()" style="margin-bottom: 20px;">🔄 Refresh List</button>
                    <div id="dags-list">Click "Refresh List" to load generated DAGs...</div>
                </div>
            </div>
        </div>

        <script>
            function showView(viewName) {
                // Hide all views
                document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
                document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
                
                // Show selected view
                document.getElementById(viewName + '-view').classList.add('active');
                event.target.classList.add('active');
                
                // Auto-load data for products and dags views
                if (viewName === 'products') loadProducts();
                if (viewName === 'dags') loadDAGs();
            }

            async function createProduct() {
                const query = document.getElementById('query').value;
                const resultDiv = document.getElementById('create-result');
                
                resultDiv.innerHTML = '<p style="color: #667eea;">Creating data product...</p>';
                
                try {
                    const response = await fetch('/process', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({query: query})
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        resultDiv.innerHTML = `
                            <div style="background: #10b981; color: white; padding: 15px; border-radius: 8px;">
                                <h3 style="margin-bottom: 10px;">✅ Data Product Created!</h3>
                                <p><strong>ID:</strong> ${data.registry.id}</p>
                                <p><strong>Version:</strong> ${data.registry.version}</p>
                                <p><strong>Status:</strong> ${data.registry.status}</p>
                                <p style="margin-top: 10px;">
                                    <a href="http://localhost:8001/api/v1/registry/data-products/${data.registry.id}" 
                                       target="_blank" 
                                       style="color: white; text-decoration: underline;">
                                        View in Registry →
                                    </a>
                                </p>
                            </div>
                        `;
                    } else {
                        resultDiv.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
                    }
                } catch (error) {
                    resultDiv.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
                }
            }

            async function loadProducts() {
                const listDiv = document.getElementById('products-list');
                listDiv.innerHTML = '<p style="color: #667eea;">Loading...</p>';
                
                try {
                    const response = await fetch('/api/products');
                    const products = await response.json();
                    
                    if (products.length === 0) {
                        listDiv.innerHTML = '<p style="color: #999;">No data products registered yet. Create one to get started!</p>';
                        return;
                    }
                    
                    let html = `
                        <table>
                            <thead>
                                <tr>
                                    <th>Name</th>
                                    <th>Version</th>
                                    <th>Status</th>
                                    <th>Owner</th>
                                    <th>Created</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                    `;
                    
                    products.forEach(p => {
                        const badgeClass = p.status === 'active' ? 'badge-active' : 'badge-draft';
                        html += `
                            <tr>
                                <td><strong>${p.name}</strong></td>
                                <td>${p.current_version}</td>
                                <td><span class="badge ${badgeClass}">${p.status}</span></td>
                                <td>${p.owner}</td>
                                <td>${new Date(p.created_at).toLocaleDateString()}</td>
                                <td>
                                    <a href="http://localhost:8001/api/v1/registry/data-products/${p.id}" 
                                       target="_blank" 
                                       style="color: #667eea;">View →</a>
                                </td>
                            </tr>
                        `;
                    });
                    
                    html += '</tbody></table>';
                    listDiv.innerHTML = html;
                } catch (error) {
                    listDiv.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
                }
            }

            async function loadDAGs() {
                const listDiv = document.getElementById('dags-list');
                listDiv.innerHTML = '<p style="color: #667eea;">Loading...</p>';
                
                try {
                    const response = await fetch('/api/dags');
                    const dags = await response.json();
                    
                    if (dags.length === 0) {
                        listDiv.innerHTML = '<p style="color: #999;">No DAGs generated yet. Create a data product to generate a DAG!</p>';
                        return;
                    }
                    
                    let html = '<div class="dag-list">';
                    
                    dags.forEach(dag => {
                        html += `
                            <div class="dag-card">
                                <h3>📄 ${dag.name}</h3>
                                <p><strong>Schedule:</strong> ${dag.schedule}</p>
                                <p><strong>Size:</strong> ${(dag.size / 1024).toFixed(1)} KB</p>
                                <p><strong>Modified:</strong> ${new Date(dag.modified).toLocaleDateString()}</p>
                                <p style="margin-top: 10px;">
                                    <a href="/api/dags/${dag.name}" target="_blank" style="color: #667eea;">View Code →</a>
                                </p>
                            </div>
                        `;
                    });
                    
                    html += '</div>';
                    listDiv.innerHTML = html;
                } catch (error) {
                    listDiv.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
                }
            }
        </script>
    </body>
    </html>
    """


@app.post("/process")
async def process_query(request: QueryRequest):
    """Process natural language query through complete pipeline"""
    try:
        query = request.query
        
        # Step 1: Generate data product specification using orchestrator
        result = orchestrator.run(query)
        spec = result.get("data_product_spec", {})
        
        # Step 2: Register in Schema Registry
        db = get_session()
        registry_service = RegistryService(db)
        
        dp_create = DataProductCreate(
            name=spec.get("metadata", {}).get("name", "unnamed_product"),
            description=spec.get("metadata", {}).get("description", ""),
            owner="demo_user",
            specification=spec,
            tags=["auto-generated", "demo"]
        )
        
        data_product = registry_service.create_data_product(dp_create, created_by="orchestrator")
        
        # Step 3: Generate Airflow workflow
        dag_code = workflow_agent.generate_airflow_dag(
            data_product_spec=spec,
            data_product_id=data_product.id
        )
        
        db.close()
        
        return JSONResponse({
            "success": True,
            "specification": spec,
            "registry": {
                "id": data_product.id,
                "version": data_product.current_version,
                "status": data_product.status.value
            },
            "workflow": dag_code
        })
        
    except Exception as e:
        import traceback
        return JSONResponse({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })


@app.get("/api/products")
async def list_products():
    """List all registered data products"""
    try:
        db = get_session()
        registry_service = RegistryService(db)
        
        products = registry_service.list_data_products()
        
        db.close()
        
        return JSONResponse([{
            "id": p.id,
            "name": p.name,
            "current_version": p.current_version,
            "status": p.status.value,
            "owner": p.owner,
            "created_at": p.created_at.isoformat(),
            "description": p.description
        } for p in products])
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/dags")
async def list_dags():
    """List all generated Airflow DAGs"""
    try:
        dags_dir = Path("generated_workflows/dags")
        
        if not dags_dir.exists():
            return JSONResponse([])
        
        dags = []
        for dag_file in dags_dir.glob("*.py"):
            # Extract schedule from DAG file
            content = dag_file.read_text()
            schedule = "Unknown"
            if 'SCHEDULE_INTERVAL = "' in content:
                start = content.find('SCHEDULE_INTERVAL = "') + len('SCHEDULE_INTERVAL = "')
                end = content.find('"', start)
                schedule = content[start:end]
            
            dags.append({
                "name": dag_file.name,
                "schedule": schedule,
                "size": dag_file.stat().st_size,
                "modified": dag_file.stat().st_mtime
            })
        
        return JSONResponse(dags)
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/dags/{dag_name}")
async def get_dag_code(dag_name: str):
    """Get DAG source code"""
    try:
        dag_file = Path(f"generated_workflows/dags/{dag_name}")
        
        if not dag_file.exists():
            return JSONResponse({"error": "DAG not found"}, status_code=404)
        
        return JSONResponse({
            "name": dag_name,
            "code": dag_file.read_text()
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "integrated_platform", "version": "2.0.0"}


if __name__ == "__main__":
    print("=" * 70)
    print("🚀 Starting Enhanced Integrated Data Product Platform v2.0")
    print("=" * 70)
    print()
    print("Features:")
    print("  ✅ Create new data products from natural language")
    print("  ✅ View all registered data products")
    print("  ✅ View all generated Airflow DAGs")
    print()
    print("📍 Server: http://localhost:8003")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 70)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8003,
        log_level="info"
    )
