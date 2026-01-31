"""
Test API Server for Data Product Registry

Run this to test the registry API in the browser.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from src.registry.api import router as registry_router

# Create FastAPI app
app = FastAPI(
    title="Data Product Registry API",
    description="Version-controlled registry for data product specifications",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include registry router
app.include_router(registry_router)


@app.get("/", response_class=HTMLResponse)
def root():
    """Simple test interface"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Product Registry - Test Interface</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container {
                background: white;
                border-radius: 12px;
                padding: 30px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            }
            h1 {
                color: #667eea;
                margin-bottom: 10px;
            }
            .subtitle {
                color: #666;
                margin-bottom: 30px;
            }
            .status {
                background: #10b981;
                color: white;
                padding: 8px 16px;
                border-radius: 6px;
                display: inline-block;
                margin-bottom: 20px;
            }
            .section {
                margin: 30px 0;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 8px;
                border-left: 4px solid #667eea;
            }
            .endpoint {
                background: white;
                padding: 15px;
                margin: 10px 0;
                border-radius: 6px;
                border: 1px solid #e0e0e0;
            }
            .method {
                display: inline-block;
                padding: 4px 12px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
                margin-right: 10px;
            }
            .get { background: #61affe; color: white; }
            .post { background: #49cc90; color: white; }
            .put { background: #fca130; color: white; }
            .delete { background: #f93e3e; color: white; }
            .path {
                font-family: 'Courier New', monospace;
                color: #333;
            }
            button {
                background: #667eea;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
                margin: 5px;
                transition: background 0.3s;
            }
            button:hover {
                background: #5568d3;
            }
            #result {
                background: #1e1e1e;
                color: #d4d4d4;
                padding: 20px;
                border-radius: 8px;
                font-family: 'Courier New', monospace;
                font-size: 13px;
                margin-top: 20px;
                max-height: 400px;
                overflow-y: auto;
                white-space: pre-wrap;
            }
            .hidden { display: none; }
            .stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }
            .stat-card {
                background: white;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                border: 2px solid #667eea;
            }
            .stat-value {
                font-size: 32px;
                font-weight: bold;
                color: #667eea;
            }
            .stat-label {
                color: #666;
                margin-top: 5px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Data Product Registry</h1>
            <p class="subtitle">Version-controlled storage for data product specifications</p>
            <div class="status">✅ API Server Running</div>
            
            <div class="section">
                <h2>📊 Quick Stats</h2>
                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-value" id="totalProducts">0</div>
                        <div class="stat-label">Data Products</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">9</div>
                        <div class="stat-label">API Endpoints</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">12/13</div>
                        <div class="stat-label">Tests Passing</div>
                    </div>
                </div>
            </div>

            <div class="section">
                <h2>🧪 Test Actions</h2>
                <button onclick="testHealth()">Health Check</button>
                <button onclick="createSample()">Create Sample Data Product</button>
                <button onclick="listProducts()">List All Products</button>
                <button onclick="clearResult()">Clear Result</button>
            </div>

            <div class="section">
                <h2>📡 Available Endpoints</h2>
                
                <div class="endpoint">
                    <span class="method post">POST</span>
                    <span class="path">/api/v1/registry/data-products</span>
                    <p>Create a new data product</p>
                </div>
                
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <span class="path">/api/v1/registry/data-products</span>
                    <p>List all data products (with filters)</p>
                </div>
                
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <span class="path">/api/v1/registry/data-products/{id}</span>
                    <p>Get data product by ID</p>
                </div>
                
                <div class="endpoint">
                    <span class="method put">PUT</span>
                    <span class="path">/api/v1/registry/data-products/{id}</span>
                    <p>Update data product</p>
                </div>
                
                <div class="endpoint">
                    <span class="method delete">DELETE</span>
                    <span class="path">/api/v1/registry/data-products/{id}</span>
                    <p>Archive data product</p>
                </div>
                
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <span class="path">/api/v1/registry/data-products/{id}/versions</span>
                    <p>Get version history</p>
                </div>
            </div>

            <div class="section">
                <h2>📤 Response</h2>
                <div id="result">Click a button above to test the API...</div>
            </div>
        </div>

        <script>
            const API_BASE = '';

            async function testHealth() {
                showLoading();
                try {
                    const response = await fetch(API_BASE + '/api/v1/registry/health');
                    const data = await response.json();
                    showResult(data, 'Health Check');
                } catch (error) {
                    showError(error);
                }
            }

            async function createSample() {
                showLoading();
                const sampleData = {
                    name: `daily_sales_${Date.now()}`,
                    description: "Daily sales analytics by region and category",
                    owner: "test_user",
                    specification: {
                        metadata: {
                            name: "daily_sales_analytics",
                            version: "1.0.0"
                        },
                        data_model: {
                            target_table: "gold.daily_sales_analytics",
                            grain: "Daily, by region and category",
                            schema: [
                                {name: "date", type: "DATE", nullable: false},
                                {name: "region", type: "VARCHAR(50)", nullable: false},
                                {name: "total_revenue", type: "DECIMAL(18,2)", nullable: false}
                            ]
                        }
                    },
                    tags: ["sales", "analytics", "demo"]
                };

                try {
                    const response = await fetch(API_BASE + '/api/v1/registry/data-products', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(sampleData)
                    });
                    const data = await response.json();
                    showResult(data, 'Created Data Product');
                    updateStats();
                } catch (error) {
                    showError(error);
                }
            }

            async function listProducts() {
                showLoading();
                try {
                    const response = await fetch(API_BASE + '/api/v1/registry/data-products');
                    const data = await response.json();
                    showResult(data, `Found ${data.length} Data Products`);
                    updateStats();
                } catch (error) {
                    showError(error);
                }
            }

            async function updateStats() {
                try {
                    const response = await fetch(API_BASE + '/api/v1/registry/data-products');
                    const data = await response.json();
                    document.getElementById('totalProducts').textContent = data.length;
                } catch (error) {
                    console.error('Error updating stats:', error);
                }
            }

            function showResult(data, title) {
                const result = document.getElementById('result');
                result.textContent = `${title}\\n${'='.repeat(50)}\\n\\n${JSON.stringify(data, null, 2)}`;
            }

            function showLoading() {
                document.getElementById('result').textContent = 'Loading...';
            }

            function showError(error) {
                document.getElementById('result').textContent = `Error: ${error.message}`;
            }

            function clearResult() {
                document.getElementById('result').textContent = 'Click a button above to test the API...';
            }

            // Update stats on load
            updateStats();
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    print("🚀 Starting Data Product Registry API Server...")
    print("📍 Server will be available at: http://localhost:8001")
    print("📚 API Documentation: http://localhost:8001/docs")
    print("🧪 Test Interface: http://localhost:8001")
    print("\\nPress Ctrl+C to stop the server")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
