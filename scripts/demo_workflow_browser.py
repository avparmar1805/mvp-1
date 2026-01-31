"""
Browser demo for WorkflowAgent

Interactive web interface to demonstrate workflow generation.
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
from typing import Dict, Any

from src.agents.workflow_agent import WorkflowAgent

# Create FastAPI app
app = FastAPI(
    title="Workflow Agent Demo",
    description="Interactive demo for Airflow DAG and cron job generation",
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

# Initialize WorkflowAgent
workflow_agent = WorkflowAgent()


class GenerateRequest(BaseModel):
    """Request model for workflow generation"""
    spec: Dict[str, Any]
    workflow_type: str  # 'airflow' or 'cron'


@app.get("/", response_class=HTMLResponse)
def root():
    """Interactive demo interface"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Workflow Agent Demo</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Inter', 'Segoe UI', sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            
            .container {
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 16px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                overflow: hidden;
            }
            
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }
            
            .header h1 {
                font-size: 32px;
                margin-bottom: 10px;
            }
            
            .header p {
                opacity: 0.9;
                font-size: 16px;
            }
            
            .content {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                padding: 30px;
            }
            
            .panel {
                background: #f8f9fa;
                border-radius: 12px;
                padding: 25px;
                border: 2px solid #e0e0e0;
            }
            
            .panel h2 {
                color: #667eea;
                margin-bottom: 20px;
                font-size: 20px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .form-group {
                margin-bottom: 20px;
            }
            
            label {
                display: block;
                margin-bottom: 8px;
                font-weight: 600;
                color: #333;
            }
            
            input, select, textarea {
                width: 100%;
                padding: 12px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-size: 14px;
                font-family: inherit;
                transition: border-color 0.3s;
            }
            
            input:focus, select:focus, textarea:focus {
                outline: none;
                border-color: #667eea;
            }
            
            textarea {
                font-family: 'Courier New', monospace;
                resize: vertical;
                min-height: 100px;
            }
            
            .button-group {
                display: flex;
                gap: 10px;
                margin-top: 20px;
            }
            
            button {
                flex: 1;
                padding: 14px 24px;
                border: none;
                border-radius: 8px;
                font-size: 15px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
            }
            
            .btn-primary {
                background: #667eea;
                color: white;
            }
            
            .btn-primary:hover {
                background: #5568d3;
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
            }
            
            .btn-secondary {
                background: #10b981;
                color: white;
            }
            
            .btn-secondary:hover {
                background: #059669;
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
            }
            
            .output {
                background: #1e1e1e;
                color: #d4d4d4;
                padding: 20px;
                border-radius: 8px;
                font-family: 'Courier New', monospace;
                font-size: 13px;
                max-height: 600px;
                overflow-y: auto;
                white-space: pre-wrap;
                word-wrap: break-word;
                line-height: 1.6;
            }
            
            .output::-webkit-scrollbar {
                width: 8px;
            }
            
            .output::-webkit-scrollbar-track {
                background: #2d2d2d;
            }
            
            .output::-webkit-scrollbar-thumb {
                background: #667eea;
                border-radius: 4px;
            }
            
            .stats {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 15px;
                margin-bottom: 20px;
            }
            
            .stat-card {
                background: white;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                border: 2px solid #667eea;
            }
            
            .stat-value {
                font-size: 28px;
                font-weight: bold;
                color: #667eea;
            }
            
            .stat-label {
                color: #666;
                margin-top: 5px;
                font-size: 13px;
            }
            
            .loading {
                display: none;
                text-align: center;
                padding: 20px;
                color: #667eea;
            }
            
            .loading.active {
                display: block;
            }
            
            .spinner {
                border: 3px solid #f3f3f3;
                border-top: 3px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto 10px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .example-link {
                color: #667eea;
                text-decoration: none;
                font-size: 13px;
                display: inline-block;
                margin-top: 5px;
            }
            
            .example-link:hover {
                text-decoration: underline;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>⚙️ Workflow Agent Demo</h1>
                <p>Generate Airflow DAGs and cron jobs from data product specifications</p>
            </div>
            
            <div class="content">
                <!-- Left Panel: Input -->
                <div class="panel">
                    <h2>📝 Data Product Specification</h2>
                    
                    <div class="stats">
                        <div class="stat-card">
                            <div class="stat-value">12/12</div>
                            <div class="stat-label">Tests Passing</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">4</div>
                            <div class="stat-label">Workflow Steps</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">2</div>
                            <div class="stat-label">Output Formats</div>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label>Data Product Name</label>
                        <input type="text" id="productName" value="daily_sales_analytics" placeholder="e.g., daily_sales_analytics">
                    </div>
                    
                    <div class="form-group">
                        <label>Schedule (SLA Freshness)</label>
                        <select id="schedule">
                            <option value="Daily at 6:00 AM UTC">Daily at 6:00 AM</option>
                            <option value="Updated hourly">Hourly</option>
                            <option value="Weekly on Monday">Weekly (Monday)</option>
                            <option value="Monthly on the 1st">Monthly (1st)</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label>Transformation Language</label>
                        <select id="language">
                            <option value="SQL">SQL (DuckDB)</option>
                            <option value="PySpark">PySpark</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label>SQL Query</label>
                        <textarea id="sqlQuery">SELECT 
    DATE(order_date) AS date,
    region,
    SUM(total_amount) AS revenue
FROM bronze.orders
WHERE status = 'completed'
GROUP BY 1, 2</textarea>
                        <a href="#" class="example-link" onclick="loadExample(); return false;">Load full example →</a>
                    </div>
                    
                    <div class="button-group">
                        <button class="btn-primary" onclick="generateWorkflow('airflow')">
                            🚀 Generate Airflow DAG
                        </button>
                        <button class="btn-secondary" onclick="generateWorkflow('cron')">
                            ⏰ Generate Cron Job
                        </button>
                    </div>
                </div>
                
                <!-- Right Panel: Output -->
                <div class="panel">
                    <h2>📤 Generated Workflow</h2>
                    
                    <div class="loading" id="loading">
                        <div class="spinner"></div>
                        <p>Generating workflow...</p>
                    </div>
                    
                    <div class="output" id="output">Click a button to generate a workflow...</div>
                </div>
            </div>
        </div>

        <script>
            function loadExample() {
                document.getElementById('sqlQuery').value = `SELECT 
    DATE(o.order_date) AS date,
    o.region,
    p.category,
    SUM(o.total_amount) AS total_revenue,
    COUNT(DISTINCT o.order_id) AS order_count
FROM bronze.orders o
JOIN bronze.products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY DATE(o.order_date), o.region, p.category`;
            }

            async function generateWorkflow(type) {
                const loading = document.getElementById('loading');
                const output = document.getElementById('output');
                
                // Show loading
                loading.classList.add('active');
                output.textContent = '';
                
                // Build specification
                const spec = {
                    metadata: {
                        name: document.getElementById('productName').value,
                        version: "1.0.0",
                        description: "Generated data product",
                        owner: "demo_user"
                    },
                    sla: {
                        freshness: document.getElementById('schedule').value
                    },
                    source_datasets: [
                        {name: "bronze.orders", type: "table"},
                        {name: "bronze.products", type: "table"}
                    ],
                    data_model: {
                        target_table: "gold." + document.getElementById('productName').value,
                        grain: "Daily aggregation"
                    },
                    transformations: {
                        language: document.getElementById('language').value,
                        code: document.getElementById('sqlQuery').value
                    },
                    quality_rules: [
                        {
                            rule_id: "no_nulls",
                            rule_type: "not_null",
                            column: "date"
                        }
                    ]
                };
                
                try {
                    const response = await fetch('/generate', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            spec: spec,
                            workflow_type: type
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        output.textContent = data.code;
                    } else {
                        output.textContent = `Error: ${data.error}`;
                    }
                } catch (error) {
                    output.textContent = `Error: ${error.message}`;
                } finally {
                    loading.classList.remove('active');
                }
            }
        </script>
    </body>
    </html>
    """


@app.post("/generate")
async def generate_workflow(request: GenerateRequest):
    """Generate workflow from specification"""
    try:
        spec = request.spec
        workflow_type = request.workflow_type
        
        if workflow_type == "airflow":
            code = workflow_agent.generate_airflow_dag(
                data_product_spec=spec,
                data_product_id="dp_demo_" + spec["metadata"]["name"]
            )
        elif workflow_type == "cron":
            code = workflow_agent.generate_cron_job(
                data_product_spec=spec,
                data_product_id="dp_demo_" + spec["metadata"]["name"]
            )
        else:
            return JSONResponse({
                "success": False,
                "error": "Invalid workflow type"
            })
        
        return JSONResponse({
            "success": True,
            "code": code,
            "workflow_type": workflow_type
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        })


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "workflow_agent_demo"}


if __name__ == "__main__":
    print("🚀 Starting Workflow Agent Demo Server...")
    print("📍 Server will be available at: http://localhost:8002")
    print("🧪 Interactive Demo: http://localhost:8002")
    print("\nPress Ctrl+C to stop the server")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        log_level="info"
    )
