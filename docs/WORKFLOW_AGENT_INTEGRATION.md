# WorkflowAgent Integration with Orchestrator

## Overview

The **WorkflowAgent** has been successfully integrated into the orchestrator pipeline as the final step in the data product generation workflow. This enables automatic generation of deployment-ready workflow orchestration code (Airflow DAGs and cron jobs) from the data product specification.

## Architecture

### Updated Pipeline Flow

```
User Request
    ↓
[Intent Agent] ────────→ Extract business metrics, dimensions
    ↓
[Discovery Agent] ─────→ Find relevant datasets from Knowledge Graph
    ↓
[Modeling Agent] ──────→ Design target schema
    ↓
[Transform Agent] ─────→ Generate SQL/PySpark code
    ↓
[Quality Agent] ───────→ Define data quality rules
    ↓
[Packaging Agent] ─────→ Compile into YAML specification
    ↓
[Workflow Agent] ──────→ Generate Airflow DAG & Cron Job ✨ NEW!
    ↓
Complete Data Product + Deployable Workflows
```

### State Management

The orchestrator state has been extended to include workflow generation results:

```python
class DataProductState(TypedDict):
    user_request: str
    intent: Optional[Dict[str, Any]]
    discovery_result: Optional[Dict[str, Any]]
    data_model: Optional[Dict[str, Any]]
    transformation: Optional[Dict[str, Any]]
    quality_checks: Optional[Dict[str, Any]]
    ml_result: Optional[Dict[str, Any]]
    data_product_spec: Optional[Dict[str, Any]]
    yaml_output: Optional[str]
    workflow_result: Optional[Dict[str, Any]]  # ✨ NEW!
    errors: List[str]
```

## Key Changes

### 1. Orchestrator Updates (`src/agents/orchestrator.py`)

#### Import WorkflowAgent
```python
from src.agents.workflow_agent import WorkflowAgent
```

#### Initialize WorkflowAgent
```python
def __init__(self):
    # ... other agents ...
    self.workflow_agent = WorkflowAgent()  # NEW
```

#### Add Workflow Node to Graph
```python
def _build_workflow(self):
    workflow = StateGraph(DataProductState)
    
    # ... other nodes ...
    workflow.add_node("process_workflow", self._run_workflow)  # NEW
    
    # Update edges
    workflow.add_edge("process_packaging", "process_workflow")  # NEW
    workflow.add_edge("process_workflow", END)  # Workflow is final step
```

#### Implement Workflow Generation Logic
```python
def _run_workflow(self, state: DataProductState) -> Dict[str, Any]:
    """Generate workflow orchestration from data product spec."""
    spec = state.get("data_product_spec")
    
    # Generate unique ID
    metadata = spec.get("metadata", {})
    name = metadata.get("name", "unnamed_product")
    version = metadata.get("version", "1.0.0")
    data_product_id = f"{name}_{version}".replace(" ", "_")
    
    # Generate Airflow DAG
    dag_code = self.workflow_agent.generate_airflow_dag(
        data_product_spec=spec,
        data_product_id=data_product_id
    )
    
    # Generate cron job
    cron_code = self.workflow_agent.generate_cron_job(
        data_product_spec=spec,
        data_product_id=data_product_id
    )
    
    # Extract schedule
    sla = spec.get("sla", {})
    schedule = self.workflow_agent._extract_schedule(sla)
    
    return {
        "workflow_result": {
            "dag_code": dag_code,
            "cron_code": cron_code,
            "schedule": schedule,
            "data_product_id": data_product_id,
            "dag_file": f"generated_workflows/dags/{...}.py",
            "cron_file": f"generated_workflows/cron/{...}.sh"
        }
    }
```

### 2. PackagingAgent Updates (`src/agents/packaging_agent.py`)

Added SLA section to data product specification to support workflow scheduling:

```python
spec = {
    # ... other sections ...
    "sla": {
        "freshness": intent.get("temporal_granularity", "Daily") + " at 6:00 AM UTC",
        "latency": "< 1 hour",
        "completeness": "> 99%"
    }
}
```

## Workflow Generation Output

### Workflow Result Structure

```python
{
    "workflow_result": {
        "dag_code": "...",           # Full Airflow DAG Python code
        "cron_code": "...",          # Full cron bash script
        "schedule": "0 6 * * *",     # Extracted cron schedule
        "data_product_id": "...",    # Unique identifier
        "dag_file": "...",           # Path to generated DAG file
        "cron_file": "..."           # Path to generated cron script
    }
}
```

### Generated Files

1. **Airflow DAG** (`generated_workflows/dags/{name}_v{version}.py`)
   - Complete Python DAG definition
   - 4-step workflow: validate → transform → quality → registry
   - Configurable schedule from SLA
   - Error handling and retries

2. **Cron Job** (`generated_workflows/cron/{name}.sh`)
   - Executable bash script
   - Same 4-step workflow
   - Logging and error handling
   - Cron schedule comment

## Usage Examples

### 1. Via Orchestrator Directly

```python
from src.agents.orchestrator import OrchestratorAgent

orchestrator = OrchestratorAgent()

result = orchestrator.run("Show me daily sales by region and category")

# Access workflow results
workflow = result.get("workflow_result", {})
print(f"Schedule: {workflow['schedule']}")
print(f"DAG File: {workflow['dag_file']}")
print(f"Cron File: {workflow['cron_file']}")
```

### 2. Via Integrated Demo API

```bash
# Start the integrated platform
python scripts/integrated_demo.py

# Create data product via API
curl -X POST "http://localhost:8003/process" \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me daily sales by region and category"}'

# Response includes workflow generation results
{
  "success": true,
  "specification": {...},
  "registry": {...},
  "workflow": "..."  # Generated DAG code
}
```

### 3. Via Test Script

```bash
# Run integration test
python scripts/test_workflow_integration.py

# Output shows complete pipeline including workflow generation
```

## Testing

### Unit Tests

Run WorkflowAgent tests:
```bash
pytest tests/test_workflow_agent.py -v
```

### Integration Test

Run the complete integration test:
```bash
python scripts/test_workflow_integration.py
```

This will:
1. Process a sample user request through all agents
2. Generate data product specification
3. Create Airflow DAG and cron job
4. Display results and save to `output/workflow_integration_test_result.json`

## Benefits

### 1. **Complete Automation**
- From natural language to deployable workflows
- No manual DAG creation needed
- Consistent workflow structure

### 2. **Standardization**
- All data products follow same workflow pattern
- Predictable execution steps
- Easier to maintain and debug

### 3. **Flexibility**
- Supports both Airflow and cron
- Configurable schedules from SLA
- Extensible for other orchestrators

### 4. **Integration**
- Seamless part of orchestrator pipeline
- Automatic file generation
- Registry integration ready

## Next Steps

### Immediate
- ✅ WorkflowAgent integrated into orchestrator
- ✅ State management updated
- ✅ SLA support added to packaging
- ✅ Test script created

### Future Enhancements
1. **Registry Integration**: Store workflow metadata in registry
2. **Deployment Automation**: Auto-deploy DAGs to Airflow
3. **Monitoring**: Add workflow execution tracking
4. **Validation**: Pre-deployment DAG validation
5. **Templates**: Support custom workflow templates
6. **Multi-Orchestrator**: Support for other tools (Prefect, Dagster)

## File Structure

```
mvp-1/
├── src/
│   └── agents/
│       ├── orchestrator.py          # ✨ Updated with WorkflowAgent
│       ├── packaging_agent.py       # ✨ Updated with SLA
│       └── workflow_agent.py        # Workflow generation logic
├── scripts/
│   ├── test_workflow_integration.py # ✨ NEW integration test
│   └── integrated_demo.py           # Already uses WorkflowAgent
├── templates/
│   ├── airflow_dag.py.j2           # Airflow DAG template
│   └── cron_job.sh.j2              # Cron script template
└── generated_workflows/
    ├── dags/                        # Generated Airflow DAGs
    └── cron/                        # Generated cron scripts
```

## Troubleshooting

### Workflow Not Generated

**Issue**: `workflow_result` is None or empty

**Solutions**:
1. Check that `data_product_spec` exists in state
2. Verify SLA section is present in spec
3. Check for errors in `result["errors"]`

### Invalid Schedule

**Issue**: Generated schedule is incorrect

**Solutions**:
1. Review SLA `freshness` field format
2. Check `_extract_schedule()` logic in WorkflowAgent
3. Use explicit cron format in SLA if needed

### File Not Created

**Issue**: DAG or cron file not found

**Solutions**:
1. Check `generated_workflows/` directory exists
2. Verify write permissions
3. Review WorkflowAgent output logs

## Conclusion

The WorkflowAgent integration completes the end-to-end automation pipeline, enabling the system to generate not just data product specifications but also deployment-ready workflow orchestration code. This significantly reduces the manual effort required to operationalize data products.
