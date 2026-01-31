# WorkflowAgent Integration - Quick Reference

## 🚀 Quick Start

### Run Integration Test
```bash
cd /Users/anshulparmar/Documents/Personal/MTech/Dissertation/Projects/mvp-1
source venv/bin/activate
python scripts/test_workflow_integration.py
```

### Use in Code
```python
from src.agents.orchestrator import OrchestratorAgent

orchestrator = OrchestratorAgent()
result = orchestrator.run("Show me daily sales by region and category")

# Access workflow results
workflow = result["workflow_result"]
print(f"DAG: {workflow['dag_file']}")
print(f"Cron: {workflow['cron_file']}")
print(f"Schedule: {workflow['schedule']}")
```

## 📁 Key Files

### Modified Files
- `src/agents/orchestrator.py` - Added WorkflowAgent integration
- `src/agents/packaging_agent.py` - Added SLA section

### New Files
- `scripts/test_workflow_integration.py` - Integration test
- `docs/WORKFLOW_AGENT_INTEGRATION.md` - Full documentation
- `docs/WORKFLOW_INTEGRATION_SUMMARY.md` - Summary
- `docs/WORKFLOW_INTEGRATION_DIAGRAMS.md` - Visual diagrams

### Generated Files (Examples)
- `generated_workflows/dags/Daily_Sales_Metrics_v1_0_0.py`
- `generated_workflows/cron/Daily Sales Metrics.sh`

## 🔧 Configuration

### SLA Format
```python
"sla": {
    "freshness": "daily at 6:00 AM UTC",  # or "hourly", "weekly", "monthly"
    "latency": "< 1 hour",
    "completeness": "> 99%"
}
```

### Schedule Mapping
| SLA Freshness | Cron Schedule |
|--------------|---------------|
| "Daily at 6:00 AM UTC" | `00 6 * * *` |
| "Updated hourly" | `@hourly` |
| "Weekly on Monday" | `0 6 * * 1` |
| "Monthly on the 1st" | `0 6 1 * *` |

## 📊 State Structure

### Input (from PackagingAgent)
```python
{
    "data_product_spec": {
        "metadata": {...},
        "data_model": {...},
        "transformation": {...},
        "quality_assurance": {...},
        "sla": {...}  # Required for workflow generation
    }
}
```

### Output (from WorkflowAgent)
```python
{
    "workflow_result": {
        "dag_code": "...",           # Full Airflow DAG code
        "cron_code": "...",          # Full cron bash script
        "schedule": "00 6 * * *",    # Extracted cron schedule
        "data_product_id": "...",    # Unique identifier
        "dag_file": "...",           # Path to DAG file
        "cron_file": "..."           # Path to cron file
    }
}
```

## 🧪 Testing

### Unit Tests
```bash
pytest tests/test_workflow_agent.py -v
```

### Integration Test
```bash
python scripts/test_workflow_integration.py
```

### Demo Script
```bash
python scripts/demo_workflow_agent.py
```

## 🔍 Troubleshooting

### Issue: workflow_result is None
**Solution**: Check that `data_product_spec` exists and has `sla` section

### Issue: Invalid schedule
**Solution**: Verify SLA freshness format or use explicit cron expression

### Issue: Files not created
**Solution**: Check `generated_workflows/` directory permissions

### Issue: croniter not found
**Solution**: Install in venv: `pip install croniter`

## 📈 Pipeline Flow

```
User Request
    ↓
Intent Agent
    ↓
Discovery Agent
    ↓
Modeling Agent
    ↓
Transformation Agent
    ↓
Quality Agent
    ↓
Packaging Agent
    ↓
Workflow Agent ⭐ NEW
    ↓
Complete Data Product + Workflows
```

## 🎯 Key Methods

### OrchestratorAgent
- `_run_workflow(state)` - Generate workflows from spec

### WorkflowAgent
- `generate_airflow_dag(spec, id)` - Create Airflow DAG
- `generate_cron_job(spec, id)` - Create cron script
- `_extract_schedule(sla)` - Parse schedule from SLA
- `validate_cron_schedule(expr)` - Validate cron expression
- `get_next_execution_times(expr, count)` - Calculate next runs

## 📝 Example Usage

### Complete Example
```python
from src.agents.orchestrator import OrchestratorAgent

# Initialize
orchestrator = OrchestratorAgent()

# Process request
result = orchestrator.run("Show me daily sales by region and category")

# Check for errors
if result.get("errors"):
    print("Errors:", result["errors"])
else:
    # Access results
    spec = result["data_product_spec"]
    workflow = result["workflow_result"]
    
    print(f"Data Product: {spec['metadata']['name']}")
    print(f"Schedule: {workflow['schedule']}")
    print(f"DAG File: {workflow['dag_file']}")
    print(f"Cron File: {workflow['cron_file']}")
```

## 🔗 Related Documentation

- [Full Integration Guide](WORKFLOW_AGENT_INTEGRATION.md)
- [Visual Diagrams](WORKFLOW_INTEGRATION_DIAGRAMS.md)
- [Integration Summary](WORKFLOW_INTEGRATION_SUMMARY.md)
- [Project README](../README.md)

## ✅ Checklist

- [x] WorkflowAgent imported in orchestrator
- [x] Workflow node added to graph
- [x] State extended with workflow_result
- [x] SLA added to packaging
- [x] Integration test created
- [x] Documentation written
- [x] Tests passing
- [x] Files generating correctly

## 🎉 Success Criteria

✅ Orchestrator runs without errors  
✅ workflow_result is populated  
✅ DAG file created in generated_workflows/dags/  
✅ Cron file created in generated_workflows/cron/  
✅ Schedule correctly extracted from SLA  
✅ Generated code is valid Python/Bash  

---

**Last Updated**: 2026-01-30  
**Status**: ✅ Complete
