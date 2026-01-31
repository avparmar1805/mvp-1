"""
Workflow Agent - Generates Airflow DAGs and cron jobs from data product specifications

This agent converts data product specs into executable workflows.
"""

import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, Literal
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
from croniter import croniter


class WorkflowAgent:
    """
    Generates workflow orchestration code (Airflow DAGs or cron jobs)
    from data product specifications.
    """
    
    def __init__(self, template_dir: Optional[str] = None, output_dir: Optional[str] = None):
        """
        Initialize WorkflowAgent
        
        Args:
            template_dir: Directory containing Jinja2 templates
            output_dir: Directory to write generated files
        """
        if template_dir is None:
            template_dir = str(Path(__file__).parent.parent.parent / "templates")
        
        if output_dir is None:
            output_dir = str(Path(__file__).parent.parent.parent / "generated_workflows")
        
        self.template_dir = template_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Setup Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
    
    def generate_airflow_dag(
        self,
        data_product_spec: Dict[str, Any],
        data_product_id: str,
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate Airflow DAG from data product specification
        
        Args:
            data_product_spec: Full data product specification
            data_product_id: Unique ID of the data product
            output_file: Optional custom output file path
            
        Returns:
            Generated DAG code as string
        """
        # Extract metadata
        metadata = data_product_spec.get("metadata", {})
        name = metadata.get("name", "unnamed_data_product")
        version = metadata.get("version", "1.0.0")
        description = metadata.get("description", "")
        
        # Extract SLA and determine schedule
        sla = data_product_spec.get("sla", {})
        schedule_interval = self._extract_schedule(sla)
        
        # Extract transformation details
        transformations = data_product_spec.get("transformations", {})
        transformation_language = transformations.get("language", "SQL")
        transformation_code = transformations.get("code", "")
        
        # Extract data model
        data_model = data_product_spec.get("data_model", {})
        target_table = data_model.get("target_table", "")
        
        # Extract source datasets
        source_datasets = data_product_spec.get("source_datasets", [])
        source_dataset_names = [ds.get("name", "") for ds in source_datasets]
        
        # Extract quality rules
        quality_rules = data_product_spec.get("quality_rules", [])
        
        # Prepare template context
        context = {
            "dag_id": self._sanitize_dag_id(name, version),
            "description": description,
            "schedule_interval": schedule_interval,
            "data_product_id": data_product_id,
            "version": version,
            "owner": metadata.get("owner", "airflow"),
            "email_on_failure": True,
            "retries": 3,
            "retry_delay_minutes": 5,
            "start_date": "days_ago(1)",
            "tags": ["data_product", name.split("_")[0]],
            "project_root": str(Path.cwd()),
            "source_datasets": source_dataset_names,
            "transformation_language": transformation_language,
            "transformation_code": transformation_code,
            "database_path": "data/warehouse.db",
            "target_table": target_table,
            "quality_rules": quality_rules,
            "registry_url": os.getenv("REGISTRY_URL", "http://localhost:8001")
        }
        
        # Render template
        template = self.jinja_env.get_template("airflow_dag.py.j2")
        dag_code = template.render(**context)
        
        # Write to file if output_file specified
        if output_file is None:
            output_file = f"{self.output_dir}/dags/{context['dag_id']}.py"
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(dag_code)
        
        print(f"✅ Generated Airflow DAG: {output_path}")
        
        return dag_code
    
    def generate_cron_job(
        self,
        data_product_spec: Dict[str, Any],
        data_product_id: str,
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate cron job bash script from data product specification
        
        Args:
            data_product_spec: Full data product specification
            data_product_id: Unique ID of the data product
            output_file: Optional custom output file path
            
        Returns:
            Generated bash script as string
        """
        # Extract metadata
        metadata = data_product_spec.get("metadata", {})
        name = metadata.get("name", "unnamed_data_product")
        version = metadata.get("version", "1.0.0")
        
        # Extract SLA and determine schedule
        sla = data_product_spec.get("sla", {})
        cron_schedule = self._extract_cron_schedule(sla)
        
        # Extract transformation details
        transformations = data_product_spec.get("transformations", {})
        transformation_language = transformations.get("language", "SQL")
        
        # Extract data model
        data_model = data_product_spec.get("data_model", {})
        target_table = data_model.get("target_table", "")
        
        # Extract source datasets
        source_datasets = data_product_spec.get("source_datasets", [])
        source_dataset_names = [ds.get("name", "") for ds in source_datasets]
        
        # Prepare template context
        context = {
            "data_product_name": name,
            "data_product_id": data_product_id,
            "version": version,
            "cron_schedule": cron_schedule,
            "log_dir": f"{Path.cwd()}/logs/data_products",
            "project_root": str(Path.cwd()),
            "source_datasets": source_dataset_names,
            "transformation_language": transformation_language,
            "sql_file_path": f"sql/{name}.sql",
            "pyspark_file_path": f"spark/{name}.py",
            "database_path": "data/warehouse.db",
            "target_table": target_table,
            "registry_url": os.getenv("REGISTRY_URL", "http://localhost:8001")
        }
        
        # Render template
        template = self.jinja_env.get_template("cron_job.sh.j2")
        script_code = template.render(**context)
        
        # Write to file if output_file specified
        if output_file is None:
            output_file = f"{self.output_dir}/cron/{name}.sh"
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(script_code)
        
        # Make executable
        os.chmod(output_path, 0o755)
        
        print(f"✅ Generated cron job script: {output_path}")
        print(f"   Schedule: {cron_schedule}")
        
        return script_code
    
    def _extract_schedule(self, sla: Dict[str, Any]) -> str:
        """
        Extract Airflow schedule interval from SLA
        
        Args:
            sla: SLA dictionary from data product spec
            
        Returns:
            Airflow schedule interval string
        """
        freshness = sla.get("freshness", "").lower()
        
        # Parse common patterns - check more specific patterns first
        if "hourly" in freshness or "hour" in freshness:
            return "@hourly"
        
        elif "weekly" in freshness or "week" in freshness:
            # Check for specific day
            if "monday" in freshness:
                return "0 6 * * 1"
            return "0 6 * * 1"  # Default: Monday at 6 AM
        
        elif "monthly" in freshness or "month" in freshness:
            return "0 6 1 * *"  # 1st of month at 6 AM
        
        elif "daily" in freshness or "day" in freshness:
            # Extract time if specified (e.g., "Daily at 6:00 AM UTC")
            time_match = re.search(r'(\d{1,2}):(\d{2})', freshness)
            if time_match:
                hour, minute = time_match.groups()
                return f"{minute} {hour} * * *"  # Cron format
            return "0 6 * * *"  # Default: 6 AM daily
        
        else:
            # Default to daily
            return "0 6 * * *"
    
    def _extract_cron_schedule(self, sla: Dict[str, Any]) -> str:
        """
        Extract cron schedule expression from SLA
        
        Args:
            sla: SLA dictionary from data product spec
            
        Returns:
            Cron schedule expression
        """
        return self._extract_schedule(sla)
    
    def _sanitize_dag_id(self, name: str, version: str) -> str:
        """
        Create a valid Airflow DAG ID from name and version
        
        Args:
            name: Data product name
            version: Version string
            
        Returns:
            Sanitized DAG ID
        """
        # Replace dots with underscores in version
        version_clean = version.replace(".", "_")
        
        # Combine name and version
        dag_id = f"{name}_v{version_clean}"
        
        # Ensure it's a valid Python identifier
        dag_id = re.sub(r'[^a-zA-Z0-9_]', '_', dag_id)
        
        return dag_id
    
    def validate_cron_schedule(self, cron_expression: str) -> bool:
        """
        Validate a cron schedule expression
        
        Args:
            cron_expression: Cron expression to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            croniter(cron_expression)
            return True
        except Exception:
            return False
    
    def get_next_execution_times(self, cron_expression: str, count: int = 5) -> list:
        """
        Get next N execution times for a cron schedule
        
        Args:
            cron_expression: Cron expression
            count: Number of future executions to return
            
        Returns:
            List of datetime objects
        """
        try:
            cron = croniter(cron_expression, datetime.now())
            return [cron.get_next(datetime) for _ in range(count)]
        except Exception as e:
            print(f"Error calculating next execution times: {e}")
            return []
