import duckdb
import os
import glob
from typing import Any, Dict, List, Tuple

class ExecutionEngine:
    """
    Executes SQL queries against the Bronze layer using DuckDB.
    Functions as a lightweight verification engine.
    """
    
    def __init__(self, data_dir: str = "data/bronze"):
        self.data_dir = data_dir
        self.conn = duckdb.connect(database=":memory:")
        self._register_tables()
        
    def _register_tables(self):
        """Register all parquet files in data_dir as tables."""
        if not os.path.exists(self.data_dir):
            print(f"Warning: Data directory {self.data_dir} does not exist.")
            return

        # Look for subdirectories (assuming each subdir is a dataset)
        items = os.listdir(self.data_dir)
        datasets = [d for d in items if os.path.isdir(os.path.join(self.data_dir, d))]
        
        print(f"Initializing Execution Engine. Found {len(datasets)} datasets:")
        
        for table_name in datasets:
            folder_path = os.path.join(self.data_dir, table_name)
            # Use glob pattern for partitioned parquet
            sql_path = os.path.join(folder_path, "*.parquet")
            
            try:
                # Create a view on the folder glob
                self.conn.execute(f"CREATE OR REPLACE VIEW {table_name} AS SELECT * FROM read_parquet('{sql_path}')")
                print(f"  - Registered table: {table_name}")
            except Exception as e:
                print(f"  - Failed to register {table_name}: {e}")

    def execute_query(self, sql: str) -> Tuple[bool, Any, str]:
        """
        Execute a SQL query.
        
        Returns:
            Tuple: (success: bool, result_data: List[Dict] or None, message: str)
        """
        try:
            # Execute
            cursor = self.conn.execute(sql)
            
            # Fetch column names and data
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            
            # Convert to list of dicts for easier consumption
            result_data = [dict(zip(columns, row)) for row in rows]
            
            msg = f"Executed successfully. Returned {len(result_data)} rows."
            return True, result_data, msg
            
        except Exception as e:
            return False, None, str(e)
            
    def execute_query_df(self, sql: str) -> Tuple[bool, Any, str]:
        """
        Execute a SQL query and return a Pandas DataFrame.
        """
        try:
            # Execute
            df = self.conn.execute(sql).fetchdf()
            msg = f"Executed successfully. Returned {len(df)} rows."
            return True, df, msg
            
        except Exception as e:
            return False, None, str(e)

    def get_registered_tables(self) -> List[str]:
        """List tables currently registered in DuckDB."""
        return [r[0] for r in self.conn.execute("SHOW TABLES").fetchall()]
