from typing import Any, Dict, List, Optional
import pandas as pd
import io
import contextlib
import base64
import matplotlib.pyplot as plt
from src.utils.llm_client import LLMClient

class MachineLearningAgent:
    """
    Agent responsible for generating and executing Python code for ML tasks.
    Can train models, generate forecasts, and create advanced visualizations.
    """
    
    def __init__(self, llm_client: LLMClient = None):
        self.llm = llm_client or LLMClient()
        
    def generate_code(self, intent: str, data_sample: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate Python code to perform the requested ML task on the data.
        """
        
        columns_info = list(data_sample.columns)
        dtypes = data_sample.dtypes.to_dict()
        head_csv = data_sample.head(3).to_csv(index=False)
        
        prompt = f"""
        You are a Python Data Science expert.
        Write a complete Python script to solve the user's request using the provided dataframe `df`.
        
        User Request: "{intent}"
        
        Input Data (`df`):
        - Columns: {columns_info}
        - Types: {dtypes}
        - Sample:
        {head_csv}
        
        Requirements:
1. Assume `df` is already loaded (it is a Pandas DataFrame).
2. Use `scikit-learn` for modeling (LinearRegression, RandomForest, KMeans, etc.).
3. DO NOT use `prophet`, `arima`, or `statsmodels`.
4. Use `matplotlib.pyplot` or `seaborn` for plotting.
5. If you train a model, save it to a variable named `model`.
6. If you create a plot, DO NOT call `plt.close()` or `plt.show()`. The engine captures the active figure.
7. Store the final result (e.g., forecast dataframe, metrics dict) in a variable named `result`.
8. SAFETY RULES (CRITICAL):
   - You MUST define every variable before you use it.
   - NO implied variables.
   - Example Correction:
     - BAD: `last_date` (undefined)
     - GOOD: `last_date = df['date'].max()`
   
   - **For Forecasting**:
     - Define `last_observed_date` explicitly before using it.
     - Use `pd.to_datetime` for date columns.
   
   - **For Mapping/Segmentation**:
     - Do NOT use `.map()` on a numpy array. Use `pd.Series(labels).map(mapping_dict)`.
     - Define the mapping dictionary explicitly.
   
9. Import necessary libraries (`import pandas as pd`, `from sklearn...`).
10. RETURN ONLY THE PYTHON CODE. NO MARKDOWN BLOCKS.
        """
        
        response_schema = {
            "type": "object",
            "properties": {
                "python_code": {"type": "string", "description": "Executable Python script"},
                "explanation": {"type": "string", "description": "Brief explanation of the approach"}
            },
            "required": ["python_code", "explanation"]
        }
        
        return self.llm.generate_structured_output(prompt, response_schema)

    def execute_code(self, code: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute the generated python code in a controlled environment.
        """
        # Capture standard output
        stdout_capture = io.StringIO()
        
        # Force Matplotlib backend to Agg to prevent GUI crashes
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Dictionary to store local variables (execution context)
        local_scope = {"df": df.copy(), "pd": pd, "plt": plt}
        
        try:
            with contextlib.redirect_stdout(stdout_capture):
                # Use the same dictionary for globals and locals to support list comprehensions
                exec(code, local_scope, local_scope)
                
            # Extract artifacts
            result = local_scope.get("result", None)
            model = local_scope.get("model", None)
            
            # Capture plot if exists
            plot_base64 = None
            if plt.get_fignums():
                img_bytes = io.BytesIO()
                plt.savefig(img_bytes, format='png')
                img_bytes.seek(0)
                plot_base64 = base64.b64encode(img_bytes.read()).decode()
                plt.close('all')
                
            return {
                "success": True,
                "result": result, # Can be DataFrame, Dict, etc.
                "model": model, # Model object
                "plot_base64": plot_base64,
                "logs": stdout_capture.getvalue()
            }
            
        except Exception as e:
            print(f"FAILED CODE:\n{code}\n")
            return {
                "success": False,
                "error": str(e),
                "logs": stdout_capture.getvalue()
            }
