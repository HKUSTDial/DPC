import pandas as pd
import traceback
import sys
import logging
import threading
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class PythonExecutor:
    """
    Executes generated Python/Pandas code.
    """

    @staticmethod
    def execute(test_data: Dict[str, List[Dict[str, Any]]], code: str, timeout: int = 30) -> Any:
        """
        Executes code directly using exec().
        """
        clean_code = code.strip()
        if clean_code.startswith("```python"):
            clean_code = clean_code[9:]
        if clean_code.endswith("```"):
            clean_code = clean_code[:-3]
        clean_code = clean_code.strip()

        # 1. Prepare local namespace with DataFrames
        # Sanitize table names: replace spaces with underscores to make them valid Python identifiers
        local_vars = {"pd": pd}
        for table_name, rows in test_data.items():
            try:
                var_name = table_name.replace(" ", "_")
                local_vars[var_name] = pd.DataFrame(rows)
            except Exception as e:
                logger.error(f"Failed to convert table {table_name} to DataFrame: {e}")

        # 2. Execute
        try:
            exec(clean_code, {"pd": pd, "__builtins__": __builtins__}, local_vars)
            return PythonExecutor._get_result(local_vars)
        except Exception as e:
            # Extract the traceback and skip the first frame (the 'execute' method)
            # to make it look like the code was run directly.
            _, _, tb = sys.exc_info()
            if tb is not None and tb.tb_next is not None:
                return "".join(traceback.format_exception(type(e), e, tb.tb_next))
            return traceback.format_exc()

    @staticmethod
    def _get_result(local_vars: Dict[str, Any]) -> Any:
        """Helper to extract and validate 'result' from local_vars."""
        if "result" not in local_vars:
            return "Error: The code did not define a 'result' variable."
        
        result = local_vars["result"]
            
        # Enforce DataFrame check & Conversion
        if not isinstance(result, pd.DataFrame):
            try:
                if isinstance(result, (list, dict, int, float, str)):
                    result = pd.DataFrame(result) if isinstance(result, (list, dict)) else pd.DataFrame([result])
                else:
                    return f"Error: 'result' is of type {type(result)}, but it MUST be a pandas DataFrame."
            except Exception:
                return "Error: 'result' could not be converted to a DataFrame."
        
        return result
