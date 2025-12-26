import pandas as pd
import traceback
import multiprocessing
import queue
from typing import Dict, List, Any, Union, Optional

def _worker(test_data, code, out_queue):
    """
    Internal worker function to run in a separate process.
    """
    # 1. Prepare local namespace
    local_vars = {"pd": pd}
    for table_name, rows in test_data.items():
        local_vars[table_name] = pd.DataFrame(rows)

    try:
        # 2. Execute
        # Note: We use a limited globals dict for basic safety
        exec(code, {"pd": pd, "__builtins__": __builtins__}, local_vars)
        
        if "result" not in local_vars:
            out_queue.put(("error", "Error: The code did not define a 'result' variable."))
            return
        
        result = local_vars["result"]
        
        # 3. Enforce DataFrame check & Conversion
        if not isinstance(result, pd.DataFrame):
            try:
                if isinstance(result, (list, dict, int, float, str)):
                    result = pd.DataFrame(result) if isinstance(result, (list, dict)) else pd.DataFrame([result])
                else:
                    out_queue.put(("error", f"Error: 'result' is of type {type(result)}, but it MUST be a pandas DataFrame."))
                    return
            except Exception:
                out_queue.put(("error", f"Error: 'result' could not be converted to a DataFrame."))
                return
        
        # Put the final DataFrame back to the queue
        out_queue.put(("success", result))
        
    except Exception:
        out_queue.put(("error", traceback.format_exc()))

class PythonExecutor:
    """
    Executes generated Python/Pandas code in a separate process for isolation and safety.
    """

    @staticmethod
    def execute(test_data: Dict[str, List[Dict[str, Any]]], code: str, timeout: int = 15) -> Any:
        """
        Executes code with a hard timeout and process isolation.
        """
        clean_code = code.strip()
        if clean_code.startswith("```python"):
            clean_code = clean_code[9:]
        if clean_code.endswith("```"):
            clean_code = clean_code[:-3]
        clean_code = clean_code.strip()

        # Use a Queue to get the result from the process
        ctx = multiprocessing.get_context('spawn')
        out_queue = ctx.Queue()
        
        process = ctx.Process(target=_worker, args=(test_data, clean_code, out_queue))
        # Important: Ensure the process is NOT daemon to allow nested multiprocessing
        process.daemon = False
        process.start()
        
        try:
            # Wait for result with timeout
            # We use get() with timeout
            try:
                status, result = out_queue.get(timeout=timeout)
            except queue.Empty:
                # Handle timeout
                if process.is_alive():
                    process.terminate()
                process.join()
                return f"Error: Python execution timed out after {timeout} seconds."
            
            process.join()
            return result
        except Exception as e:
            if process.is_alive():
                process.terminate()
            process.join()
            return f"Error during process management: {str(e)}"

