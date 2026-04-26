import pandas as pd
import traceback
import sys
import logging
import multiprocessing
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


def _run_generated_code_worker(
    test_data: Dict[str, List[Dict[str, Any]]],
    clean_code: str,
    conn: Any
) -> None:
    try:
        result = PythonExecutor._execute_clean_code(test_data, clean_code)
        conn.send(("ok", result))
    except BaseException:
        conn.send(("error", traceback.format_exc()))
    finally:
        conn.close()


class PythonExecutor:
    """
    Executes generated Python/Pandas code.
    """

    @staticmethod
    def execute(test_data: Dict[str, List[Dict[str, Any]]], code: str, timeout: int = 30) -> Any:
        """
        Executes code in a child process so the timeout can be enforced.
        """
        clean_code = code.strip()
        if clean_code.startswith("```python"):
            clean_code = clean_code[9:]
        if clean_code.endswith("```"):
            clean_code = clean_code[:-3]
        clean_code = clean_code.strip()

        ctx_name = "fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn"
        ctx = multiprocessing.get_context(ctx_name)
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_run_generated_code_worker, args=(test_data, clean_code, child_conn))
        proc.start()
        child_conn.close()

        try:
            if parent_conn.poll(timeout):
                status, result = parent_conn.recv()
                proc.join(timeout=1)
                if status == "ok":
                    return result
                return result

            proc.terminate()
            proc.join(timeout=1)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=1)
            return f"Error: Python execution timed out after {timeout} seconds."
        finally:
            parent_conn.close()

    @staticmethod
    def _execute_clean_code(test_data: Dict[str, List[Dict[str, Any]]], clean_code: str) -> Any:
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
