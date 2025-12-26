import sqlite3
import threading
from typing import List, Any, Tuple, Union, Optional
import pandas as pd

def execute_sql(db_path: str, sql: str, timeout: int = 30) -> List[Tuple[Any, ...]]:
    """
    Execute a SQL query on a SQLite database and return the results as a list of tuples.
    Uses a timer thread to interrupt long-running queries.
    
    Args:
        db_path: Path to the SQLite database file.
        sql: The SQL query to execute.
        timeout: Execution timeout in seconds.
        
    Returns:
        List of result rows.
    """
    conn = sqlite3.connect(db_path)
    
    # Timer to interrupt the connection if it runs too long
    timer = threading.Timer(timeout, conn.interrupt)
    timer.start()
    
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        return results
    except sqlite3.OperationalError as e:
        if "interrupted" in str(e).lower():
            # You can decide to raise an error or return a specific error flag
            # Here we raise a TimeoutError for the caller to handle
            raise TimeoutError(f"SQL execution timed out after {timeout} seconds")
        raise e
    finally:
        timer.cancel()
        conn.close()

def execute_sql_pd(db_path: str, sql: str, timeout: int = 30) -> pd.DataFrame:
    """
    Execute a SQL query and return results as a Pandas DataFrame.
    """
    conn = sqlite3.connect(db_path)
    timer = threading.Timer(timeout, conn.interrupt)
    timer.start()
    try:
        df = pd.read_sql_query(sql, conn)
        return df
    except (pd.errors.DatabaseError, sqlite3.OperationalError) as e:
        if "interrupted" in str(e).lower():
            raise TimeoutError(f"Pandas SQL execution timed out after {timeout} seconds")
        raise e
    finally:
        timer.cancel()
        conn.close()

