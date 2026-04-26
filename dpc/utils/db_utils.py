import sqlite3
import threading
from pathlib import Path
from typing import List, Any, Tuple, Union, Optional
import pandas as pd


def _strip_leading_sql_comments(sql: str) -> str:
    remaining = sql.lstrip()
    while True:
        if remaining.startswith("--"):
            newline_idx = remaining.find("\n")
            if newline_idx == -1:
                return ""
            remaining = remaining[newline_idx + 1:].lstrip()
            continue
        if remaining.startswith("/*"):
            end_idx = remaining.find("*/")
            if end_idx == -1:
                return ""
            remaining = remaining[end_idx + 2:].lstrip()
            continue
        return remaining


def ensure_readonly_query(sql: str) -> None:
    """
    Accept only query-shaped SQL before handing it to SQLite.

    SQLite's read-only URI and PRAGMA query_only enforce the real safety
    boundary; this check gives callers a clear failure for obvious writes.
    """
    if not sql or not str(sql).strip():
        raise ValueError("SQL query is empty.")

    stripped = _strip_leading_sql_comments(str(sql))
    first_token = stripped.split(None, 1)[0].lower() if stripped else ""
    if first_token not in {"select", "with"}:
        raise ValueError(f"Only SELECT/WITH queries are allowed, got: {first_token or '<empty>'}")


def open_readonly_sqlite(db_path: str) -> sqlite3.Connection:
    """Open a SQLite database in read-only/query-only mode."""
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.execute("PRAGMA query_only = ON;")
    return conn


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
    ensure_readonly_query(sql)
    conn = open_readonly_sqlite(db_path)
    
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
    ensure_readonly_query(sql)
    conn = open_readonly_sqlite(db_path)
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
