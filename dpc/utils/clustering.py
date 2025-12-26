from typing import List, Dict, Any, Tuple, Set, Optional, Union
from collections import Counter, defaultdict
from .db_utils import execute_sql

class ExecutionGroup:
    """
    Represents a group of SQL queries that yield the same execution result (order-independent).
    Only valid executions are stored here.
    """
    def __init__(self, result_set: frozenset, sql_list: List[str]):
        self.result_set = result_set
        self.sql_list = sql_list
        self.size = len(sql_list)

    @property
    def representative_sql(self) -> str:
        """Pick the shortest SQL as the representative (heuristic for simplicity)."""
        return min(self.sql_list, key=len)

def cluster_sql_candidates(db_path: str, candidate_sqls: List[str], timeout: int = 30) -> List[ExecutionGroup]:
    """
    Groups candidate SQLs by their execution results on the database.
    Invalid SQLs (that throw errors or timeout) are excluded from clustering.
    
    Args:
        db_path: Path to the SQLite database.
        candidate_sqls: A list of SQL strings.
        timeout: Maximum execution time for each SQL.
        
    Returns:
        A list of ExecutionGroup objects, sorted by group size descending.
    """
    result_to_sqls = defaultdict(list)
    
    for sql in candidate_sqls:
        if not sql or not sql.strip():
            continue
            
        try:
            # Execute and get result (List[Tuple])
            result = execute_sql(db_path, sql, timeout=timeout)
            
            # Convert to frozenset for order-independent clustering.
            # Even an empty result [] becomes an empty frozenset().
            result_key = frozenset(result)
            result_to_sqls[result_key].append(sql)
            
        except Exception:
            # Skip SQLs that fail to execute or timeout
            # In DPC, we only care about candidates that produce a valid result set
            continue
            
    # Create groups from successful executions
    groups = [ExecutionGroup(res, sqls) for res, sqls in result_to_sqls.items()]
    
    # Sort groups by size (Majority Vote)
    groups.sort(key=lambda g: g.size, reverse=True)
    
    return groups

def select_champion_and_challenger(groups: List[ExecutionGroup]) -> Tuple[Optional[str], Optional[str]]:
    """
    Selects the top-1 (Champion) and top-2 (Challenger) SQLs from the valid groups.
    """
    if len(groups) == 0:
        return None, None
        
    champion = groups[0].representative_sql
    challenger = groups[1].representative_sql if len(groups) > 1 else None
    
    return champion, challenger

