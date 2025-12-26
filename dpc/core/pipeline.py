import logging
from typing import List, Dict, Any, Optional, Tuple
from ..utils.clustering import cluster_sql_candidates, select_champion_and_challenger
from ..utils.schema_utils import SchemaExtractor, TableSchema
from ..utils.db_utils import execute_sql_pd
from ..agents.slicer_agent import SlicerAgent
from ..agents.tester_agent import TesterAgent
from ..agents.solver_agent import PythonSolverAgent
from ..eval.metrics import DPCEvaluator

logger = logging.getLogger(__name__)

class DPCPipeline:
    """
    The main DPC (Dual-Program Consistency) execution pipeline.
    """
    def __init__(self, slicer: SlicerAgent, tester: TesterAgent, solver: PythonSolverAgent):
        self.slicer = slicer
        self.tester = tester
        self.solver = solver

    def run(
        self, 
        question: str, 
        db_path: str, 
        candidate_sqls: List[str], 
        evidence: Optional[str] = None,
        sql_timeout: int = 30,
        epsilon: float = 0.05,
        max_correction_attempts: int = 3
    ) -> Dict[str, Any]:
        """
        Runs the full DPC process for a single NL query.
        
        Returns:
            A dictionary containing the selected SQL and execution metadata.
        """
        # Phase 1: Generation & Clustering (Selection)
        groups = cluster_sql_candidates(db_path, candidate_sqls, timeout=sql_timeout)
        champion_sql, challenger_sql = select_champion_and_challenger(groups)
        
        if not champion_sql:
            raise ValueError("No valid SQL candidates to process.")
            
        # If there's no challenger, we default to the champion
        if not challenger_sql:
            logger.info("No challenger found, returning champion.")
            return self._format_result(champion_sql, "No Challenger")

        # Phase 2: Evidence Generation
        full_schema = SchemaExtractor.extract(db_path)
        
        # 1. Schema Slicing
        try:
            sliced_schema = self.slicer.run(
                candidate_sqls=[champion_sql, challenger_sql], 
                full_schema=full_schema,
                max_correction_attempts=max_correction_attempts
            )
        except Exception as e:
            logger.error(f"Slicer failed: {e}. Falling back to champion.")
            return self._format_result(champion_sql, f"Slicer Error: {str(e)}")

        # 2. Distinguishing Test Data Generation
        try:
            test_data = self.tester.run(
                question=question,
                sql_1=champion_sql,
                sql_2=challenger_sql,
                sliced_schema=sliced_schema,
                evidence=evidence,
                max_correction_attempts=max_correction_attempts
            )
        except Exception as e:
            logger.error(f"Tester failed: {e}. Falling back to champion.")
            return self._format_result(champion_sql, f"Tester Error: {str(e)}")

        # Phase 3: Cross-Modal Verification (The Duel)
        try:
            # 1. Python Solver (The Proxy Ground Truth)
            py_result = self.solver.run(
                question=question,
                test_data=test_data,
                sliced_schema=sliced_schema,
                evidence=evidence,
                max_correction_attempts=max_correction_attempts
            )
            
            # 2. SQL Results on Test Data
            # Note: We execute on the generated test_data, not the original DB.
            sql_res_1 = self._execute_sql_on_data(champion_sql, test_data, timeout=sql_timeout)
            sql_res_2 = self._execute_sql_on_data(challenger_sql, test_data, timeout=sql_timeout)
            
            # 3. Triangulation (F1 Scoring)
            score_1 = DPCEvaluator.evaluate(sql_res_1, py_result)
            score_2 = DPCEvaluator.evaluate(sql_res_2, py_result)
            
            logger.info(f"DPC Duel: Champ Score={score_1:.4f}, Chall Score={score_2:.4f}")
            
            # Decision Logic
            if score_2 > score_1 + epsilon:
                logger.info("Challenger wins the duel!")
                return self._format_result(challenger_sql, "Challenger Won Duel", score_1, score_2)
            else:
                logger.info("Champion retained.")
                return self._format_result(champion_sql, "Champion Retained", score_1, score_2)

        except Exception as e:
            logger.error(f"Duel failed: {e}. Falling back to champion.")
            return self._format_result(champion_sql, f"Duel Error: {str(e)}")

    def _execute_sql_on_data(
        self, 
        sql: str, 
        test_data: Dict[str, List[Dict[str, Any]]],
        timeout: int = 10
    ) -> Any:
        """Helper to run SQL on the synthetic test data."""
        import sqlite3
        import pandas as pd
        import threading
        
        # Create an in-memory SQLite database
        conn = sqlite3.connect(":memory:")
        
        # Add timeout protection even for in-memory DB
        timer = threading.Timer(timeout, conn.interrupt)
        timer.start()
        
        try:
            for table_name, rows in test_data.items():
                pd.DataFrame(rows).to_sql(table_name, conn, index=False)
            
            # Execute SQL
            return pd.read_sql_query(sql, conn)
        except Exception as e:
            if "interrupted" in str(e).lower():
                logger.error(f"SQL execution on test data timed out after {timeout}s")
            raise e
        finally:
            timer.cancel()
            conn.close()

    def _format_result(self, sql: str, reason: str, s1: float = 0.0, s2: float = 0.0) -> Dict[str, Any]:
        return {
            "selected_sql": sql,
            "selection_reason": reason,
            "champion_score": s1,
            "challenger_score": s2
        }

