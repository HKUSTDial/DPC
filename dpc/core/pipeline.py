import logging
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..utils.clustering import cluster_sql_candidates, select_champion_and_challenger
from ..utils.schema_utils import SchemaExtractor, TableSchema
from ..utils.db_utils import execute_sql_pd
from ..agents.slicer_agent import SlicerAgent
from ..agents.tester_agent import TesterAgent
from ..agents.solver_agent import PythonSolverAgent
from ..eval.metrics import DPCEvaluator, normalize_result, canonicalize_result

logger = logging.getLogger(__name__)

class DPCPipeline:
    """
    The main DPC (Dual-Program Consistency) execution pipeline.
    """
    def __init__(self, slicer: SlicerAgent, tester: TesterAgent, solver: PythonSolverAgent, llm: Optional[Any] = None):
        self.slicer = slicer
        self.tester = tester
        self.solver = solver
        # Use provided llm or extract from one of the agents
        self.llm = llm or slicer.llm

    def run(
        self, 
        question: str, 
        db_path: str, 
        candidate_sqls: List[str], 
        evidence: Optional[str] = None,
        sql_timeout: int = 30,
        python_timeout: int = 30,
        epsilon: float = 0.05,
        max_correction_attempts: int = 3,
        num_test_data: int = 1,
        num_solver_attempts: int = 1
    ) -> Dict[str, Any]:
        """
        Runs the DPC process with a nested Ensemble mechanism:
        - num_test_data: Number of independent test data sets to generate.
        - num_solver_attempts: Number of Python solver attempts per test data set.
        """
        self.llm.reset_usage()

        logger.info(f"--- Starting DPC Pipeline for Question: {question[:80]}... ---")
        
        # Phase 1: Generation & Clustering (Selection)
        logger.info(f"Phase 1: Clustering {len(candidate_sqls)} candidate SQLs...")
        groups = cluster_sql_candidates(db_path, candidate_sqls, timeout=sql_timeout)
        champion_sql, challenger_sql = select_champion_and_challenger(groups)
        
        if not champion_sql:
            logger.error("No valid SQL candidates to process.")
            raise ValueError("No valid SQL candidates to process.")
            
        if not challenger_sql:
            logger.info("No challenger found (all candidates consistent). Returning champion.")
            return self._format_result(champion_sql, "No Challenger")

        logger.info(f"Duel detected: [Champ] {champion_sql[:60]}... VS [Chall] {challenger_sql[:60]}...")

        # Phase 2: Evidence Generation (Schema Slicing - stable context)
        full_schema = SchemaExtractor.extract(db_path)
        try:
            logger.info("Attempting Schema Slicing...")
            sliced_schema = self.slicer.run(
                candidate_sqls=[champion_sql, challenger_sql], 
                full_schema=full_schema,
                max_correction_attempts=max_correction_attempts
            )
            logger.info(f"Slicer successful: {len(sliced_schema)} tables kept.")
        except Exception as e:
            logger.error(f"Slicer failed: {e}. Falling back to champion.")
            return self._format_result(champion_sql, f"Slicer Error: {str(e)}")

        # Phase 3: Verification (Nested Ensemble)
        logger.info(f"Phase 3: Verification (Test Data: {num_test_data}, Solver Attempts: {num_solver_attempts})...")
        
        challenger_wins = 0
        total_score_1 = 0.0
        total_score_2 = 0.0
        valid_data_points = 0

        def run_single_data_iteration(j):
            data_prefix = f"[Data {j+1}/{num_test_data}]"
            try:
                # 1. Distinguishing Test Data Generation
                logger.info(f"{data_prefix} Generating distinguishing test data...")
                test_data = self.tester.run(
                    question=question,
                    sql_1=champion_sql,
                    sql_2=challenger_sql,
                    sliced_schema=sliced_schema,
                    evidence=evidence,
                    max_correction_attempts=max_correction_attempts
                )

                # 2. SQL Execution on current test data
                sql_res_1 = self._execute_sql_on_data(champion_sql, test_data, timeout=sql_timeout)
                sql_res_2 = self._execute_sql_on_data(challenger_sql, test_data, timeout=sql_timeout)
                
                # 3. Parallel Python Solver Ensemble for THIS test data
                py_results = []
                with ThreadPoolExecutor(max_workers=num_solver_attempts) as solver_pool:
                    solver_futures = [
                        solver_pool.submit(
                            self.solver.run,
                    question=question,
                    test_data=test_data,
                    sliced_schema=sliced_schema,
                    evidence=evidence,
                    max_correction_attempts=max_correction_attempts,
                    python_timeout=python_timeout
                        ) for _ in range(num_solver_attempts)
                    ]
                    for i, fut in enumerate(as_completed(solver_futures)):
                        solver_prefix = f"{data_prefix}[Solver {i+1}/{num_solver_attempts}]"
                        try:
                            py_results.append(fut.result())
                        except Exception as e:
                            logger.warning(f"{solver_prefix} Solver attempt failed: {e}")

                if not py_results:
                    logger.warning(f"{data_prefix} All solver attempts failed for this data set.")
                    return None

                # 4. Voting & Scoring
                proxy_ground_truth = self._vote_on_python_results(py_results)
                s1 = DPCEvaluator.evaluate(sql_res_1, proxy_ground_truth)
                s2 = DPCEvaluator.evaluate(sql_res_2, proxy_ground_truth)
                logger.info(f"{data_prefix} Scores: [Champ] {s1:.4f} | [Chall] {s2:.4f}")
                
                # Decide winner for THIS data iteration
                is_challenger_win = s2 > s1 + epsilon
                return s1, s2, is_challenger_win

            except Exception as e:
                logger.error(f"{data_prefix} Data iteration failed: {e}")
                return None

        # Execute all data iterations in parallel
        with ThreadPoolExecutor(max_workers=num_test_data) as data_pool:
            data_futures = [data_pool.submit(run_single_data_iteration, j) for j in range(num_test_data)]
            for fut in as_completed(data_futures):
                res = fut.result()
                if res:
                    s1, s2, win = res
                    total_score_1 += s1
                    total_score_2 += s2
                    if win:
                        challenger_wins += 1
                    valid_data_points += 1

        # Final Decision across all valid data points (Majority Vote)
        if valid_data_points == 0:
            logger.error("Verification Phase failed for all data iterations. Falling back to champion.")
            return self._format_result(champion_sql, "Verification Error (All Data Failed)")

        avg_score_1 = total_score_1 / valid_data_points
        avg_score_2 = total_score_2 / valid_data_points
        
        logger.info(f"Final Verification ({valid_data_points} data sets): [Challenger Wins: {challenger_wins}/{valid_data_points}]")
        logger.info(f"Final Scores (Avg): [Champ] {avg_score_1:.4f} | [Chall] {avg_score_2:.4f}")

        # Final Decision based on Vote Count
        if challenger_wins > valid_data_points / 2:
            logger.info(f"SUCCESS: Challenger wins by majority vote ({challenger_wins}/{valid_data_points})!")
            return self._format_result(challenger_sql, f"Challenger Won Duel (Votes: {challenger_wins}/{valid_data_points})", avg_score_1, avg_score_2)
        else:
            logger.info(f"Champion retained. Challenger only won {challenger_wins}/{valid_data_points} rounds.")
            return self._format_result(champion_sql, f"Champion Retained (Votes: {valid_data_points-challenger_wins}/{valid_data_points})", avg_score_1, avg_score_2)

    def _vote_on_python_results(self, py_results: List[Any]) -> Any:
        """
        Majority vote on the results of multiple Python Solver executions.
        Uses canonicalized results for robust voting.
        """
        if not py_results:
            return None
        if len(py_results) == 1:
            return py_results[0]
            
        counts = {}
        for i, res in enumerate(py_results):
            # Normalize and Canonicalize for consistent comparison (order-insensitive)
            norm = normalize_result(res)
            canonical = canonicalize_result(norm)
            
            # Use string representation of the canonical list of tuples for hashing
            key = str(canonical) 
            if key not in counts:
                counts[key] = {"count": 0, "first_index": i}
            counts[key]["count"] += 1
            
        # Find the one with highest count
        winner_key = max(counts, key=lambda k: counts[k]["count"])
        winner_index = counts[winner_key]["first_index"]
        
        logger.info(f"Voting result: {counts[winner_key]['count']}/{len(py_results)} agreed on the majority result.")
        return py_results[winner_index]

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
        usage = self.llm.get_usage()
        
        return {
            "selected_sql": sql,
            "selection_reason": reason,
            "champion_score": s1,
            "challenger_score": s2,
            "token_usage": usage
        }

