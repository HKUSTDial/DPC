import os
import sys
import json
import argparse
import logging
import multiprocessing
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional, Set

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpc.datasets.spider_loader import SpiderLoader
from dpc.datasets.bird_loader import BirdLoader
from dpc.utils.db_utils import execute_sql as execute_sql_rows

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Eval-Pass@N")

def execute_sql(sql: str, db_path: str, timeout: int = 30) -> Optional[Set[Tuple[Any, ...]]]:
    """
    Executes SQL and returns the result set as a set of tuples.
    This matches the logic in eval_ex.py (row-order and duplicate insensitive).
    """
    if not sql:
        return None
    try:
        results = execute_sql_rows(db_path, sql, timeout=timeout)
        return set(tuple(row) for row in results)
    except Exception:
        return None

def process_sample_pass_n(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker function to calculate SC Pass@N for a single sample.
    """
    qid = task["qid"]
    candidate_sqls = task["candidate_sqls"]
    gold_sql = task["gold_sql"]
    db_path = task["db_path"]
    sql_timeout = task["sql_timeout"]
    k = task["k"]

    # 1. Execute Gold SQL to get correct result
    gold_res = execute_sql(gold_sql, db_path, timeout=sql_timeout)

    # 2. Execute all candidates and group them
    result_to_count = Counter()
    all_results = set()
    
    for sql in candidate_sqls:
        res = execute_sql(sql, db_path, timeout=sql_timeout)
        if res is not None:
            res_key = frozenset(res)
            result_to_count[res_key] += 1
            all_results.add(res_key)
            
    # 3. Sort groups by frequency (SC ranking)
    sorted_groups = [res for res, count in result_to_count.most_common()]

    # 4. Calculate metrics
    pass_at = [False] * max(1, k)
    upper_bound = False

    if gold_res is not None:
        gold_key = frozenset(gold_res)
        upper_bound = (gold_key in all_results)
        for i in range(1, max(1, k) + 1):
            top_i = sorted_groups[:i]
            pass_at[i - 1] = (gold_key in top_i)
    
    return {
        "qid": qid,
        "pass_at": pass_at,
        "upper_bound": upper_bound
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate Pass@K and Upper Bound for SQL candidates.")
    parser.add_argument("--candidates_path", type=str, required=True, help="Path to candidate SQLs JSON file.")
    parser.add_argument("--dataset_type", type=str, choices=["spider", "bird"], required=True, help="Type of dataset.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset file.")
    parser.add_argument("--db_root_path", type=str, required=True, help="Path to database root directory.")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout for SQL execution.")
    parser.add_argument("--k", type=int, default=2, help="Top-K groups for Pass@K (default: 2)")
    parser.add_argument("--num_workers", type=int, default=min(multiprocessing.cpu_count(), 8), help="Number of parallel workers.")
    
    args = parser.parse_args()

    # 1. Load Dataset
    if args.dataset_type.lower() == "spider":
        loader = SpiderLoader(args.data_path, args.db_root_path)
    elif args.dataset_type.lower() == "bird":
        loader = BirdLoader(args.data_path, args.db_root_path)
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    # 2. Load Candidate SQLs
    if not os.path.exists(args.candidates_path):
        raise FileNotFoundError(f"Candidate SQLs file not found: {args.candidates_path}")
    
    with open(args.candidates_path, 'r', encoding='utf-8') as f:
        candidates_map = json.load(f)
    
    # 3. Prepare Tasks
    tasks = []
    for i in range(len(loader)):
        item = loader.get_item(i)
        qid = str(item.question_id)
        
        if qid not in candidates_map:
            continue
            
        tasks.append({
            "qid": qid,
            "candidate_sqls": candidates_map[qid],
            "gold_sql": item.ground_truth,
            "db_path": loader.get_db_path(item.db_id),
            "sql_timeout": args.timeout,
            "k": args.k
        })

    # 4. Parallel Evaluation
    total = len(tasks)
    pass_correct = [0] * max(1, args.k)
    ub_correct = 0
    
    logger.info(f"Starting Pass@N evaluation for {total} samples...")
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_sample_pass_n, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=total, desc="Evaluating Pass@N"):
            res = future.result()
            pass_at = res.get("pass_at", [])
            for i in range(min(len(pass_at), len(pass_correct))):
                if pass_at[i]:
                    pass_correct[i] += 1
            
            if res.get("upper_bound"):
                ub_correct += 1

    # 5. Report
    def get_acc(c, t):
        return (c / t) * 100 if t > 0 else 0

    logger.info("=" * 40)
    logger.info(f"Evaluation Results (N={total}):")
    for i in range(1, max(1, args.k) + 1):
        c = pass_correct[i - 1]
        logger.info(f"Pass@{i} (SC):    {get_acc(c, total):.2f}% ({c}/{total})")
    logger.info(f"Upper Bound:     {get_acc(ub_correct, total):.2f}% ({ub_correct}/{total})")
    logger.info("=" * 40)

if __name__ == "__main__":
    main()
