import os
import sys
import json
import argparse
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpc.datasets.spider_loader import SpiderLoader
from dpc.datasets.bird_loader import BirdLoader
from dpc.utils.db_utils import execute_sql as execute_sql_rows
from baseline.common import save_json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Suppress noisy HTTP logs from libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger("EX-Guided-Selection")

def execute_sql(sql: str, db_path: str, timeout: int = 30) -> Optional[set]:
    """Executes SQL and returns the result set as a set of tuples. Returns None on error."""
    if not sql:
        return None
    try:
        results = execute_sql_rows(db_path, sql, timeout=timeout)
        return set(tuple(row) for row in results)
    except Exception:
        return None

def process_sample_ex_guided(task: Dict[str, Any]) -> Dict[str, Any]:
    """Worker function to perform EX-guided selection: pick the first executable SQL."""
    qid = task["qid"]
    candidates = task["candidates"]
    db_path = task["db_path"]
    sql_timeout = task["sql_timeout"]

    if not candidates:
        return {"qid": qid, "selected_sql": None}

    # 1. Iterate through candidates and pick the first one that executes successfully
    for sql in candidates:
        res = execute_sql(sql, db_path, timeout=sql_timeout)
        if res is not None:
            return {"qid": qid, "selected_sql": sql}
    
    # 2. Fallback: If none execute successfully, pick the first candidate
    return {"qid": qid, "selected_sql": candidates[0]}

def main():
    parser = argparse.ArgumentParser(description="Perform EX-guided selection on SQL candidates: pick the first executable SQL.")
    parser.add_argument("--pred_path", type=str, required=True, help="Path to predicted SQL candidates JSON file.")
    parser.add_argument("--dataset_type", type=str, choices=["spider", "bird"], required=True, help="Type of dataset.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset file.")
    parser.add_argument("--db_root_path", type=str, required=True, help="Path to database root directory.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the selected SQLs.")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout for SQL execution.")
    parser.add_argument("--num_workers", type=int, default=min(multiprocessing.cpu_count(), 8), help="Number of parallel workers.")
    
    args = parser.parse_args()

    # 1. Load Dataset
    if args.dataset_type.lower() == "spider":
        loader = SpiderLoader(args.data_path, args.db_root_path)
    elif args.dataset_type.lower() == "bird":
        loader = BirdLoader(args.data_path, args.db_root_path)
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    # 2. Load Candidates
    if not os.path.exists(args.pred_path):
        raise FileNotFoundError(f"Predicted SQLs file not found: {args.pred_path}")
    
    with open(args.pred_path, 'r', encoding='utf-8') as f:
        all_candidates = json.load(f)
    
    # Ensure all keys are strings
    all_candidates = {str(k): v for k, v in all_candidates.items()}

    # 3. Prepare Tasks
    tasks = []
    for i in range(len(loader)):
        item = loader.get_item(i)
        qid = str(item.question_id)
        
        if qid not in all_candidates:
            continue
            
        tasks.append({
            "qid": qid,
            "candidates": all_candidates[qid],
            "db_path": loader.get_db_path(item.db_id),
            "sql_timeout": args.timeout
        })

    # 4. Parallel EX-Guided Selection
    total = len(tasks)
    selected_sqls = {}
    
    logger.info(f"Starting parallel EX-guided selection for {total} samples...")
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_sample_ex_guided, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=total, desc="EX Selection"):
            res = future.result()
            selected_sqls[res["qid"]] = res["selected_sql"]

    # 5. Save Results
    save_json(args.output_path, selected_sqls, indent=4)
    
    logger.info(f"Selected SQLs saved to {args.output_path}")

if __name__ == "__main__":
    main()
