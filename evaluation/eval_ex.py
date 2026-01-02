import os
import sys
import json
import argparse
import logging
import sqlite3
import threading
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpc.datasets.spider_loader import SpiderLoader
from dpc.datasets.bird_loader import BirdLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Eval-EX")

def execute_sql(sql: str, db_path: str, timeout: int = 30) -> Optional[set]:
    """Executes SQL and returns the result set as a set of tuples."""
    if not sql:
        return None
    conn = None
    timer = None
    try:
        conn = sqlite3.connect(db_path)
        # Use a timer thread to interrupt long-running queries
        timer = threading.Timer(timeout, conn.interrupt)
        timer.start()
        
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        return set(tuple(row) for row in results)
    except Exception:
        return None
    finally:
        if timer:
            timer.cancel()
        if conn:
            conn.close()

def process_sample_ex(task: Dict[str, Any]) -> Dict[str, Any]:
    """Worker function to evaluate execution accuracy for a single sample."""
    qid = task["qid"]
    pred_sql = task["pred_sql"]
    gold_sql = task["gold_sql"]
    db_path = task["db_path"]
    sql_timeout = task["sql_timeout"]
    difficulty = task["difficulty"]

    if not pred_sql:
        return {"qid": qid, "correct": False, "difficulty": difficulty}

    # Execute both
    pred_res = execute_sql(pred_sql, db_path, timeout=sql_timeout)
    gold_res = execute_sql(gold_sql, db_path, timeout=sql_timeout)
    
    # Direct set comparison
    is_correct = (pred_res is not None and gold_res is not None and pred_res == gold_res)
    return {"qid": qid, "correct": is_correct, "difficulty": difficulty}

def main():
    parser = argparse.ArgumentParser(description="Evaluate SQL execution accuracy.")
    parser.add_argument("--pred_path", type=str, required=True, help="Path to predicted SQLs JSON file.")
    parser.add_argument("--dataset_type", type=str, choices=["spider", "bird"], required=True, help="Type of dataset.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset file.")
    parser.add_argument("--db_root_path", type=str, required=True, help="Path to database root directory.")
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

    # 2. Load Predicted SQLs
    if not os.path.exists(args.pred_path):
        raise FileNotFoundError(f"Predicted SQLs file not found: {args.pred_path}")
    
    with open(args.pred_path, 'r', encoding='utf-8') as f:
        pred_map = json.load(f)
    
    # Standardized format: {"qid": "sql"}
    # Ensure all keys are strings
    pred_map = {str(k): v for k, v in pred_map.items()}

    # 3. Prepare Tasks
    tasks = []
    for i in range(len(loader)):
        item = loader.get_item(i)
        qid = str(item.question_id)
        
        if qid not in pred_map:
            continue
            
        tasks.append({
            "qid": qid,
            "pred_sql": pred_map[qid],
            "gold_sql": item.ground_truth,
            "db_path": loader.get_db_path(item.db_id),
            "sql_timeout": args.timeout,
            "difficulty": item.difficulty
        })

    # 4. Parallel Evaluation
    total = len(tasks)
    correct = 0
    diff_stats = {} # {difficulty: {"correct": 0, "total": 0}}
    
    logger.info(f"Starting parallel execution evaluation for {total} samples...")
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_sample_ex, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=total, desc="Evaluating EX"):
            res = future.result()
            diff = res["difficulty"]
            if diff not in diff_stats:
                diff_stats[diff] = {"correct": 0, "total": 0}
            
            diff_stats[diff]["total"] += 1
            if res["correct"]:
                correct += 1
                diff_stats[diff]["correct"] += 1

    # 5. Report
    accuracy = (correct / total) * 100 if total > 0 else 0
    logger.info("=" * 30)
    logger.info(f"Overall Execution Accuracy: {accuracy:.2f}%")
    logger.info(f"Total Evaluated: {total}")
    logger.info(f"Correct: {correct}")
    
    if diff_stats:
        logger.info("-" * 20)
        logger.info("Difficulty Breakdown:")
        # Sort difficulties for consistent output
        for diff in sorted(diff_stats.keys()):
            c = diff_stats[diff]["correct"]
            t = diff_stats[diff]["total"]
            acc = (c / t) * 100 if t > 0 else 0
            logger.info(f"  {diff:12}: {acc:6.2f}% ({c}/{t})")
            
    logger.info("=" * 30)

if __name__ == "__main__":
    main()

