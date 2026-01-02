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
logger = logging.getLogger("Majority-Analysis")

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

def process_sample(task: Dict[str, Any]) -> Dict[str, Any]:
    """Worker function to process a single sample."""
    qid = task["qid"]
    candidates = task["candidates"]
    sc_pred = task["sc_pred"]
    dpc_pred = task["dpc_pred"]
    gold_sql = task["gold_sql"]
    db_path = task["db_path"]
    sql_timeout = task["sql_timeout"]
    
    # Execute gold SQL to get correct result
    gold_res = execute_sql(gold_sql, db_path, timeout=sql_timeout)
    if gold_res is None:
        return {
            "qid": qid,
            "has_correct_candidate": False,
            "sc_correct": False,
            "dpc_correct": False,
            "sc_pred": sc_pred,
            "dpc_pred": dpc_pred
        }
    
    # Check if at least one candidate is correct
    has_correct_candidate = False
    for sql in candidates:
        if sql:
            res = execute_sql(sql, db_path, timeout=sql_timeout)
            if res is not None and res == gold_res:
                has_correct_candidate = True
                break
    
    # Check SC prediction correctness
    sc_correct = False
    if sc_pred:
        sc_res = execute_sql(sc_pred, db_path, timeout=sql_timeout)
        sc_correct = (sc_res is not None and sc_res == gold_res)
    
    # Check DPC prediction correctness
    dpc_correct = False
    if dpc_pred:
        dpc_res = execute_sql(dpc_pred, db_path, timeout=sql_timeout)
        dpc_correct = (dpc_res is not None and dpc_res == gold_res)
    
    return {
        "qid": qid,
        "has_correct_candidate": has_correct_candidate,
        "sc_correct": sc_correct,
        "dpc_correct": dpc_correct,
        "sc_pred": sc_pred,
        "dpc_pred": dpc_pred
    }

def main():
    parser = argparse.ArgumentParser(
        description="Analyze SC and DPC performance on Majority-Correct and Majority-Incorrect sets."
    )
    parser.add_argument("--candidate_path", type=str, required=True, 
                       help="Path to candidate SQLs JSON file (format: {\"qid\": [\"sql1\", \"sql2\", ...]}).")
    parser.add_argument("--sc_pred_path", type=str, required=True,
                       help="Path to SC predicted SQLs JSON file (format: {\"qid\": \"sql\"}).")
    parser.add_argument("--dpc_pred_path", type=str, required=True,
                       help="Path to DPC predicted SQLs JSON file (format: {\"qid\": \"sql\"}).")
    parser.add_argument("--dataset_type", type=str, choices=["spider", "bird"], required=True,
                       help="Type of dataset.")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to dataset file.")
    parser.add_argument("--db_root_path", type=str, required=True,
                       help="Path to database root directory.")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path to save the analysis results JSON.")
    parser.add_argument("--timeout", type=int, default=30,
                       help="Timeout for SQL execution.")
    parser.add_argument("--num_workers", type=int, default=min(multiprocessing.cpu_count(), 8),
                       help="Number of parallel workers.")
    
    args = parser.parse_args()
    
    # 1. Load Dataset
    if args.dataset_type.lower() == "spider":
        loader = SpiderLoader(args.data_path, args.db_root_path)
    elif args.dataset_type.lower() == "bird":
        loader = BirdLoader(args.data_path, args.db_root_path)
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")
    
    # 2. Load Candidates
    if not os.path.exists(args.candidate_path):
        raise FileNotFoundError(f"Candidate SQLs file not found: {args.candidate_path}")
    with open(args.candidate_path, 'r', encoding='utf-8') as f:
        all_candidates = json.load(f)
    all_candidates = {str(k): v for k, v in all_candidates.items()}
    
    # 3. Load SC Predictions
    if not os.path.exists(args.sc_pred_path):
        raise FileNotFoundError(f"SC predictions file not found: {args.sc_pred_path}")
    with open(args.sc_pred_path, 'r', encoding='utf-8') as f:
        sc_preds = json.load(f)
    sc_preds = {str(k): v for k, v in sc_preds.items()}
    
    # 4. Load DPC Predictions
    if not os.path.exists(args.dpc_pred_path):
        raise FileNotFoundError(f"DPC predictions file not found: {args.dpc_pred_path}")
    with open(args.dpc_pred_path, 'r', encoding='utf-8') as f:
        dpc_preds = json.load(f)
    dpc_preds = {str(k): v for k, v in dpc_preds.items()}
    
    # 5. Prepare Tasks
    tasks = []
    for i in range(len(loader)):
        item = loader.get_item(i)
        qid = str(item.question_id)
        
        if qid not in all_candidates:
            continue
        if qid not in sc_preds:
            continue
        if qid not in dpc_preds:
            continue
        
        tasks.append({
            "qid": qid,
            "candidates": all_candidates[qid],
            "sc_pred": sc_preds[qid],
            "dpc_pred": dpc_preds[qid],
            "gold_sql": item.ground_truth,
            "db_path": loader.get_db_path(item.db_id),
            "sql_timeout": args.timeout
        })
    
    logger.info(f"Prepared {len(tasks)} tasks for analysis...")
    
    # 6. Process Samples in Parallel
    results = {}
    logger.info(f"Starting parallel processing for {len(tasks)} samples...")
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_sample, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing"):
            res = future.result()
            results[res["qid"]] = res
    
    # 7. Filter out samples with no correct candidates
    filtered_results = {
        qid: res for qid, res in results.items()
        if res["has_correct_candidate"]
    }
    
    logger.info(f"Filtered out {len(results) - len(filtered_results)} samples with no correct candidates.")
    logger.info(f"Remaining samples: {len(filtered_results)}")
    
    # 8. Split into Majority-Correct and Majority-Incorrect sets
    majority_correct_set = []
    majority_incorrect_set = []
    
    for qid, res in filtered_results.items():
        if res["sc_correct"]:
            majority_correct_set.append(res)
        else:
            majority_incorrect_set.append(res)
    
    logger.info(f"Majority-Correct Set: {len(majority_correct_set)} samples")
    logger.info(f"Majority-Incorrect Set: {len(majority_incorrect_set)} samples")
    
    # 9. Calculate Statistics
    def calculate_stats(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(samples)
        if total == 0:
            return {
                "total": 0,
                "sc_correct": 0,
                "dpc_correct": 0,
                "sc_accuracy": 0.0,
                "dpc_accuracy": 0.0,
                "dpc_gain": 0.0
            }
        
        sc_correct = sum(1 for s in samples if s["sc_correct"])
        dpc_correct = sum(1 for s in samples if s["dpc_correct"])
        
        sc_accuracy = (sc_correct / total) * 100
        dpc_accuracy = (dpc_correct / total) * 100
        dpc_gain = dpc_accuracy - sc_accuracy
        
        return {
            "total": total,
            "sc_correct": sc_correct,
            "dpc_correct": dpc_correct,
            "sc_accuracy": sc_accuracy,
            "dpc_accuracy": dpc_accuracy,
            "dpc_gain": dpc_gain
        }
    
    mc_stats = calculate_stats(majority_correct_set)
    mi_stats = calculate_stats(majority_incorrect_set)
    overall_stats = calculate_stats(list(filtered_results.values()))
    
    # 10. Prepare Output
    output = {
        "summary": {
            "total_samples": len(results),
            "filtered_samples": len(filtered_results),
            "excluded_samples": len(results) - len(filtered_results),
            "majority_correct_count": len(majority_correct_set),
            "majority_incorrect_count": len(majority_incorrect_set)
        },
        "majority_correct_set": {
            "statistics": mc_stats,
            "samples": [
                {
                    "qid": res["qid"],
                    "sc_correct": res["sc_correct"],
                    "dpc_correct": res["dpc_correct"]
                }
                for res in majority_correct_set
            ]
        },
        "majority_incorrect_set": {
            "statistics": mi_stats,
            "samples": [
                {
                    "qid": res["qid"],
                    "sc_correct": res["sc_correct"],
                    "dpc_correct": res["dpc_correct"]
                }
                for res in majority_incorrect_set
            ]
        },
        "overall": {
            "statistics": overall_stats
        }
    }
    
    # 11. Save Results
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
    
    # 12. Print Table
    logger.info("")
    logger.info("=" * 100)
    logger.info("Majority Analysis Results Table")
    logger.info("=" * 100)
    header = f"{'Set':<30} {'Total':<10} {'SC Correct':<12} {'SC Acc (%)':<12} " \
             f"{'DPC Correct':<12} {'DPC Acc (%)':<12} {'DPC Gain (%)':<12}"
    logger.info(header)
    logger.info("-" * 100)
    
    def format_row(set_name, stats):
        return (f"{set_name:<30} "
                f"{stats['total']:<10} "
                f"{stats['sc_correct']:<12} "
                f"{stats['sc_accuracy']:>10.2f}  "
                f"{stats['dpc_correct']:<12} "
                f"{stats['dpc_accuracy']:>10.2f}  "
                f"{stats['dpc_gain']:>+10.2f}  ")
    
    logger.info(format_row("Majority-Correct Set", mc_stats))
    logger.info(format_row("Majority-Incorrect Set", mi_stats))
    logger.info("-" * 100)
    logger.info(format_row("Overall", overall_stats))
    logger.info("=" * 100)
    
    logger.info(f"\nDetailed results saved to: {args.output_path}")

if __name__ == "__main__":
    main()

