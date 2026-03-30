import os
import sys
import json
import logging
import argparse
import multiprocessing
import signal
from typing import List, Dict, Any

# 1. IMMEDIATE LOGGING CONFIGURATION (Must be at the very top)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True 
)
# Suppress noisy HTTP logs from libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger("DPC-Batch")

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 2. NOW IMPORT OTHER MODULES
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from dpc.llm.openai_llm import OpenAILLM
from dpc.agents.slicer_agent import SlicerAgent
from dpc.agents.tester_agent import TesterAgent
from dpc.agents.solver_agent import PythonSolverAgent
from dpc.agents.selector_agent import EquivalenceGrouperAgent
from dpc.core.pipeline import DPCPipeline
from dpc.datasets.spider_loader import SpiderLoader
from dpc.datasets.bird_loader import BirdLoader

def init_worker():
    """
    Initialize worker process to ignore SIGINT (Ctrl+C).
    This allows the main process to handle the interrupt and clean up.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def process_sample(item_data: Dict[str, Any], candidate_sqls: List[str], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Worker function to process a single Text-to-SQL sample.
    Initializes its own LLM and Agents to ensure process isolation.
    """
    try:
        # 1. Initialize LLM inside the process
        llm = OpenAILLM(
            model_name=args.model_name,
            api_key=args.api_key,
            base_url=args.base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay
        )

        # 2. Initialize Agents
        slicer = SlicerAgent(llm)
        tester = TesterAgent(llm)
        solver = PythonSolverAgent(llm)
        grouper = EquivalenceGrouperAgent(llm)
        pipeline = DPCPipeline(slicer=slicer, tester=tester, solver=solver, grouper=grouper)

        # 3. Run Pipeline
        result = pipeline.run(
            question=item_data["question"],
            db_path=item_data["db_path"],
            candidate_sqls=candidate_sqls,
            evidence=item_data.get("evidence"),
            sql_timeout=args.sql_timeout,
            python_timeout=args.python_timeout,
            epsilon=args.epsilon,
            max_correction_attempts=args.max_correction_attempts,
            num_test_data=args.num_test_data,
            num_solver_attempts=args.num_solver_attempts,
            num_grouping_attempts=args.num_grouping_attempts,
            phase1_selection_mode=args.phase1_selection_mode,
            eval_metric=args.eval_metric
        )

        return {
            "question_id": item_data["question_id"],
            "success": True,
            "result": result
        }
    except Exception as e:
        return {
            "question_id": item_data["question_id"],
            "success": False,
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="Run DPC-SQL Pipeline")
    
    # Dataset Arguments
    parser.add_argument("--dataset_type", type=str, default="bird", choices=["bird", "spider"], help="Dataset type")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset JSON file")
    parser.add_argument("--db_root_path", type=str, required=True, help="Directory containing databases")
    parser.add_argument("--pred_sqls_path", type=str, required=True, help="Path to predicted SQL candidates JSON")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for final results")
    
    # LLM Arguments
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="LLM model name")
    parser.add_argument("--api_key", type=str, default=None, help="LLM API Key")
    parser.add_argument("--base_url", type=str, default=None, help="LLM Base URL")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens per response")
    parser.add_argument("--max_retries", type=int, default=3, help="Max retries for LLM calls")
    parser.add_argument("--retry_delay", type=int, default=2, help="Delay between retries")
    
    # Pipeline Arguments
    parser.add_argument("--sql_timeout", type=int, default=30, help="Timeout for SQL execution")
    parser.add_argument("--python_timeout", type=int, default=30, help="Timeout for Python solver execution")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Epsilon for numeric comparison")
    parser.add_argument("--max_correction_attempts", type=int, default=3, help="Max attempts to correct Python solver")
    parser.add_argument("--num_test_data", type=int, default=1, help="Number of test data sets to generate")
    parser.add_argument("--num_solver_attempts", type=int, default=1, help="Number of solver attempts per test data")
    parser.add_argument("--num_grouping_attempts", type=int, default=1, help="Number of grouping attempts for phase1 llm_prompt SC merging")
    parser.add_argument(
        "--eval_metric",
        type=str,
        default="bs_f1",
        choices=["bs_f1", "ex"],
        help="Verification metric for SQL-vs-proxy scoring: bs_f1 (default) or ex",
    )
    parser.add_argument(
        "--phase1_selection_mode",
        type=str,
        default="execution",
        choices=["execution", "llm_prompt"],
        help="Phase 1 selection mode: execution clustering or LLM prompt-based selection"
    )
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel workers")

    args = parser.parse_args()

    # Initialization
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    # Load Dataset
    if args.dataset_type.lower() == "spider":
        loader = SpiderLoader(args.data_path, args.db_root_path)
    elif args.dataset_type.lower() == "bird":
        loader = BirdLoader(args.data_path, args.db_root_path)
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    # Load Predicted SQLs
    if not os.path.exists(args.pred_sqls_path):
        raise FileNotFoundError(f"Predicted SQLs file not found: {args.pred_sqls_path}")
    
    with open(args.pred_sqls_path, 'r', encoding='utf-8') as f:
        all_pred_sqls = json.load(f)

    # Prepare Tasks
    tasks = []
    for i in range(len(loader)):
        item = loader.get_item(i)
        if str(item.question_id) in all_pred_sqls:
            tasks.append({
                "item_data": {
                    "question_id": item.question_id,
                    "question": item.question,
                    "db_path": loader.get_db_path(item.db_id),
                    "evidence": item.evidence
                },
                "candidate_sqls": all_pred_sqls[str(item.question_id)]
            })

    logger.info(f"Loaded {len(tasks)} samples from dataset.")

    # Check for Resume Logic
    output_path = args.output_path
    results = {}
    processed_ids = set()
    
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
                if isinstance(results, dict):
                    processed_ids = {str(qid) for qid in results.keys()}
                    logger.info(f"Resuming from existing results. {len(processed_ids)} samples already processed.")
                else:
                    results = {}
        except Exception as e:
            logger.warning(f"Could not load existing results for resume: {e}. Starting fresh.")

    # Filter out tasks that are already processed
    tasks_to_run = [t for t in tasks if str(t["item_data"]["question_id"]) not in processed_ids]
    
    if not tasks_to_run:
        logger.info("All samples have already been processed. Nothing to do.")
        return

    logger.info(f"Starting parallel processing for {len(tasks_to_run)} remaining samples...")

    # Execute in Parallel with Real-time Saving
    num_workers = min(len(tasks_to_run), args.num_workers) 
    
    executor = ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker)
    
    total_prompt_tokens = 0
    total_completion_tokens = 0
    count = 0

    try:
        futures = {executor.submit(process_sample, t["item_data"], t["candidate_sqls"], args): t for t in tasks_to_run}
        
        for future in tqdm(as_completed(futures), total=len(tasks_to_run), desc="DPC Processing"):
            res = future.result()
            qid = str(res["question_id"])
            if res["success"]:
                # Save only the SQL string to JSON for compatibility
                results[qid] = res["result"]["selected_sql"]
                
                # Aggregate stats for console output
                usage = res["result"].get("token_usage", {})
                total_prompt_tokens += usage.get("prompt_tokens", 0)
                total_completion_tokens += usage.get("completion_tokens", 0)
                count += 1
            else:
                results[qid] = None
                logger.error(f"Sample {qid} failed: {res.get('error')}")
            
            # Real-time saving to disk
            try:
                if os.path.dirname(output_path):
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)
            except Exception as e:
                logger.error(f"Failed to save real-time results for ID {qid}: {e}")
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user. Terminating all processes forcefully...")
        executor.shutdown(wait=False, cancel_futures=True)
        try:
            os.killpg(0, signal.SIGKILL)
        except Exception:
            os._exit(1)
    finally:
        executor.shutdown(wait=True)

    # Summary
    success_count = sum(1 for sql in results.values() if sql is not None)
    logger.info(f"Batch processing completed. Total processed: {len(results)}. Success (non-null): {success_count}/{len(results)}")
    
    if count > 0:
        avg_prompt = total_prompt_tokens / count
        avg_completion = total_completion_tokens / count
        avg_total = (total_prompt_tokens + total_completion_tokens) / count
        logger.info(f"--- Statistics (Average per question in this run) ---")
        logger.info(f"Prompt Tokens: {avg_prompt:.1f}")
        logger.info(f"Completion Tokens: {avg_completion:.1f}")
        logger.info(f"Total Tokens: {avg_total:.1f}")
        logger.info(f"---------------------------------------------------")

    logger.info(f"All results are permanently saved to {output_path}")

if __name__ == "__main__":
    main()
