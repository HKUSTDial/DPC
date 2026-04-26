import os
import sys
import argparse
import logging
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.common import (
    UsageStats,
    build_llm_config,
    build_loader,
    iter_dataset_with_candidates,
    load_candidate_map,
    save_json,
)
from dpc.llm.openai_llm import OpenAILLM
from dpc.utils.schema_utils import SchemaExtractor
from dpc.utils.response_parser import parse_json_response
from dpc.utils.db_utils import execute_sql as execute_sql_rows

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Suppress noisy HTTP logs from libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger("MCS-Selection")

def execute_sql_with_time(sql: str, db_path: str, timeout: int = 30) -> Tuple[Optional[frozenset], float]:
    """Executes SQL and returns the result set and execution time."""
    if not sql:
        return None, 0.0
    start_time = time.time()
    try:
        results = execute_sql_rows(db_path, sql, timeout=timeout)
        execution_time = time.time() - start_time
        return frozenset(tuple(row) for row in results), execution_time
    except Exception:
        return None, 0.0

def get_mcs_prompt(schema_text: str, question: str, candidates: List[str], evidence: Optional[str] = None) -> str:
    """Constructs the MCQ prompt for the LLM."""
    candidates_str = "\n".join([f"{i+1}. {sql}" for i, sql in enumerate(candidates)])
    evidence_str = f"\n### External Knowledge:\n{evidence}" if evidence else ""
    
    prompt = f"""### For a given DB schema and question, select the most accurate query among the candidate SQL queries.
### DB schema:
{schema_text}

### Question:
{question}{evidence_str}

### Candidate SQLs:
{candidates_str}

### Checklist:
1. The SQL should accurately represent the question.
2. The SQL should accurately use the given knowledge evidence.
3. The SELECT clause should not include any additional columns that are not included in the question.
4. The order of columns in the SELECT clause must be the same as the order in the question.
5. Check if the operations are being performed correctly according to the column type.

### Instruction:
- If the first SQL satisfies all the conditions of the checklist, please choose the first SQL. If not, move on to the next SQL.
- If there’s no SQL that satisfies all the requirements on the checklist, just choose the first SQL.
- Provide a detailed step-by-step explanation following the order of the checklist when checking whether each SQL satisfies the checklist.
- Your answer should strictly follow the following json format.
{{
  "reasoning": "The reasoning steps for choosing the best SQL", 
  "sql": "The final chosen SQL."
}}

### Your Answer:"""
    return prompt

def process_sample_mcs(task: Dict[str, Any], llm_config: Dict[str, Any]) -> Dict[str, Any]:
    """Worker function to perform MCS selection for a single sample."""
    qid = task["qid"]
    candidates = task["candidates"]
    db_path = task["db_path"]
    sql_timeout = task["sql_timeout"]
    question = task["question"]
    evidence = task["evidence"]
    schema_text = task["schema_text"]
    
    if not candidates:
        return {
            "qid": qid, 
            "selected_sql": None, 
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

    # 1 & 2. Clustering
    result_groups = defaultdict(list) # result -> list of sqls
    
    for sql in candidates:
        res, _ = execute_sql_with_time(sql, db_path, timeout=sql_timeout)
        if res is not None:
            result_groups[res].append(sql)
            
    filtered_candidates = []
    for res, group in result_groups.items():
        # Keep one representative SQL per result group (e.g., the first one)
        filtered_candidates.append(group[0])
            
    if not filtered_candidates:
        # Fallback to the first candidate if all filtered or failed
        return {
            "qid": qid, 
            "selected_sql": candidates[0],
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
    
    if len(filtered_candidates) == 1:
        return {
            "qid": qid, 
            "selected_sql": filtered_candidates[0],
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

    # 3. LLM MCQ with 1 trial
    llm = OpenAILLM(**llm_config)
    llm.reset_usage()
    prompt = get_mcs_prompt(schema_text, question, filtered_candidates, evidence)
    
    responses = []
    for _ in range(5):
        try:
            res_text = llm.ask([{"role": "user", "content": prompt}])
            parsed = parse_json_response(res_text)
            responses.append(parsed.get("sql", filtered_candidates[0]))
        except Exception:
            responses.append(filtered_candidates[0])
            
    # Majority vote on the trials
    if not responses:
        selected_sql = filtered_candidates[0]
    else:
        counts = Counter(responses)
        selected_sql = counts.most_common(1)[0][0]
    
    # Ensure the selected SQL is actually one of the candidates (safety check)
    if selected_sql not in filtered_candidates:
        # If model hallucinated a new SQL, pick the most common valid one or the first filtered one
        valid_responses = [r for r in responses if r in filtered_candidates]
        if valid_responses:
            selected_sql = Counter(valid_responses).most_common(1)[0][0]
        else:
            selected_sql = filtered_candidates[0]

    return {
        "qid": qid, 
        "selected_sql": selected_sql,
        "token_usage": llm.get_usage()
    }

def main():
    parser = argparse.ArgumentParser(description="Perform Multi-Choice Selection (MCS) on SQL candidates.")
    parser.add_argument("--pred_path", type=str, required=True, help="Path to predicted SQL candidates JSON file.")
    parser.add_argument("--dataset_type", type=str, choices=["spider", "bird"], required=True, help="Type of dataset.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset file.")
    parser.add_argument("--db_root_path", type=str, required=True, help="Path to database root directory.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the selected SQLs.")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="LLM model name.")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens per response")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout for SQL execution.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel samples to process.")
    
    args = parser.parse_args()

    # 1. Load Dataset
    loader = build_loader(args.dataset_type, args.data_path, args.db_root_path)

    # 2. Load Candidates
    all_candidates = load_candidate_map(args.pred_path)

    # 3. Prepare Tasks
    tasks = []
    for qid, item, candidates in tqdm(
        iter_dataset_with_candidates(loader, all_candidates),
        total=len(all_candidates),
        desc="Loading tasks",
    ):
        db_path = loader.get_db_path(item.db_id)
        schema = loader.get_schema(item.db_id)
        schema_text = SchemaExtractor.to_readable_text(schema, include_stats=False, include_examples=False, include_descriptions=False)
        
        tasks.append({
            "qid": qid,
            "candidates": candidates,
            "db_path": db_path,
            "sql_timeout": args.timeout,
            "question": item.question,
            "evidence": item.evidence,
            "schema_text": schema_text
        })

    # 4. LLM Config
    llm_config = build_llm_config(args)

    # 5. Parallel Processing
    selected_sqls = {}
    logger.info(f"Starting parallel MCS selection for {len(tasks)} samples...")
    
    usage_stats = UsageStats()

    # Note: Using ThreadPoolExecutor because LLM calls are I/O bound
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_sample_mcs, task, llm_config): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="MCS Selection"):
            res = future.result()
            selected_sqls[res["qid"]] = res["selected_sql"]
            
            # Aggregate stats
            usage = res.get("token_usage", {})
            usage_stats.update(usage)

    # 6. Save Results
    save_json(args.output_path, selected_sqls, indent=4)
    
    logger.info(f"MCS selection complete. Results saved to {args.output_path}")
    usage_stats.log_average(logger, label="Statistics")

if __name__ == "__main__":
    main()
