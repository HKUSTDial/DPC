import os
import sys
import json
import argparse
import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Dict, Any, Optional

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpc.datasets.spider_loader import SpiderLoader
from dpc.datasets.bird_loader import BirdLoader
from dpc.llm.openai_llm import OpenAILLM
from dpc.utils.schema_utils import SchemaExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# Suppress noisy HTTP logs from libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger("MCS-Selection-WO-Execution")


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


def process_sample_mcs_wo_execution(task: Dict[str, Any], llm_config: Dict[str, Any]) -> Dict[str, Any]:
    """Worker function to perform MCS selection for a single sample without execution-based clustering."""
    qid = task["qid"]
    candidates = task["candidates"]
    question = task["question"]
    evidence = task["evidence"]
    schema_text = task["schema_text"]

    if not candidates:
        return {
            "qid": qid,
            "selected_sql": None,
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    if len(candidates) == 1:
        return {
            "qid": qid,
            "selected_sql": candidates[0],
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    # Directly use all candidates in prompt (no execution clustering).
    llm = OpenAILLM(**llm_config)
    llm.reset_usage()
    prompt = get_mcs_prompt(schema_text, question, candidates, evidence)

    responses = []
    for _ in range(1):
        try:
            res_text = llm.ask([{"role": "user", "content": prompt}])
            json_match = re.search(r"\{.*\}", res_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                responses.append(parsed.get("sql", candidates[0]))
            else:
                responses.append(candidates[0])
        except Exception:
            responses.append(candidates[0])

    if not responses:
        selected_sql = candidates[0]
    else:
        counts = Counter(responses)
        selected_sql = counts.most_common(1)[0][0]

    return {
        "qid": qid,
        "selected_sql": selected_sql,
        "token_usage": llm.get_usage(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Perform Multi-Choice Selection (MCS) on SQL candidates without execution-based clustering."
    )
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
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel samples to process.")

    args = parser.parse_args()

    # 1. Load Dataset
    if args.dataset_type.lower() == "spider":
        loader = SpiderLoader(args.data_path, args.db_root_path)
    elif args.dataset_type.lower() == "bird":
        loader = BirdLoader(args.data_path, args.db_root_path)
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    # 2. Load Candidates
    with open(args.pred_path, "r", encoding="utf-8") as f:
        all_candidates = json.load(f)
    all_candidates = {str(k): v for k, v in all_candidates.items()}

    # 3. Prepare Tasks
    tasks = []
    for i in tqdm(range(len(loader)), desc="Loading tasks"):
        item = loader.get_item(i)
        qid = str(item.question_id)

        if qid not in all_candidates:
            continue

        schema = loader.get_schema(item.db_id)
        schema_text = SchemaExtractor.to_readable_text(
            schema, include_stats=False, include_examples=False, include_descriptions=False
        )

        tasks.append(
            {
                "qid": qid,
                "candidates": all_candidates[qid],
                "question": item.question,
                "evidence": item.evidence,
                "schema_text": schema_text,
            }
        )

    # 4. LLM Config
    llm_config = {
        "model_name": args.model_name,
        "api_key": args.api_key,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    # 5. Parallel Processing
    selected_sqls = {}
    logger.info("Starting parallel MCS selection (wo execution) for %s samples...", len(tasks))

    total_prompt_tokens = 0
    total_completion_tokens = 0
    count = 0

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_sample_mcs_wo_execution, task, llm_config): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="MCS Selection WO Execution"):
            res = future.result()
            selected_sqls[res["qid"]] = res["selected_sql"]

            usage = res.get("token_usage", {})
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)
            count += 1

    # 6. Save Results
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(selected_sqls, f, indent=4, ensure_ascii=False)

    logger.info("MCS selection (wo execution) complete. Results saved to %s", args.output_path)

    if count > 0:
        avg_prompt = total_prompt_tokens / count
        avg_completion = total_completion_tokens / count
        avg_total = (total_prompt_tokens + total_completion_tokens) / count
        logger.info("--- Statistics (Average per question) ---")
        logger.info("Prompt Tokens: %.1f", avg_prompt)
        logger.info("Completion Tokens: %.1f", avg_completion)
        logger.info("Total Tokens: %.1f", avg_total)
        logger.info("----------------------------------------")


if __name__ == "__main__":
    main()

