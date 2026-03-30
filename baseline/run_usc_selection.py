import os
import sys
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple

from tqdm import tqdm

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


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger("USC-Selection")


def build_usc_prompt(
    schema_text: str,
    question: str,
    candidates: List[str],
    evidence: Optional[str] = None
) -> str:
    candidates_str = "\n\n".join([f"{i+1}. {sql}" for i, sql in enumerate(candidates)])
    evidence_str = f"\nExternal Knowledge: {evidence}" if evidence else ""

    return f"""You are a Text-to-SQL evaluator.

Given:
- Database schema
- Natural language question
- Candidate SQLs

Task:
1. Group SQL candidates by logical consistency (queries in the same group should represent the same intent and likely produce the same answer).
2. Rank groups by size descending (largest consensus group first). If tied, prefer the group more likely to match the question.
3. Choose one SQL from rank-1 group as the final SQL.

Database Schema:
{schema_text}

Question: {question}{evidence_str}

Candidate SQLs (1-based index):
{candidates_str}

Output format (strict JSON only):
{{
  "groups": [
    {{"rank": 1, "member_indices": [1, 3]}},
    {{"rank": 2, "member_indices": [2]}}
  ],
  "selected_index": 1
}}
"""


def parse_selected_sql(response_text: str, candidates: List[str]) -> Tuple[Optional[str], Optional[int]]:
    parsed = parse_json_response(response_text)
    selected_index = parsed.get("selected_index")

    if not isinstance(selected_index, int):
        raise ValueError("selected_index must be an integer.")
    if selected_index < 1 or selected_index > len(candidates):
        raise ValueError(f"selected_index out of range: {selected_index}")

    return candidates[selected_index - 1], selected_index


def process_sample_usc(task: Dict[str, Any], llm_config: Dict[str, Any], num_trials: int) -> Dict[str, Any]:
    qid = task["qid"]
    candidates = task["candidates"]
    question = task["question"]
    evidence = task["evidence"]
    schema_text = task["schema_text"]

    if not candidates:
        return {
            "qid": qid,
            "selected_sql": None,
            "selected_index": None,
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

    if len(candidates) == 1:
        return {
            "qid": qid,
            "selected_sql": candidates[0],
            "selected_index": 1,
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

    llm = OpenAILLM(**llm_config)
    llm.reset_usage()
    prompt = build_usc_prompt(schema_text, question, candidates, evidence)

    voted_indices = []
    for _ in range(max(1, num_trials)):
        try:
            res_text = llm.ask([{"role": "user", "content": prompt}])
            _, selected_index = parse_selected_sql(res_text, candidates)
            voted_indices.append(selected_index)
        except Exception:
            voted_indices.append(1)

    # Majority vote over selected indices
    counts: Dict[int, int] = {}
    for idx in voted_indices:
        counts[idx] = counts.get(idx, 0) + 1
    final_index = max(counts.items(), key=lambda kv: kv[1])[0]

    return {
        "qid": qid,
        "selected_sql": candidates[final_index - 1],
        "selected_index": final_index,
        "token_usage": llm.get_usage()
    }


def main():
    parser = argparse.ArgumentParser(
        description="Perform Unexecuted Self-Consistency (USC) selection on SQL candidates using prompt LLM only."
    )
    parser.add_argument("--pred_path", type=str, required=True, help="Path to predicted SQL candidates JSON file.")
    parser.add_argument("--dataset_type", type=str, choices=["spider", "bird"], required=True, help="Type of dataset.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset file.")
    parser.add_argument("--db_root_path", type=str, required=True, help="Path to database root directory.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save selected SQLs.")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="LLM model name.")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature.")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens per response.")
    parser.add_argument("--num_trials", type=int, default=1, help="Number of repeated LLM selections per sample.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers.")

    args = parser.parse_args()

    # 1) Load dataset
    loader = build_loader(args.dataset_type, args.data_path, args.db_root_path)

    # 2) Load candidates
    all_candidates = load_candidate_map(args.pred_path)

    # 3) Build tasks
    tasks = []
    for qid, item, candidates in tqdm(
        iter_dataset_with_candidates(loader, all_candidates),
        total=len(all_candidates),
        desc="Loading tasks",
    ):
        schema = loader.get_schema(item.db_id)
        schema_text = SchemaExtractor.to_readable_text(
            schema,
            include_stats=False,
            include_examples=False,
            include_descriptions=False
        )

        tasks.append(
            {
                "qid": qid,
                "candidates": candidates,
                "question": item.question,
                "evidence": item.evidence,
                "schema_text": schema_text,
            }
        )

    llm_config = build_llm_config(args)

    # 4) Parallel USC selection
    logger.info("Starting parallel USC selection for %s samples...", len(tasks))
    selected_sqls: Dict[str, Optional[str]] = {}

    usage_stats = UsageStats()

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(process_sample_usc, task, llm_config, args.num_trials): task for task in tasks
        }
        for future in tqdm(as_completed(futures), total=len(tasks), desc="USC Selection"):
            res = future.result()
            selected_sqls[res["qid"]] = res["selected_sql"]

            usage = res.get("token_usage", {})
            usage_stats.update(usage)

    # 5) Save
    save_json(args.output_path, selected_sqls, indent=4)

    logger.info("USC selection complete. Results saved to %s", args.output_path)
    usage_stats.log_average(logger, label="Statistics")


if __name__ == "__main__":
    main()
