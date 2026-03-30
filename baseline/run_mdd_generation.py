import os
import sys
import json
import logging
import argparse
import multiprocessing
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Optional

from tqdm import tqdm

# 1. IMMEDIATE LOGGING CONFIGURATION
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger("MDD-Generation")

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.common import (
    build_loader,
    init_worker_ignore_sigint,
    iter_dataset_with_candidates,
    load_candidate_map,
    save_json,
    save_json_atomic,
)
from dpc.llm.openai_llm import OpenAILLM
from dpc.agents.slicer_agent import SlicerAgent
from dpc.agents.tester_agent import TesterAgent
from dpc.agents.selector_agent import EquivalenceGrouperAgent
from dpc.utils.clustering import (
    cluster_sql_candidates,
    select_champion_and_challenger,
    select_champion_and_challenger_from_sql_groups,
)
from dpc.utils.schema_utils import SchemaExtractor

DISTINGUISH_FAILURE_KEYWORD = "Tester failed to generate distinguishing data after"


def init_worker() -> None:
    init_worker_ignore_sigint()


def process_sample(item_data: Dict[str, Any], candidate_sqls: List[str], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Generate MDD (distinguishing test data) for one sample:
    Phase1 selection -> Schema slicing -> Tester generation.
    """
    qid = str(item_data["question_id"])
    difficulty = item_data.get("difficulty", "unknown") or "unknown"

    try:
        tester_max_correction_attempts = (
            args.tester_max_correction_attempts
            if args.tester_max_correction_attempts is not None
            else args.max_correction_attempts
        )

        llm = OpenAILLM(
            model_name=args.model_name,
            api_key=args.api_key,
            base_url=args.base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
        )
        llm.reset_usage()

        slicer = SlicerAgent(llm)
        tester = TesterAgent(llm)
        grouper = EquivalenceGrouperAgent(llm)

        question = item_data["question"]
        db_path = item_data["db_path"]
        evidence = item_data.get("evidence")

        # Phase 1: Select champion/challenger
        full_schema = None
        if args.phase1_selection_mode == "execution":
            groups = cluster_sql_candidates(db_path, candidate_sqls, timeout=args.sql_timeout)
            champion_sql, challenger_sql = select_champion_and_challenger(groups)
        elif args.phase1_selection_mode == "llm_prompt":
            full_schema = SchemaExtractor.extract(db_path)
            grouping = grouper.run(
                question=question,
                candidate_sqls=candidate_sqls,
                full_schema=full_schema,
                evidence=evidence,
                max_correction_attempts=args.max_correction_attempts,
                num_grouping_attempts=args.num_grouping_attempts,
            )
            champion_sql, challenger_sql = select_champion_and_challenger_from_sql_groups(grouping["sql_groups"])
        else:
            raise ValueError(f"Unsupported phase1_selection_mode: {args.phase1_selection_mode}")

        if not champion_sql:
            return {
                "question_id": qid,
                "success": True,
                "difficulty": difficulty,
                "no_challenger": False,
                "no_champion": True,
                "result": {
                    "champion_sql": None,
                    "challenger_sql": None,
                    "test_data": None,
                    "token_usage": llm.get_usage(),
                    "status": "no_champion",
                },
                "error": "No valid champion SQL selected.",
            }
        if not challenger_sql:
            # Keep this as a dedicated status for dual-metric stats.
            return {
                "question_id": qid,
                "success": False,
                "difficulty": difficulty,
                "no_challenger": True,
                "no_champion": False,
                "result": {
                    "champion_sql": champion_sql,
                    "challenger_sql": None,
                    "test_data": None,
                    "token_usage": llm.get_usage(),
                    "status": "no_challenger",
                },
                "error": "No challenger SQL selected; cannot generate distinguishing test data.",
            }

        # Phase 2: Schema slicing
        if full_schema is None:
            full_schema = SchemaExtractor.extract(db_path)

        sliced_schema = slicer.run(
            candidate_sqls=[champion_sql, challenger_sql],
            full_schema=full_schema,
            max_correction_attempts=args.max_correction_attempts,
        )

        # Phase 3: MDD generation (Tester)
        test_data = tester.run(
            question=question,
            sql_1=champion_sql,
            sql_2=challenger_sql,
            sliced_schema=sliced_schema,
            evidence=evidence,
            max_correction_attempts=tester_max_correction_attempts,
        )

        return {
            "question_id": qid,
            "success": True,
            "difficulty": difficulty,
            "no_challenger": False,
            "no_champion": False,
            "result": {
                "champion_sql": champion_sql,
                "challenger_sql": challenger_sql,
                "test_data": test_data,
                "token_usage": llm.get_usage(),
            },
        }
    except Exception as e:
        return {
            "question_id": qid,
            "success": False,
            "difficulty": difficulty,
            "no_challenger": False,
            "no_champion": False,
            "error": str(e),
        }


def compute_category_stats(tasks: List[Dict[str, Any]], results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build mutually exclusive categories:
    1) no_champion
    2) no_challenger
    3) mdd_success
    4) mdd_distinguish_failure
    5) other_failure
    """
    categories = ["no_champion", "no_challenger", "mdd_success", "mdd_distinguish_failure", "other_failure"]

    total = len(tasks)
    overall_counts = {k: 0 for k in categories}
    by_difficulty: Dict[str, Dict[str, int]] = {}

    def classify(rec: Any) -> str:
        if isinstance(rec, dict) and bool(rec.get("no_champion")):
            return "no_champion"
        if isinstance(rec, dict) and bool(rec.get("no_challenger")):
            return "no_challenger"
        if isinstance(rec, dict) and rec.get("success") is True:
            return "mdd_success"
        err = str(rec.get("error", "")) if isinstance(rec, dict) else ""
        if DISTINGUISH_FAILURE_KEYWORD in err:
            return "mdd_distinguish_failure"
        return "other_failure"

    for task in tasks:
        qid = str(task["item_data"]["question_id"])
        difficulty = task["item_data"].get("difficulty", "unknown") or "unknown"

        if difficulty not in by_difficulty:
            by_difficulty[difficulty] = {"total": 0, **{k: 0 for k in categories}}
        by_difficulty[difficulty]["total"] += 1

        cls = classify(results.get(qid))
        overall_counts[cls] += 1
        by_difficulty[difficulty][cls] += 1

    overall = {"total": total}
    for k in categories:
        overall[k] = overall_counts[k]
    overall["mdd_success_rate"] = (overall_counts["mdd_success"] / total * 100.0) if total > 0 else 0.0

    breakdown: Dict[str, Any] = {}
    for diff, cnt in sorted(by_difficulty.items()):
        diff_total = cnt["total"]
        item = {"total": diff_total}
        for k in categories:
            item[k] = cnt[k]
        item["mdd_success_rate"] = (cnt["mdd_success"] / diff_total * 100.0) if diff_total > 0 else 0.0
        breakdown[diff] = item

    return {"overall": overall, "by_difficulty": breakdown}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MDD(TestData) only for DPC pipeline.")

    # Dataset arguments
    parser.add_argument("--dataset_type", type=str, default="bird", choices=["bird", "spider"], help="Dataset type")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset JSON file")
    parser.add_argument("--db_root_path", type=str, required=True, help="Directory containing databases")
    parser.add_argument("--pred_sqls_path", type=str, required=True, help="Path to predicted SQL candidates JSON")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for generated MDD results")
    parser.add_argument("--stats_output_path", type=str, default=None, help="Optional path for summary stats JSON")

    # LLM arguments
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="LLM model name")
    parser.add_argument("--api_key", type=str, default=None, help="LLM API Key")
    parser.add_argument("--base_url", type=str, default=None, help="LLM Base URL")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max tokens per response")
    parser.add_argument("--max_retries", type=int, default=3, help="Max retries for LLM calls")
    parser.add_argument("--retry_delay", type=int, default=2, help="Delay between retries")

    # Pipeline-lite arguments
    parser.add_argument("--sql_timeout", type=int, default=30, help="Timeout for SQL execution in selection")
    parser.add_argument("--max_correction_attempts", type=int, default=3, help="Max correction attempts")
    parser.add_argument(
        "--tester_max_correction_attempts",
        type=int,
        default=None,
        help="Override correction attempts only for TesterAgent (default: use --max_correction_attempts)",
    )
    parser.add_argument("--num_grouping_attempts", type=int, default=1, help="Grouping SC attempts for llm_prompt mode")
    parser.add_argument(
        "--phase1_selection_mode",
        type=str,
        default="execution",
        choices=["execution", "llm_prompt"],
        help="Phase 1 mode: execution clustering or llm prompt grouping",
    )
    parser.add_argument("--num_workers", type=int, default=8, help="Parallel workers")

    args = parser.parse_args()

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # Load dataset
    loader = build_loader(args.dataset_type, args.data_path, args.db_root_path)

    # Load candidates
    all_pred_sqls = load_candidate_map(args.pred_sqls_path)

    # Prepare tasks
    tasks = []
    for qid, item, candidate_sqls in iter_dataset_with_candidates(loader, all_pred_sqls):
        tasks.append(
            {
                "item_data": {
                    "question_id": qid,
                    "question": item.question,
                    "db_path": loader.get_db_path(item.db_id),
                    "evidence": item.evidence,
                    "difficulty": item.difficulty or "unknown",
                },
                "candidate_sqls": candidate_sqls,
            }
        )

    logger.info("Loaded %s samples with candidate SQLs.", len(tasks))

    # Resume
    output_path = args.output_path
    results: Dict[str, Any] = {}
    processed_ids = set()

    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                results = loaded
                processed_ids = set(results.keys())
                logger.info("Resuming from existing output. Processed: %s", len(processed_ids))
        except Exception as e:
            logger.warning("Failed to load existing output: %s", e)

    tasks_to_run = [t for t in tasks if str(t["item_data"]["question_id"]) not in processed_ids]
    if not tasks_to_run:
        logger.info("All samples already processed.")
    else:
        logger.info("Starting MDD generation for %s remaining samples...", len(tasks_to_run))
        num_workers = min(len(tasks_to_run), args.num_workers)
        executor = ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker)

        try:
            futures = {
                executor.submit(process_sample, t["item_data"], t["candidate_sqls"], args): t
                for t in tasks_to_run
            }
            for future in tqdm(as_completed(futures), total=len(tasks_to_run), desc="MDD Generation"):
                res = future.result()
                qid = str(res["question_id"])
                results[qid] = res

                if not res.get("success", False):
                    logger.error("Sample %s failed: %s", qid, res.get("error"))

                try:
                    save_json_atomic(output_path, results, indent=4)
                except Exception as e:
                    logger.error("Failed to save output for %s: %s", qid, e)
        except KeyboardInterrupt:
            logger.warning("Interrupted by user. Terminating workers...")
            executor.shutdown(wait=False, cancel_futures=True)
            try:
                os.killpg(0, signal.SIGKILL)
            except Exception:
                os._exit(1)
        finally:
            executor.shutdown(wait=True)

    # Stats (four-category breakdown)
    stats = compute_category_stats(tasks, results)
    overall = stats["overall"]
    logger.info(
        "Overall | total=%s | mdd_success=%s (%.2f%%) | no_champion=%s | no_challenger=%s | mdd_distinguish_failure=%s | other_failure=%s",
        overall["total"],
        overall["mdd_success"],
        overall["mdd_success_rate"],
        overall["no_champion"],
        overall["no_challenger"],
        overall["mdd_distinguish_failure"],
        overall["other_failure"],
    )
    logger.info("Difficulty breakdown:")
    for diff, s in stats["by_difficulty"].items():
        logger.info(
            "  - %s | total=%s | mdd_success=%s (%.2f%%) | no_champion=%s | no_challenger=%s | mdd_distinguish_failure=%s | other_failure=%s",
            diff,
            s["total"],
            s["mdd_success"],
            s["mdd_success_rate"],
            s["no_champion"],
            s["no_challenger"],
            s["mdd_distinguish_failure"],
            s["other_failure"],
        )

    logger.info("Full stats JSON:\n%s", json.dumps(stats, indent=2, ensure_ascii=False))

    stats_output_path = args.stats_output_path
    if not stats_output_path:
        if output_path.endswith(".json"):
            stats_output_path = output_path[:-5] + "_stats.json"
        else:
            stats_output_path = output_path + "_stats.json"

    try:
        save_json(stats_output_path, stats, indent=4)
        logger.info("Saved stats to %s", stats_output_path)
    except Exception as e:
        logger.error("Failed to save stats JSON: %s", e)


if __name__ == "__main__":
    main()
