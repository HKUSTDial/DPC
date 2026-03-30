import os
import json
import argparse
import logging
from collections import Counter
from typing import Dict, Any


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MDD-Posthoc-Stats")


DISTINGUISH_FAILURE_KEYWORD = "Tester failed to generate distinguishing data after"


def classify_item(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return "other_failure"

    status = str(item.get("result", {}).get("status", ""))
    if item.get("no_champion") is True or status == "no_champion":
        return "no_champion"
    if item.get("no_challenger") is True or status == "no_challenger":
        return "no_challenger"
    if item.get("success") is True:
        return "mdd_success"

    error = str(item.get("error", ""))
    if DISTINGUISH_FAILURE_KEYWORD in error:
        return "mdd_distinguish_failure"
    return "other_failure"


def compute_stats(results: Dict[str, Any]) -> Dict[str, Any]:
    categories = [
        "no_champion",
        "no_challenger",
        "mdd_success",
        "mdd_distinguish_failure",
        "other_failure",
    ]

    overall_counts = {c: 0 for c in categories}
    by_difficulty: Dict[str, Dict[str, int]] = {}
    other_failure_errors = Counter()

    for _, item in results.items():
        difficulty = "unknown"
        if isinstance(item, dict):
            difficulty = str(item.get("difficulty", "unknown") or "unknown")

        cls = classify_item(item)
        overall_counts[cls] += 1

        if difficulty not in by_difficulty:
            by_difficulty[difficulty] = {"total": 0, **{c: 0 for c in categories}}
        by_difficulty[difficulty]["total"] += 1
        by_difficulty[difficulty][cls] += 1

        if cls == "other_failure":
            err = str(item.get("error", "")) if isinstance(item, dict) else "non-dict result item"
            other_failure_errors[err] += 1

    total = len(results)
    overall = {"total": total}
    for c in categories:
        overall[c] = overall_counts[c]
        overall[f"{c}_rate"] = (overall_counts[c] / total * 100.0) if total > 0 else 0.0

    by_difficulty_rates: Dict[str, Any] = {}
    for diff, cnt in sorted(by_difficulty.items()):
        dt = cnt["total"]
        row = {"total": dt}
        for c in categories:
            row[c] = cnt[c]
            row[f"{c}_rate"] = (cnt[c] / dt * 100.0) if dt > 0 else 0.0
        by_difficulty_rates[diff] = row

    return {
        "definition": {
            "mdd_distinguish_failure": f"error contains: '{DISTINGUISH_FAILURE_KEYWORD}'",
            "other_failure": "failed cases excluding mdd_distinguish_failure",
        },
        "overall": overall,
        "by_difficulty": by_difficulty_rates,
        "top_other_failure_errors": other_failure_errors.most_common(20),
    }


def print_stats(stats: Dict[str, Any]) -> None:
    o = stats["overall"]
    logger.info(
        "Overall | total=%s | no_champion=%s (%.2f%%) | no_challenger=%s (%.2f%%) | "
        "mdd_success=%s (%.2f%%) | mdd_distinguish_failure=%s (%.2f%%) | other_failure=%s (%.2f%%)",
        o["total"],
        o["no_champion"],
        o["no_champion_rate"],
        o["no_challenger"],
        o["no_challenger_rate"],
        o["mdd_success"],
        o["mdd_success_rate"],
        o["mdd_distinguish_failure"],
        o["mdd_distinguish_failure_rate"],
        o["other_failure"],
        o["other_failure_rate"],
    )

    logger.info("Difficulty breakdown:")
    for diff, s in stats["by_difficulty"].items():
        logger.info(
            "  - %s | total=%s | no_champion=%s | no_challenger=%s | mdd_success=%s | "
            "mdd_distinguish_failure=%s | other_failure=%s",
            diff,
            s["total"],
            s["no_champion"],
            s["no_challenger"],
            s["mdd_success"],
            s["mdd_distinguish_failure"],
            s["other_failure"],
        )

    if stats["top_other_failure_errors"]:
        logger.info("Top other_failure error messages:")
        for msg, cnt in stats["top_other_failure_errors"]:
            logger.info("  - (%s) %s", cnt, msg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc stats for MDD generation results.")
    parser.add_argument("--mdd_result_path", type=str, required=True, help="Path to MDD result JSON.")
    parser.add_argument("--output_path", type=str, default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    if not os.path.exists(args.mdd_result_path):
        raise FileNotFoundError(f"MDD result file not found: {args.mdd_result_path}")

    with open(args.mdd_result_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    if not isinstance(results, dict):
        raise ValueError("Expected top-level dict: {qid: result_item}")

    stats = compute_stats(results)
    print_stats(stats)
    logger.info("Full stats JSON:\n%s", json.dumps(stats, indent=2, ensure_ascii=False))

    output_path = args.output_path
    if not output_path:
        if args.mdd_result_path.endswith(".json"):
            output_path = args.mdd_result_path[:-5] + "_posthoc_stats.json"
        else:
            output_path = args.mdd_result_path + "_posthoc_stats.json"

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)
    logger.info("Saved post-hoc stats to %s", output_path)


if __name__ == "__main__":
    main()

