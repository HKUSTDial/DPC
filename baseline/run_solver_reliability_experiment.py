import os
import sys
import json
import re
import math
import argparse
import logging
import signal
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

import pandas as pd
from tqdm import tqdm

# Logging first
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logger = logging.getLogger("Solver-Reliability-Exp")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpc.llm.openai_llm import OpenAILLM
from dpc.agents.slicer_agent import SlicerAgent
from dpc.agents.tester_agent import TesterAgent
from dpc.agents.selector_agent import EquivalenceGrouperAgent
from dpc.prompts.factory import PromptFactory
from dpc.utils.python_executor import PythonExecutor
from dpc.utils.clustering import (
    cluster_sql_candidates,
    select_champion_and_challenger,
    select_champion_and_challenger_from_sql_groups,
)
from dpc.utils.schema_utils import SchemaExtractor
from dpc.datasets.spider_loader import SpiderLoader
from dpc.datasets.bird_loader import BirdLoader
from dpc.eval.metrics import DPCEvaluator, normalize_result
from dpc.utils.db_utils import execute_sql


ERROR_TAXONOMY = """1. Schema Related
- 1.1 Column/Table Selection & Mapping
- 1.2 Join Path Error

2. Value Related
- 2.1 Value Mapping Error
- 2.2 Data Format/Casting

3. Function Related
- 3.1 Aggregation Error
- 3.2 Calculation & Processing

4. Logic Related
- 4.1 Filtering & Set Logic
- 4.2 Presentation Logic
"""


def init_worker() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _extract_result_block(text: str) -> str:
    start_tag = "<result>"
    end_tag = "</result>"
    idx = text.rfind(start_tag)
    if idx == -1:
        return text.strip()
    content = text[idx + len(start_tag):]
    end_idx = content.find(end_tag)
    if end_idx != -1:
        return content[:end_idx].strip()
    return content.strip()


def _extract_json_obj(text: str) -> Dict[str, Any]:
    body = _extract_result_block(text)
    body = re.sub(r"^```json\s*", "", body.strip())
    body = re.sub(r"^```\s*", "", body.strip())
    body = re.sub(r"\s*```$", "", body.strip())
    m = re.search(r"\{[\s\S]*\}", body)
    if not m:
        raise ValueError("No JSON object found in LLM response.")
    return json.loads(m.group(0))


def _safe_jsonable_result(x: Any) -> Any:
    if isinstance(x, pd.DataFrame):
        return {
            "columns": [str(c) for c in x.columns.tolist()],
            "rows": x.astype(object).where(pd.notnull(x), None).values.tolist(),
        }
    if isinstance(x, list):
        return x
    if x is None:
        return None
    if isinstance(x, (int, float, str, bool, dict)):
        return x
    return str(x)


def _execute_sql_on_data(sql: str, test_data: Dict[str, List[Dict[str, Any]]], timeout: int = 30) -> Any:
    import sqlite3
    import threading

    conn = sqlite3.connect(":memory:")
    timer = threading.Timer(timeout, conn.interrupt)
    timer.start()
    try:
        for table_name, rows in test_data.items():
            pd.DataFrame(rows).to_sql(table_name, conn, index=False)
        return pd.read_sql_query(sql, conn)
    finally:
        timer.cancel()
        conn.close()


def _strict_ex(pred: Any, gold: Any) -> float:
    # Strict set-equality on raw execution rows (no value normalization).
    def to_row_set(x: Any) -> set:
        if x is None:
            return set()
        if isinstance(x, pd.DataFrame):
            rows = x.values.tolist()
        elif isinstance(x, list):
            rows = x
        else:
            rows = [[x]]

        out = set()
        for row in rows:
            if isinstance(row, (list, tuple)):
                out.add(tuple(row))
            else:
                out.add((row,))
        return out

    p = to_row_set(pred)
    g = to_row_set(gold)
    return 1.0 if p == g else 0.0


def _safe_ex_against_gold_on_db(db_path: str, pred_sql: str, gold_rows: Any, timeout: int) -> float:
    try:
        pred_rows = execute_sql(db_path, pred_sql, timeout=timeout)
        return _strict_ex(pred_rows, gold_rows)
    except Exception:
        return 0.0


def _pearson_corr(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def _rankdata(vals: List[float]) -> List[float]:
    pairs = sorted((v, i) for i, v in enumerate(vals))
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[pairs[k][1]] = avg_rank
        i = j + 1
    return ranks


def _spearman_corr(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    rx = _rankdata(xs)
    ry = _rankdata(ys)
    return _pearson_corr(rx, ry)


def _run_solver_with_trace(
    llm: OpenAILLM,
    question: str,
    test_data: Dict[str, List[Dict[str, Any]]],
    sliced_schema: Dict[str, Any],
    evidence: Optional[str],
    max_correction_attempts: int,
    python_timeout: int,
) -> Dict[str, Any]:
    messages = PromptFactory.get_solver_prompt(
        question=question,
        sliced_schema=sliced_schema,
        test_data=test_data,
        evidence=evidence,
    )
    for attempt in range(max_correction_attempts + 1):
        try:
            response_text = llm.ask(messages)
            messages.append({"role": "assistant", "content": response_text})
            code = _extract_result_block(response_text)
            exec_result = PythonExecutor.execute(test_data, code, timeout=python_timeout)
            if isinstance(exec_result, str) and ("Traceback" in exec_result or exec_result.startswith("Error:")):
                if attempt < max_correction_attempts:
                    messages.extend(PromptFactory.get_solver_retry_prompt(exec_result))
                    continue
                raise ValueError(f"Solver execution failed: {exec_result}")
            return {"success": True, "python_script": code, "python_answer": exec_result}
        except Exception as e:
            if attempt < max_correction_attempts:
                messages.extend(PromptFactory.get_solver_retry_prompt(str(e)))
            else:
                return {"success": False, "error": str(e), "python_script": None, "python_answer": None}
    return {"success": False, "error": "Unexpected solver loop exit", "python_script": None, "python_answer": None}


def _vote_on_solver_samples(samples: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Vote on multiple solver outputs using the same criterion as pipeline:
    treat two answers as equivalent if Soft-F1 > 0.99.
    """
    if not samples:
        raise ValueError("No successful solver samples to vote on.")
    if len(samples) == 1:
        return samples[0], {
            "groups": 1,
            "winning_group_votes": 1,
            "total_successful_attempts": 1,
        }

    groups: List[Dict[str, Any]] = []
    for sample in samples:
        placed = False
        for g in groups:
            score = DPCEvaluator.evaluate(sample["python_answer"], g["representative_answer"])
            if score > 0.99:
                g["count"] += 1
                g["members"].append(sample)
                placed = True
                break
        if not placed:
            groups.append(
                {
                    "representative_answer": sample["python_answer"],
                    "members": [sample],
                    "count": 1,
                }
            )

    groups.sort(key=lambda x: x["count"], reverse=True)
    winner_group = groups[0]
    winner_sample = winner_group["members"][0]
    vote_meta = {
        "groups": len(groups),
        "winning_group_votes": winner_group["count"],
        "total_successful_attempts": len(samples),
    }
    return winner_sample, vote_meta


def _judge_python_correctness(
    llm: OpenAILLM,
    question: str,
    evidence: Optional[str],
    gold_schema_text: str,
    gold_sql: str,
    gold_answer: Any,
    python_script: str,
    python_answer: Any,
) -> Dict[str, Any]:
    prompt = f"""You are evaluating whether a Python/Pandas solution is logically correct for a Text-to-SQL task.

Inputs:
- Question: {question}
- Evidence: {evidence or ""}
- Sliced Schema (from gold SQL):
{gold_schema_text}
- Gold SQL:
{gold_sql}
- Gold Answer:
{json.dumps(_safe_jsonable_result(gold_answer), ensure_ascii=False)}
- Python Script:
{python_script}
- Python Answer:
{json.dumps(_safe_jsonable_result(python_answer), ensure_ascii=False)}

Decision rule:
- Mark correct if Python solution is logically equivalent to Gold SQL for answering the user question.
- Do NOT require strict row order or column order.
- Focus on semantic correctness and whether the answer addresses the question.

Return strict JSON:
{{
  "reason": "your reasoning process",
  "is_correct": true/false
}}
"""
    parsed = _extract_json_obj(llm.ask([{"role": "user", "content": prompt}]))
    return {
        "reason": str(parsed.get("reason", "")),
        "is_correct": bool(parsed.get("is_correct", False)),
    }


def _judge_champion_equivalence(
    llm: OpenAILLM,
    question: str,
    evidence: Optional[str],
    gold_schema_text: str,
    champion_sql: str,
    champion_answer: Any,
    gold_sql: str,
    gold_answer: Any,
) -> Dict[str, Any]:
    prompt = f"""You are evaluating whether champion SQL is logically equivalent to gold SQL for answering a question.

Inputs:
- Question: {question}
- Evidence: {evidence or ""}
- Sliced Schema (from gold SQL):
{gold_schema_text}
- Champion SQL:
{champion_sql}
- Champion SQL Answer:
{json.dumps(_safe_jsonable_result(champion_answer), ensure_ascii=False)}
- Gold SQL:
{gold_sql}
- Gold SQL Answer:
{json.dumps(_safe_jsonable_result(gold_answer), ensure_ascii=False)}

Decision rule:
- Mark equivalent if champion SQL is logically consistent with gold SQL for this task.
- Do NOT require strict row/column order equality.
- Focus on semantic equivalence and whether the answer addresses the user question.

Return strict JSON:
{{
  "reason": "your reasoning process",
  "is_equivalent": true/false
}}
"""
    parsed = _extract_json_obj(llm.ask([{"role": "user", "content": prompt}]))
    return {
        "reason": str(parsed.get("reason", "")),
        "is_equivalent": bool(parsed.get("is_equivalent", False)),
    }


def process_sample(task: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    qid = task["qid"]
    difficulty = task["difficulty"]
    try:
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

        question = task["question"]
        evidence = task.get("evidence")
        db_path = task["db_path"]
        gold_sql = task["gold_sql"]
        candidate_sqls = task["candidate_sqls"]

        # 1) Champion/Challenger selection
        full_schema = SchemaExtractor.extract(db_path)
        if args.phase1_selection_mode == "execution":
            groups = cluster_sql_candidates(db_path, candidate_sqls, timeout=args.sql_timeout)
            champion_sql, challenger_sql = select_champion_and_challenger(groups)
        else:
            grouping = grouper.run(
                question=question,
                candidate_sqls=candidate_sqls,
                full_schema=full_schema,
                evidence=evidence,
                max_correction_attempts=args.max_correction_attempts,
                num_grouping_attempts=args.num_grouping_attempts,
            )
            champion_sql, challenger_sql = select_champion_and_challenger_from_sql_groups(grouping["sql_groups"])

        if not champion_sql:
            return {"qid": qid, "difficulty": difficulty, "status": "no_champion"}
        if not challenger_sql:
            return {"qid": qid, "difficulty": difficulty, "status": "no_challenger", "champion_sql": champion_sql}

        # 1.5) Filter confounded cases:
        # Keep only samples where champion/challenger has at least one EX-correct SQL vs gold on real DB.
        try:
            gold_rows_db = execute_sql(db_path, gold_sql, timeout=args.sql_timeout)
        except Exception as e:
            return {
                "qid": qid,
                "difficulty": difficulty,
                "status": "gold_sql_exec_failed",
                "error": str(e),
            }

        champion_ex_gold = _safe_ex_against_gold_on_db(db_path, champion_sql, gold_rows_db, args.sql_timeout)
        challenger_ex_gold = _safe_ex_against_gold_on_db(db_path, challenger_sql, gold_rows_db, args.sql_timeout)
        if champion_ex_gold == 0.0 and challenger_ex_gold == 0.0:
            return {
                "qid": qid,
                "difficulty": difficulty,
                "status": "filtered_both_wrong_vs_gold",
                "champion_sql": champion_sql,
                "challenger_sql": challenger_sql,
                "champion_ex_gold": champion_ex_gold,
                "challenger_ex_gold": challenger_ex_gold,
            }

        # 2) Generate test data from champ/chall
        sliced_schema = slicer.run(
            candidate_sqls=[champion_sql, challenger_sql],
            full_schema=full_schema,
            max_correction_attempts=args.max_correction_attempts,
        )
        test_data = tester.run(
            question=question,
            sql_1=champion_sql,
            sql_2=challenger_sql,
            sliced_schema=sliced_schema,
            evidence=evidence,
            max_correction_attempts=args.max_correction_attempts,
        )

        # 3) Generate Python answers with multi-sampling + voting
        solver_samples: List[Dict[str, Any]] = []
        solver_errors: List[str] = []
        for _ in range(max(1, args.num_solver_attempts)):
            solver_trace = _run_solver_with_trace(
                llm=llm,
                question=question,
                test_data=test_data,
                sliced_schema=sliced_schema,
                evidence=evidence,
                max_correction_attempts=args.solver_max_correction_attempts,
                python_timeout=args.python_timeout,
            )
            if solver_trace["success"]:
                solver_samples.append(solver_trace)
            else:
                solver_errors.append(str(solver_trace.get("error", "")))

        if not solver_samples:
            return {
                "qid": qid,
                "difficulty": difficulty,
                "status": "solver_failed",
                "champion_sql": champion_sql,
                "challenger_sql": challenger_sql,
                "error": solver_errors[-1] if solver_errors else "All solver attempts failed.",
            }
        winner_solver_trace, solver_vote_meta = _vote_on_solver_samples(solver_samples)
        python_script = winner_solver_trace["python_script"]
        python_answer = winner_solver_trace["python_answer"]

        # 4) Compute answers on same generated test data
        champion_answer = _execute_sql_on_data(champion_sql, test_data, timeout=args.sql_timeout)
        gold_answer = _execute_sql_on_data(gold_sql, test_data, timeout=args.sql_timeout)
        proxy_ex = _strict_ex(champion_answer, python_answer)
        proxy_bsf1 = DPCEvaluator.evaluate(champion_answer, python_answer)

        # 5) LLM judge python correctness with gold-sliced schema
        gold_sliced_schema = slicer.run(
            candidate_sqls=[gold_sql],
            full_schema=full_schema,
            max_correction_attempts=args.max_correction_attempts,
        )
        gold_schema_text = SchemaExtractor.to_readable_text(
            gold_sliced_schema, include_stats=True, include_examples=True, include_descriptions=True
        )
        py_judge = _judge_python_correctness(
            llm=llm,
            question=question,
            evidence=evidence,
            gold_schema_text=gold_schema_text,
            gold_sql=gold_sql,
            gold_answer=gold_answer,
            python_script=python_script,
            python_answer=python_answer,
        )

        # 6) LLM gold label for champion-vs-gold equivalence (replaces EX-based gold label)
        champion_vs_gold = _judge_champion_equivalence(
            llm=llm,
            question=question,
            evidence=evidence,
            gold_schema_text=gold_schema_text,
            champion_sql=champion_sql,
            champion_answer=champion_answer,
            gold_sql=gold_sql,
            gold_answer=gold_answer,
        )
        gold_label = 1 if champion_vs_gold["is_equivalent"] else 0

        return {
            "qid": qid,
            "difficulty": difficulty,
            "status": "ok",
            "question": question,
            "evidence": evidence,
            "champion_sql": champion_sql,
            "challenger_sql": challenger_sql,
            "champion_ex_gold": champion_ex_gold,
            "challenger_ex_gold": challenger_ex_gold,
            "gold_sql": gold_sql,
            "python_script": python_script,
            "python_answer": _safe_jsonable_result(python_answer),
            "champion_answer": _safe_jsonable_result(champion_answer),
            "gold_answer": _safe_jsonable_result(gold_answer),
            "gold_label": gold_label,
            "gold_label_source": "llm_champion_vs_gold_equivalence",
            "champion_vs_gold_judge": champion_vs_gold,
            "proxy_ex": float(proxy_ex),
            "proxy_bsf1": float(proxy_bsf1),
            "python_judge": py_judge,
            "num_solver_attempts": max(1, args.num_solver_attempts),
            "successful_solver_attempts": len(solver_samples),
            "solver_vote": solver_vote_meta,
            "token_usage": llm.get_usage(),
        }
    except Exception as e:
        return {"qid": qid, "difficulty": difficulty, "status": "pipeline_error", "error": str(e)}


def summarize(results: Dict[str, Any]) -> Dict[str, Any]:
    items = list(results.values())
    status_counts = Counter(it.get("status", "unknown") for it in items if isinstance(it, dict))

    ok_items = [it for it in items if isinstance(it, dict) and it.get("status") == "ok"]
    if not ok_items:
        return {"status_counts": dict(status_counts), "message": "No valid ok samples."}

    # Python reliability
    py_correct = sum(1 for it in ok_items if it["python_judge"]["is_correct"])
    py_reliability = py_correct / len(ok_items) * 100.0

    # Champion reliability (EX vs gold on real DB)
    champion_ex_correct = sum(1 for it in ok_items if float(it.get("champion_ex_gold", 0.0)) >= 1.0)
    champion_reliability = champion_ex_correct / len(ok_items) * 100.0

    # Metric reliability correlation with gold_label
    ys = [float(it["gold_label"]) for it in ok_items]
    xs_ex = [float(it["proxy_ex"]) for it in ok_items]
    xs_bsf1 = [float(it["proxy_bsf1"]) for it in ok_items]
    corr_ex = _pearson_corr(xs_ex, ys)
    corr_bsf1 = _pearson_corr(xs_bsf1, ys)
    spear_ex = _spearman_corr(xs_ex, ys)
    spear_bsf1 = _spearman_corr(xs_bsf1, ys)

    # Joint correctness outcomes (python vs sql)
    both_wrong = 0
    python_correct_sql_wrong = 0
    python_wrong_sql_correct = 0
    both_correct = 0
    for it in ok_items:
        py_ok = bool(it["python_judge"]["is_correct"])
        sql_ok = int(it.get("gold_label", 0)) == 1
        if (not py_ok) and (not sql_ok):
            both_wrong += 1
        elif py_ok and (not sql_ok):
            python_correct_sql_wrong += 1
        elif (not py_ok) and sql_ok:
            python_wrong_sql_correct += 1
        else:
            both_correct += 1

    # Difficulty-wise python reliability
    by_diff: Dict[str, Dict[str, Any]] = {}
    for it in ok_items:
        diff = it.get("difficulty", "unknown")
        by_diff.setdefault(diff, {"total": 0, "python_correct": 0, "champion_ex_correct": 0})
        by_diff[diff]["total"] += 1
        by_diff[diff]["python_correct"] += 1 if it["python_judge"]["is_correct"] else 0
        by_diff[diff]["champion_ex_correct"] += 1 if float(it.get("champion_ex_gold", 0.0)) >= 1.0 else 0
    for diff in by_diff:
        d = by_diff[diff]
        d["python_reliability_rate"] = d["python_correct"] / d["total"] * 100.0 if d["total"] > 0 else 0.0
        d["champion_reliability_ex_rate"] = d["champion_ex_correct"] / d["total"] * 100.0 if d["total"] > 0 else 0.0

    return {
        "status_counts": dict(status_counts),
        "filtering": {
            "kept_for_solver_eval": len(ok_items),
            "filtered_both_wrong_vs_gold": status_counts.get("filtered_both_wrong_vs_gold", 0),
            "gold_sql_exec_failed": status_counts.get("gold_sql_exec_failed", 0),
        },
        "python_reliability": {
            "evaluated_samples": len(ok_items),
            "python_correct": py_correct,
            "python_reliability_rate": py_reliability,
        },
        "champion_reliability_ex": {
            "evaluated_samples": len(ok_items),
            "champion_ex_correct": champion_ex_correct,
            "champion_reliability_ex_rate": champion_reliability,
        },
        "metric_reliability": {
            "pearson_corr_proxy_ex_vs_gold_label": corr_ex,
            "pearson_corr_proxy_bsf1_vs_gold_label": corr_bsf1,
            "spearman_corr_proxy_ex_vs_gold_label": spear_ex,
            "spearman_corr_proxy_bsf1_vs_gold_label": spear_bsf1,
            "better_metric_by_pearson": "bs_f1" if corr_bsf1 > corr_ex else "ex",
            "better_metric_by_spearman": "bs_f1" if spear_bsf1 > spear_ex else "ex",
        },
        "joint_outcome_python_vs_sql": {
            "total": len(ok_items),
            "both_wrong": both_wrong,
            "both_wrong_rate": both_wrong / len(ok_items) * 100.0,
            "python_correct_sql_wrong": python_correct_sql_wrong,
            "python_correct_sql_wrong_rate": python_correct_sql_wrong / len(ok_items) * 100.0,
            "python_wrong_sql_correct": python_wrong_sql_correct,
            "python_wrong_sql_correct_rate": python_wrong_sql_correct / len(ok_items) * 100.0,
            "both_correct": both_correct,
            "both_correct_rate": both_correct / len(ok_items) * 100.0,
        },
        "by_difficulty_python_reliability": by_diff,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate solver reliability, BS-F1 reliability, and error distribution.")
    parser.add_argument("--dataset_type", type=str, choices=["bird", "spider"], required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--db_root_path", type=str, required=True)
    parser.add_argument("--pred_sqls_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True, help="Per-sample detailed result json.")
    parser.add_argument("--summary_output_path", type=str, required=True, help="Summary stats json.")
    parser.add_argument("--phase1_selection_mode", type=str, default="execution", choices=["execution", "llm_prompt"])
    parser.add_argument("--num_grouping_attempts", type=int, default=1)
    parser.add_argument("--sql_timeout", type=int, default=30)
    parser.add_argument("--python_timeout", type=int, default=30)
    parser.add_argument("--max_correction_attempts", type=int, default=3)
    parser.add_argument("--solver_max_correction_attempts", type=int, default=1)
    parser.add_argument("--num_solver_attempts", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_delay", type=int, default=2)
    args = parser.parse_args()

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    if args.dataset_type == "bird":
        loader = BirdLoader(args.data_path, args.db_root_path)
    else:
        loader = SpiderLoader(args.data_path, args.db_root_path)

    with open(args.pred_sqls_path, "r", encoding="utf-8") as f:
        all_candidates = json.load(f)
    all_candidates = {str(k): v for k, v in all_candidates.items()}

    tasks = []
    for i in range(len(loader)):
        item = loader.get_item(i)
        qid = str(item.question_id)
        if qid not in all_candidates:
            continue
        tasks.append(
            {
                "qid": qid,
                "difficulty": item.difficulty or "unknown",
                "question": item.question,
                "evidence": item.evidence,
                "gold_sql": item.ground_truth,
                "db_path": loader.get_db_path(item.db_id),
                "candidate_sqls": all_candidates[qid],
            }
        )

    logger.info("Prepared %s tasks.", len(tasks))

    results: Dict[str, Any] = {}
    with ProcessPoolExecutor(max_workers=min(args.num_workers, max(1, len(tasks))), initializer=init_worker) as ex:
        futures = {ex.submit(process_sample, t, args): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Solver Reliability"):
            r = fut.result()
            results[str(r["qid"])] = r

    summary = summarize(results)

    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    sum_dir = os.path.dirname(args.summary_output_path)
    if sum_dir:
        os.makedirs(sum_dir, exist_ok=True)
    with open(args.summary_output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("Summary:\n%s", json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info("Saved per-sample results to %s", args.output_path)
    logger.info("Saved summary to %s", args.summary_output_path)


if __name__ == "__main__":
    main()

