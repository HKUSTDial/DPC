import json
import logging
import os
import signal
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Tuple

from dpc.datasets.bird_loader import BirdLoader
from dpc.datasets.spider_loader import SpiderLoader


def build_loader(dataset_type: str, data_path: str, db_root_path: str):
    dataset_type = dataset_type.lower()
    if dataset_type == "spider":
        return SpiderLoader(data_path, db_root_path)
    if dataset_type == "bird":
        return BirdLoader(data_path, db_root_path)
    raise ValueError(f"Unknown dataset type: {dataset_type}")


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_candidate_map(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Candidate JSON file not found: {path}")
    loaded = load_json(path)
    if not isinstance(loaded, dict):
        raise ValueError("Expected top-level dict for candidate JSON.")
    return {str(k): v for k, v in loaded.items()}


def iter_dataset_with_candidates(loader, candidate_map: Dict[str, Any]) -> Iterator[Tuple[str, Any, Any]]:
    for i in range(len(loader)):
        item = loader.get_item(i)
        qid = str(item.question_id)
        if qid not in candidate_map:
            continue
        yield qid, item, candidate_map[qid]


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def save_json(path: str, data: Any, indent: int = 4) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def save_json_atomic(path: str, data: Any, indent: int = 4) -> None:
    ensure_parent_dir(path)
    parent = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile("w", dir=parent, delete=False, encoding="utf-8") as tmp:
        json.dump(data, tmp, indent=indent, ensure_ascii=False)
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def init_worker_ignore_sigint() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def terminate_executor_workers(executor: Any) -> None:
    """
    Best-effort targeted shutdown for ProcessPoolExecutor workers.

    This avoids killing the whole process group on Ctrl-C, which can also take
    down the parent shell or unrelated processes launched in the same group.
    """
    processes = list((getattr(executor, "_processes", None) or {}).values())
    for proc in processes:
        if proc.is_alive():
            proc.terminate()
    for proc in processes:
        proc.join(timeout=1)
    for proc in processes:
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=1)


def build_llm_config(args) -> Dict[str, Any]:
    config = {
        "model_name": args.model_name,
        "api_key": getattr(args, "api_key", None),
        "base_url": getattr(args, "base_url", None),
        "temperature": getattr(args, "temperature", 0.0),
        "max_tokens": getattr(args, "max_tokens", 2048),
    }
    if hasattr(args, "max_retries"):
        config["max_retries"] = args.max_retries
    if hasattr(args, "retry_delay"):
        config["retry_delay"] = args.retry_delay
    return config


@dataclass
class UsageStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    samples: int = 0

    def update(self, usage: Dict[str, Any]) -> None:
        self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        self.samples += 1

    def log_average(self, logger: logging.Logger, label: str = "Statistics") -> None:
        if self.samples <= 0:
            return
        avg_prompt = self.prompt_tokens / self.samples
        avg_completion = self.completion_tokens / self.samples
        avg_total = (self.prompt_tokens + self.completion_tokens) / self.samples
        logger.info("--- %s (Average per question) ---", label)
        logger.info("Prompt Tokens: %.1f", avg_prompt)
        logger.info("Completion Tokens: %.1f", avg_completion)
        logger.info("Total Tokens: %.1f", avg_total)
        logger.info("----------------------------------------")
