import os
import json
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any
from tqdm import tqdm

from dpc.llm.openai_llm import OpenAILLM
from dpc.agents.slicer_agent import SlicerAgent
from dpc.agents.tester_agent import TesterAgent
from dpc.agents.solver_agent import PythonSolverAgent
from dpc.core.pipeline import DPCPipeline
from dpc.datasets.spider_loader import SpiderLoader
from dpc.datasets.bird_loader import BirdLoader
from dpc.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("dpc_run.log"), logging.StreamHandler()]
)
logger = logging.getLogger("DPC-Batch")

def process_sample(item_data: Dict[str, Any], candidate_sqls: List[str]) -> Dict[str, Any]:
    """
    Worker function to process a single Text-to-SQL sample.
    Initializes its own LLM and Agents to ensure process isolation.
    """
    try:
        # 1. Initialize LLM inside the process
        llm = OpenAILLM(
            model_name=config.llm.model_name,
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            max_retries=config.llm.max_retries,
            retry_delay=config.llm.retry_delay
        )

        # 2. Initialize Agents
        slicer = SlicerAgent(llm)
        tester = TesterAgent(llm)
        solver = PythonSolverAgent(llm)
        pipeline = DPCPipeline(slicer=slicer, tester=tester, solver=solver)

        # 3. Run Pipeline
        result = pipeline.run(
            question=item_data["question"],
            db_path=item_data["db_path"],
            candidate_sqls=candidate_sqls,
            evidence=item_data.get("evidence"),
            sql_timeout=config.pipeline.sql_timeout,
            epsilon=config.pipeline.epsilon,
            max_correction_attempts=config.pipeline.max_correction_attempts
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
    # 1. Initialization
    multiprocessing.set_start_method('spawn', force=True)
    
    # 2. Load External Config (Optional config.toml)
    if os.path.exists("config.toml"):
        logger.info("Loading configuration from config.toml")
        config.load_from_toml("config.toml")
    
    # 3. Load Dataset
    if config.dataset.dataset_type.lower() == "spider":
        loader = SpiderLoader(config.dataset.data_path, config.dataset.db_root_path)
    elif config.dataset.dataset_type.lower() == "bird":
        loader = BirdLoader(config.dataset.data_path, config.dataset.db_root_path)
    else:
        raise ValueError(f"Unknown dataset type: {config.dataset.dataset_type}")

    # 4. Load Predicted SQLs
    if not os.path.exists(config.dataset.pred_sqls_path):
        raise FileNotFoundError(f"Predicted SQLs file not found: {config.dataset.pred_sqls_path}")
    
    with open(config.dataset.pred_sqls_path, 'r', encoding='utf-8') as f:
        all_pred_sqls = json.load(f)

    # 5. Prepare Tasks
    tasks = []
    for i in range(len(loader)):
        item = loader.get_item(i)
        if item.question_id in all_pred_sqls:
            tasks.append({
                "item_data": {
                    "question_id": item.question_id,
                    "question": item.question,
                    "db_path": loader.get_db_path(item.db_id),
                    "evidence": item.evidence
                },
                "candidate_sqls": all_pred_sqls[item.question_id]
            })

    logger.info(f"Loaded {len(tasks)} samples from dataset.")

    # 6. Check for Resume Logic
    output_path = config.dataset.output_path
    results = []
    processed_ids = set()
    
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
                # Ensure it's a list of dicts
                if isinstance(results, list):
                    processed_ids = {str(r["question_id"]) for r in results}
                    logger.info(f"Resuming from existing results. {len(processed_ids)} samples already processed.")
                else:
                    results = []
        except Exception as e:
            logger.warning(f"Could not load existing results for resume: {e}. Starting fresh.")

    # Filter out tasks that are already processed
    tasks_to_run = [t for t in tasks if str(t["item_data"]["question_id"]) not in processed_ids]
    
    if not tasks_to_run:
        logger.info("All samples have already been processed. Nothing to do.")
        return

    logger.info(f"Starting parallel processing for {len(tasks_to_run)} remaining samples...")

    # 7. Execute in Parallel with Real-time Saving
    num_workers = min(multiprocessing.cpu_count(), 8) 
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_sample, t["item_data"], t["candidate_sqls"]): t for t in tasks_to_run}
        
        for future in tqdm(as_completed(futures), total=len(tasks_to_run), desc="DPC Processing"):
            res = future.result()
            results.append(res)
            
            # Real-time saving to disk
            try:
                # We save the entire list to maintain a valid JSON structure
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)
            except Exception as e:
                logger.error(f"Failed to save real-time results for ID {res.get('question_id')}: {e}")

    # 8. Summary
    success_count = sum(1 for r in results if r.get("success"))
    logger.info(f"Batch processing completed. Total processed: {len(results)}. Success: {success_count}/{len(results)}")
    logger.info(f"All results are permanently saved to {output_path}")

if __name__ == "__main__":
    main()
