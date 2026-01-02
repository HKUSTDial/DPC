import os
import sys
import json
import logging
import argparse
from tqdm import tqdm
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpc.llm.openai_llm import OpenAILLM
from dpc.datasets.spider_loader import SpiderLoader
from dpc.datasets.bird_loader import BirdLoader
from dpc.utils.schema_utils import SchemaExtractor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Suppress noisy HTTP logs from libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger("Baseline-Generator")

PROMPT_TEMPLATE = """You are an expert SQL programmer.
Your task is to generate a SQL query for a given natural language question based on the provided database schema.

### Database Schema:
{schema_text}

### Natural Language Question:
{question}

{evidence_text}

### Instructions:
1. Think step-by-step (Chain-of-Thought) to understand the requirements and the schema.
2. Formulate the SQL query based on your analysis.

### Output Format:
Your response should follow this format:
<thought>
[Your step-by-step reasoning here]
</thought>
<sql>
[Your final SQL query here]
</sql>

### Analysis and SQL:
"""

def extract_sql(text: str) -> str:
    """Extracts SQL from <sql> tags or returns the whole text if tags are missing."""
    if "<sql>" in text and "</sql>" in text:
        return text.split("<sql>")[-1].split("</sql>")[0].strip()
    return text.strip()

def run_baseline(args):
    # 1. Initialize LLM
    llm = OpenAILLM(
        model_name=args.model_name,
        api_key=args.api_key,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    
    # 2. Load Dataset
    if args.dataset_type.lower() == "spider":
        loader = SpiderLoader(args.data_path, args.db_root_path)
    elif args.dataset_type.lower() == "bird":
        loader = BirdLoader(args.data_path, args.db_root_path)
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")
    
    results = {}
    output_path = args.output
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Check for resume
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            logger.info(f"Resuming from {len(results)} already processed samples.")
        except Exception as e:
            logger.warning(f"Could not load existing results: {e}. Starting fresh.")

    # 3. Define process_item function for parallel execution
    def process_item(index):
        item = loader.get_item(index)
        question_id = str(item.question_id)
        
        if question_id in results:
            return None # Skip
            
        schema = loader.get_schema(item.db_id)
        schema_text = SchemaExtractor.to_readable_text(
            schema, 
            include_stats=False, 
            include_examples=False, 
            include_descriptions=False
        )
        
        evidence_text = f"### Evidence:\n{item.evidence}\n" if item.evidence else ""
        
        prompt = PROMPT_TEMPLATE.format(
            schema_text=schema_text,
            question=item.question,
            evidence_text=evidence_text
        )
        
        candidates = []
        # Each question gets its own LLM instance for better token tracking per-question if needed,
        # but here we can just use the shared one and track the total.
        # Actually, let's create a local LLM to track usage accurately for this question.
        local_llm = OpenAILLM(
            model_name=args.model_name,
            api_key=args.api_key,
            base_url=args.base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        
        # Multi-threading for candidates of a single question
        with ThreadPoolExecutor(max_workers=args.num_candidates) as candidate_executor:
            futures = [candidate_executor.submit(local_llm.ask, [{"role": "user", "content": prompt}]) for _ in range(args.num_candidates)]
            for future in as_completed(futures):
                try:
                    response = future.result()
                    sql = extract_sql(response)
                    candidates.append(sql)
                except Exception as e:
                    logger.error(f"Error generating SQL for ID {question_id}: {e}")
        
        usage = local_llm.get_usage()
        
        return question_id, candidates, usage

    # 4. Generate Predictions using ThreadPoolExecutor
    tasks_indices = [i for i in range(len(loader)) if str(loader.get_item(i).question_id) not in results]
    
    if not tasks_indices:
        logger.info("All samples have already been processed.")
        return

    logger.info(f"Starting parallel generation for {len(tasks_indices)} samples with {args.num_workers} workers...")
    
    total_prompt_tokens = 0
    total_completion_tokens = 0
    count = 0

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_item, i): i for i in tasks_indices}
        
        for i, future in enumerate(tqdm(as_completed(futures), total=len(tasks_indices), desc="Generating Baseline SQLs")):
            res = future.result()
            if res:
                question_id, candidates, usage = res
                results[question_id] = candidates
                
                total_prompt_tokens += usage["prompt_tokens"]
                total_completion_tokens += usage["completion_tokens"]
                count += 1
            
            # Periodic save
            if i % 10 == 0:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)
                
    # Final save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    logger.info(f"Completed! Results saved to {output_path}")

    if count > 0:
        avg_prompt = total_prompt_tokens / count
        avg_completion = total_completion_tokens / count
        avg_total = (total_prompt_tokens + total_completion_tokens) / count
        logger.info(f"--- Statistics (Average per question) ---")
        logger.info(f"Prompt Tokens: {avg_prompt:.1f}")
        logger.info(f"Completion Tokens: {avg_completion:.1f}")
        logger.info(f"Total Tokens: {avg_total:.1f}")
        logger.info(f"----------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM baseline for SQL generation")
    
    # Dataset Arguments
    parser.add_argument("--dataset_type", type=str, default="bird", choices=["bird", "spider"], help="Dataset type")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset JSON file")
    parser.add_argument("--db_root_path", type=str, required=True, help="Directory containing databases")
    parser.add_argument("--output", type=str, required=True, help="Output path for pred_sqls JSON")
    
    # LLM Arguments
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="LLM model name")
    parser.add_argument("--api_key", type=str, default=None, help="LLM API Key (optional, fallbacks to env)")
    parser.add_argument("--base_url", type=str, default=None, help="LLM Base URL (optional)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens per response")
    
    # Execution Arguments
    parser.add_argument("--num_candidates", type=int, default=5, help="Number of candidate SQLs per question")
    parser.add_argument("--num_workers", type=int, default=10, help="Number of samples to process in parallel")
    
    args = parser.parse_args()
    run_baseline(args)
