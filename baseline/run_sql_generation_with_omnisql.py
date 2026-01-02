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

logger = logging.getLogger("OmniSQL-Generator")

# Custom Prompt Template for OmniSQL
OMNISQL_PROMPT_TEMPLATE = '''Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

Database Engine:
SQLite

Database Schema:
{db_details}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{question}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.

Output Format:
In your answer, please enclose the generated SQL query in a code block:
```
-- Your SQL query
```

Take a deep breath and think step by step to find the correct SQL query.'''

def extract_sql(text: str) -> str:
    """
    Extracts SQL from the response. 
    OmniSQL prompt expects SQL in a code block.
    """
    # Regex to find content in triple backticks
    import re
    # Matches ```sql ... ``` or just ``` ... ```
    match = re.search(r'```(?:sql)?\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Fallback to the whole text if no code block found
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
            
        # Extract DDL for OmniSQL
        db_path = loader.get_db_path(item.db_id)
        db_details = SchemaExtractor.get_db_ddl(db_path)
        
        # Prepare question with evidence if available
        question_text = item.question
        if item.evidence:
            question_text = f"{question_text}\n{item.evidence}"
        
        # Format the specialized OmniSQL prompt
        prompt = OMNISQL_PROMPT_TEMPLATE.format(
            db_details=db_details,
            question=question_text
        )
        
        candidates = []
        # Multi-threading for candidates of a single question
        with ThreadPoolExecutor(max_workers=args.num_candidates) as candidate_executor:
            futures = [candidate_executor.submit(llm.ask, [{"role": "user", "content": prompt}]) for _ in range(args.num_candidates)]
            for future in as_completed(futures):
                try:
                    response = future.result()
                    sql = extract_sql(response)
                    candidates.append(sql)
                except Exception as e:
                    logger.error(f"Error generating SQL for ID {question_id}: {e}")
        
        return question_id, candidates

    # 4. Generate Predictions using ThreadPoolExecutor
    tasks_indices = [i for i in range(len(loader)) if str(loader.get_item(i).question_id) not in results]
    
    if not tasks_indices:
        logger.info("All samples have already been processed.")
        return

    logger.info(f"Starting parallel generation for {len(tasks_indices)} samples with {args.num_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_item, i): i for i in tasks_indices}
        
        for count, future in enumerate(tqdm(as_completed(futures), total=len(tasks_indices), desc="Generating OmniSQL SQLs")):
            res = future.result()
            if res:
                question_id, candidates = res
                results[question_id] = candidates
            
            # Periodic save
            if count % 10 == 0:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)
                
    # Final save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    logger.info(f"Completed! Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OmniSQL specialized SQL generation")
    
    # Dataset Arguments
    parser.add_argument("--dataset_type", type=str, default="bird", choices=["bird", "spider"], help="Dataset type")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset JSON file")
    parser.add_argument("--db_root_path", type=str, required=True, help="Directory containing databases")
    parser.add_argument("--output", type=str, required=True, help="Output path for pred_sqls JSON")
    
    # LLM Arguments
    parser.add_argument("--model_name", type=str, default="omnisql-v1", help="LLM model name")
    parser.add_argument("--api_key", type=str, default=None, help="LLM API Key")
    parser.add_argument("--base_url", type=str, default=None, help="LLM Base URL")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens per response")
    
    # Execution Arguments
    parser.add_argument("--num_candidates", type=int, default=5, help="Number of candidate SQLs per question")
    parser.add_argument("--num_workers", type=int, default=5, help="Number of samples to process in parallel")
    
    args = parser.parse_args()
    run_baseline(args)

