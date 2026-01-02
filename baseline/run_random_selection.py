import os
import sys
import json
import argparse
import logging
import random
from typing import List, Dict, Any

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Random-Selection")

def main():
    parser = argparse.ArgumentParser(description="Randomly select one SQL from candidates for each question.")
    parser.add_argument("--pred_path", type=str, required=True, help="Path to predicted SQL candidates JSON file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the selected SQLs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    args = parser.parse_args()

    # 1. Set seed
    random.seed(args.seed)

    # 2. Load Candidates
    if not os.path.exists(args.pred_path):
        raise FileNotFoundError(f"Predicted SQLs file not found: {args.pred_path}")
    
    with open(args.pred_path, 'r', encoding='utf-8') as f:
        all_candidates = json.load(f)
    
    # 3. Perform Random Selection
    selected_sqls = {}
    for qid, candidates in all_candidates.items():
        if not candidates:
            selected_sqls[qid] = None
            continue
        
        # Randomly select one
        selected_sqls[qid] = random.choice(candidates)

    # 4. Save Results
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(selected_sqls, f, indent=4, ensure_ascii=False)
    
    logger.info(f"Random selection complete. Selected SQLs saved to {args.output_path}")

if __name__ == "__main__":
    main()

