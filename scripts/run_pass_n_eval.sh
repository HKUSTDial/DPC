#!/bin/bash

# --- Configuration ---
DATASET_TYPE=${DATASET_TYPE:-"bird"}
DATA_PATH=${DATA_PATH:-"data/bird/dev/mini_dev.json"}
DB_ROOT_PATH=${DB_ROOT_PATH:-"data/bird/dev/dev_databases"}
CANDIDATES_PATH=${CANDIDATES_PATH:-"results/candidates/Qwen2.5-Coder-7B-Instruct_BIRD_Mini_Dev.json"}

# --- Execution Configuration ---
TIMEOUT=${TIMEOUT:-30}
NUM_WORKERS=${NUM_WORKERS:-8}

echo "Starting Pass@N Evaluation..."
echo "Dataset: $DATASET_TYPE"
echo "Candidates: $CANDIDATES_PATH"
echo "Workers: $NUM_WORKERS"

# Run the evaluation script
python evaluation/eval_pass_n.py \
    --dataset_type "$DATASET_TYPE" \
    --data_path "$DATA_PATH" \
    --db_root_path "$DB_ROOT_PATH" \
    --candidates_path "$CANDIDATES_PATH" \
    --timeout "$TIMEOUT" \
    --num_workers "$NUM_WORKERS"

