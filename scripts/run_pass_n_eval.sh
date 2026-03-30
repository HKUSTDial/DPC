#!/bin/bash

# --- Configuration ---
ARTIFACT_ROOT=${ARTIFACT_ROOT:-"artifacts"}
DATASET_TYPE=${DATASET_TYPE:-"spider"}
DATA_PATH=${DATA_PATH:-"data/spider/test.json"}
DB_ROOT_PATH=${DB_ROOT_PATH:-"data/spider/test_database"}
CANDIDATES_PATH=${CANDIDATES_PATH:-"${ARTIFACT_ROOT}/candidates/Qwen2.5-Coder-7B-Instruct_SPIDER_Test.json"}

# --- Execution Configuration ---
TIMEOUT=${TIMEOUT:-30}
PASS_K=${PASS_K:-2}
NUM_WORKERS=${NUM_WORKERS:-8}

echo "Starting Pass@N Evaluation..."
echo "Dataset: $DATASET_TYPE"
echo "Candidates: $CANDIDATES_PATH"
echo "Pass@K: $PASS_K"
echo "Workers: $NUM_WORKERS"

# Run the evaluation script
python evaluation/eval_pass_n.py \
    --dataset_type "$DATASET_TYPE" \
    --data_path "$DATA_PATH" \
    --db_root_path "$DB_ROOT_PATH" \
    --candidates_path "$CANDIDATES_PATH" \
    --timeout "$TIMEOUT" \
    --k "$PASS_K" \
    --num_workers "$NUM_WORKERS"
