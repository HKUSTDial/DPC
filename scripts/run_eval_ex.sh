#!/bin/bash

# --- Configuration ---
# You can override these variables or pass them as environment variables
DATASET_TYPE=${DATASET_TYPE:-"bird"}
DATA_PATH=${DATA_PATH:-"data/bird/dev/mini_dev.json"}
DB_ROOT_PATH=${DB_ROOT_PATH:-"data/bird/dev/dev_databases"}
PRED_PATH=${PRED_PATH:-"results/temp/Qwen2.5-Coder-7B-Instruct_BIRD_Mini_Dev_DPC.json"}

# --- Execution Configuration ---
TIMEOUT=${TIMEOUT:-30}
NUM_WORKERS=${NUM_WORKERS:-8}

echo "Starting Execution Accuracy (EX) Evaluation..."
echo "Dataset: $DATASET_TYPE"
echo "Data Path: $DATA_PATH"
echo "Predictions: $PRED_PATH"
echo "Workers: $NUM_WORKERS"

# Build command arguments
CMD_ARGS=(
    --dataset_type "$DATASET_TYPE"
    --data_path "$DATA_PATH"
    --db_root_path "$DB_ROOT_PATH"
    --pred_path "$PRED_PATH"
    --timeout "$TIMEOUT"
    --num_workers "$NUM_WORKERS"
)

# Run the evaluation script
python evaluation/eval_ex.py "${CMD_ARGS[@]}"

