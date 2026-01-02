#!/bin/bash

# --- Configuration ---
# Use the output of your generation baseline as PRE_PATH
DATASET_TYPE=${DATASET_TYPE:-"bird"}
DATA_PATH=${DATA_PATH:-"data/bird/dev/mini_dev.json"}
DB_ROOT_PATH=${DB_ROOT_PATH:-"data/bird/dev/dev_databases"}
PRED_PATH=${PRED_PATH:-"results/candidates/GPT-5_BIRD_Mini_Dev.json"}
OUTPUT_PATH=${OUTPUT_PATH:-"results/selected/GPT-5_BIRD_Mini_Dev_SC.json"}
NUM_WORKERS=${NUM_WORKERS:-8}
TIMEOUT=${TIMEOUT:-30}

echo "Starting Self-Consistency (SC) Selection..."
echo "Dataset: $DATASET_TYPE"
echo "Input Candidates: $PRED_PATH"
echo "Output Selected: $OUTPUT_PATH"
echo "Workers: $NUM_WORKERS"

python baseline/run_sc_selection.py \
    --pred_path "$PRED_PATH" \
    --dataset_type "$DATASET_TYPE" \
    --data_path "$DATA_PATH" \
    --db_root_path "$DB_ROOT_PATH" \
    --output_path "$OUTPUT_PATH" \
    --num_workers "$NUM_WORKERS" \
    --timeout "$TIMEOUT"

