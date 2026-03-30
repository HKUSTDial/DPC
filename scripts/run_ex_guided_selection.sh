#!/bin/bash

# --- Configuration ---
if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x ".venv/bin/python" ]; then
        PYTHON_BIN=".venv/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi
ARTIFACT_ROOT=${ARTIFACT_ROOT:-"artifacts"}
DATASET_TYPE=${DATASET_TYPE:-"bird"}
DATA_PATH=${DATA_PATH:-"data/bird/dev/mini_dev.json"}
DB_ROOT_PATH=${DB_ROOT_PATH:-"data/bird/dev/dev_databases"}
CANDIDATES_PATH=${CANDIDATES_PATH:-"${ARTIFACT_ROOT}/candidates/Qwen2.5-Coder-7B-Instruct_BIRD_Mini_Dev.json"}
OUTPUT_PATH=${OUTPUT_PATH:-"${ARTIFACT_ROOT}/selected/Qwen2.5-Coder-7B-Instruct_BIRD_Mini_Dev_EX_Guided.json"}

# --- Execution Configuration ---
TIMEOUT=${TIMEOUT:-30}
NUM_WORKERS=${NUM_WORKERS:-8}

echo "Starting EX-Guided Selection Baseline..."
echo "Input: $CANDIDATES_PATH"
echo "Output: $OUTPUT_PATH"
echo "Workers: $NUM_WORKERS"

# Run the script
"$PYTHON_BIN" baseline/run_ex_guided_selection.py \
    --dataset_type "$DATASET_TYPE" \
    --data_path "$DATA_PATH" \
    --db_root_path "$DB_ROOT_PATH" \
    --pred_path "$CANDIDATES_PATH" \
    --output_path "$OUTPUT_PATH" \
    --timeout "$TIMEOUT" \
    --num_workers "$NUM_WORKERS"
