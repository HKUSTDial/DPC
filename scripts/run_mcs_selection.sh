#!/bin/bash

# --- Configuration ---
ARTIFACT_ROOT=${ARTIFACT_ROOT:-"artifacts"}
DATASET_TYPE=${DATASET_TYPE:-"bird"}
DATA_PATH=${DATA_PATH:-"data/bird/dev/mini_dev.json"}
DB_ROOT_PATH=${DB_ROOT_PATH:-"data/bird/dev/dev_databases"}
CANDIDATES_PATH=${CANDIDATES_PATH:-"${ARTIFACT_ROOT}/candidates/Qwen2.5-Coder-7B-Instruct_BIRD_Mini_Dev.json"}
OUTPUT_PATH=${OUTPUT_PATH:-"${ARTIFACT_ROOT}/temp/Qwen2.5-Coder-7B-Instruct_BIRD_Mini_Dev_MCS.json"}

# --- LLM Configuration ---
MODEL_NAME=${MODEL_NAME:-"qwen2.5-coder-7b-instruct"}
API_KEY=${API_KEY:-""}
BASE_URL=${BASE_URL:-""}
TEMPERATURE=${TEMPERATURE:-0.7}
MAX_TOKENS=${MAX_TOKENS:-4096}

# --- Execution Configuration ---
TIMEOUT=${TIMEOUT:-30}
NUM_WORKERS=${NUM_WORKERS:-8}

echo "Starting Multi-Choice Selection (MCS) Baseline..."
echo "Input: $CANDIDATES_PATH"
echo "Output: $OUTPUT_PATH"
echo "Model: $MODEL_NAME"
echo "Workers: $NUM_WORKERS"

# Build command arguments
CMD_ARGS=(
    --dataset_type "$DATASET_TYPE"
    --data_path "$DATA_PATH"
    --db_root_path "$DB_ROOT_PATH"
    --pred_path "$CANDIDATES_PATH"
    --output_path "$OUTPUT_PATH"
    --model_name "$MODEL_NAME"
    --temperature "$TEMPERATURE"
    --max_tokens "$MAX_TOKENS"
    --timeout "$TIMEOUT"
    --num_workers "$NUM_WORKERS"
)

# Add API Key and Base URL if provided
if [ -n "$API_KEY" ]; then
    CMD_ARGS+=(--api_key "$API_KEY")
fi

if [ -n "$BASE_URL" ]; then
    CMD_ARGS+=(--base_url "$BASE_URL")
fi

# Run the script
python baseline/run_mcs_selection.py "${CMD_ARGS[@]}"
