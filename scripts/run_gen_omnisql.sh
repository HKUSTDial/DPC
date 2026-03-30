#!/bin/bash

# --- Configuration ---
ARTIFACT_ROOT=${ARTIFACT_ROOT:-"artifacts"}
DATASET_TYPE=${DATASET_TYPE:-"bird"}
DATA_PATH=${DATA_PATH:-"data/bird/dev/mini_dev.json"}
DB_ROOT_PATH=${DB_ROOT_PATH:-"data/bird/dev/dev_databases"}
OUTPUT_PATH=${OUTPUT_PATH:-"${ARTIFACT_ROOT}/candidates/OmniSQL-7B_BIRD_Mini_Dev.json"}

# --- LLM Configuration ---
MODEL_NAME=${MODEL_NAME:-"OmniSQL-7B"}
API_KEY=${API_KEY:-""}
BASE_URL=${BASE_URL:-"http://localhost:9009/v1"}
TEMPERATURE=${TEMPERATURE:-0.0}
MAX_TOKENS=${MAX_TOKENS:-2048}

# --- Execution Configuration ---
NUM_CANDIDATES=${NUM_CANDIDATES:-5}
NUM_WORKERS=${NUM_WORKERS:-5}

echo "Starting OmniSQL SQL Generation..."
echo "Dataset: $DATASET_TYPE"
echo "Model: $MODEL_NAME"
echo "Candidates per sample: $NUM_CANDIDATES"
echo "Workers: $NUM_WORKERS"

# Build command arguments
CMD_ARGS=(
    --dataset_type "$DATASET_TYPE"
    --data_path "$DATA_PATH"
    --db_root_path "$DB_ROOT_PATH"
    --output "$OUTPUT_PATH"
    --model_name "$MODEL_NAME"
    --temperature "$TEMPERATURE"
    --max_tokens "$MAX_TOKENS"
    --num_candidates "$NUM_CANDIDATES"
    --num_workers "$NUM_WORKERS"
)

# Add API Key and Base URL if provided
if [ -n "$API_KEY" ]; then
    CMD_ARGS+=(--api_key "$API_KEY")
fi

if [ -n "$BASE_URL" ]; then
    CMD_ARGS+=(--base_url "$BASE_URL")
fi

python baseline/run_sql_generation_with_omnisql.py "${CMD_ARGS[@]}"
