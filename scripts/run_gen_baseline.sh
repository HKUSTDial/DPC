#!/bin/bash

# --- Configuration ---
# You can override these variables or pass them as environment variables
if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x ".venv/bin/python" ]; then
        PYTHON_BIN=".venv/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi
ARTIFACT_ROOT=${ARTIFACT_ROOT:-"artifacts"}
DATASET_TYPE=${DATASET_TYPE:-"spider"}
DATA_PATH=${DATA_PATH:-"data/spider/test.json"}
DB_ROOT_PATH=${DB_ROOT_PATH:-"data/spider/test_database"}
OUTPUT_PATH=${OUTPUT_PATH:-"${ARTIFACT_ROOT}/candidates/GPT-5_SPIDER_Test.json"}
# --- LLM Configuration ---
MODEL_NAME=${MODEL_NAME:-"gpt-5-chat-latest"}
API_KEY=${API_KEY:-""}
BASE_URL=${BASE_URL:-""}
TEMPERATURE=${TEMPERATURE:-0.7}
MAX_TOKENS=${MAX_TOKENS:-4096}

# --- Execution Configuration ---
NUM_CANDIDATES=${NUM_CANDIDATES:-5}
NUM_WORKERS=${NUM_WORKERS:-8}

echo "Starting Baseline SQL Generation..."
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

"$PYTHON_BIN" baseline/run_sql_generation.py "${CMD_ARGS[@]}"
