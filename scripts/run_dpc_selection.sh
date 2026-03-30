#!/bin/bash

# --- Dataset Configuration ---
ARTIFACT_ROOT=${ARTIFACT_ROOT:-"artifacts"}
DATASET_TYPE=${DATASET_TYPE:-"bird"}
DATA_PATH=${DATA_PATH:-"data/bird/dev/mini_dev.json"}
DB_ROOT_PATH=${DB_ROOT_PATH:-"data/bird/dev/dev_databases"}
PRED_SQLS_PATH=${PRED_SQLS_PATH:-"${ARTIFACT_ROOT}/candidates/Qwen2.5-Coder-7B-Instruct_BIRD_Mini_Dev.json"}
OUTPUT_PATH=${OUTPUT_PATH:-"${ARTIFACT_ROOT}/temp/Qwen2.5-Coder-7B-Instruct_BIRD_Mini_Dev_DPC.json"}

# --- LLM Configuration ---
MODEL_NAME=${MODEL_NAME:-"qwen2.5-coder-7b-instruct"}
API_KEY=${API_KEY:-""}
BASE_URL=${BASE_URL:-""}
TEMPERATURE=${TEMPERATURE:-1.0}
MAX_TOKENS=${MAX_TOKENS:-4096}
MAX_RETRIES=${MAX_RETRIES:-3}
RETRY_DELAY=${RETRY_DELAY:-2}

# --- Pipeline Configuration ---
NUM_WORKERS=${NUM_WORKERS:-8}
SQL_TIMEOUT=${SQL_TIMEOUT:-30}
PYTHON_TIMEOUT=${PYTHON_TIMEOUT:-30}
EPSILON=${EPSILON:-0.0}
MAX_CORRECTION_ATTEMPTS=${MAX_CORRECTION_ATTEMPTS:-0}
NUM_TEST_DATA=${NUM_TEST_DATA:-3}
NUM_SOLVER_ATTEMPTS=${NUM_SOLVER_ATTEMPTS:-3}

echo "Starting DPC-SQL Pipeline..."
echo "Dataset: $DATASET_TYPE"
echo "Input: $PRED_SQLS_PATH"
echo "Output: $OUTPUT_PATH"
echo "Model: $MODEL_NAME"

# Build command arguments
CMD_ARGS=(
    --dataset_type "$DATASET_TYPE"
    --data_path "$DATA_PATH"
    --db_root_path "$DB_ROOT_PATH"
    --pred_sqls_path "$PRED_SQLS_PATH"
    --output_path "$OUTPUT_PATH"
    --model_name "$MODEL_NAME"
    --temperature "$TEMPERATURE"
    --max_tokens "$MAX_TOKENS"
    --max_retries "$MAX_RETRIES"
    --retry_delay "$RETRY_DELAY"
    --num_workers "$NUM_WORKERS"
    --sql_timeout "$SQL_TIMEOUT"
    --python_timeout "$PYTHON_TIMEOUT"
    --epsilon "$EPSILON"
    --max_correction_attempts "$MAX_CORRECTION_ATTEMPTS"
    --num_test_data "$NUM_TEST_DATA"
    --num_solver_attempts "$NUM_SOLVER_ATTEMPTS"
)

# Add API Key and Base URL if provided
if [ -n "$API_KEY" ]; then
    CMD_ARGS+=(--api_key "$API_KEY")
fi

if [ -n "$BASE_URL" ]; then
    CMD_ARGS+=(--base_url "$BASE_URL")
fi

python baseline/run_dpc_selection.py "${CMD_ARGS[@]}"
