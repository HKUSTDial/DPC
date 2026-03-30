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
DATASET_TYPE=${DATASET_TYPE:-"bird"}
DATA_PATH=${DATA_PATH:-"data/bird/dev/mini_dev.json"}
DB_ROOT_PATH=${DB_ROOT_PATH:-"data/bird/dev/dev_databases"}
CANDIDATE_PATH=${CANDIDATE_PATH:-"${ARTIFACT_ROOT}/candidates/Qwen2.5-Coder-7B-Instruct_BIRD_Mini_Dev.json"}
SC_PRED_PATH=${SC_PRED_PATH:-"${ARTIFACT_ROOT}/selected/Qwen2.5-Coder-7B-Instruct_BIRD_Mini_Dev_SC.json"}
DPC_PRED_PATH=${DPC_PRED_PATH:-"${ARTIFACT_ROOT}/selected/Qwen2.5-Coder-7B-Instruct_BIRD_Mini_Dev_DPC.json"}
OUTPUT_PATH=${OUTPUT_PATH:-"${ARTIFACT_ROOT}/analysis/Qwen2.5-Coder-7B-Instruct_BIRD_Mini_Dev_majority_analysis.json"}

# --- Execution Configuration ---
TIMEOUT=${TIMEOUT:-30}
NUM_WORKERS=${NUM_WORKERS:-8}

echo "Starting Majority Analysis..."
echo "Dataset: $DATASET_TYPE"
echo "Data Path: $DATA_PATH"
echo "Candidate Path: $CANDIDATE_PATH"
echo "SC Pred Path: $SC_PRED_PATH"
echo "DPC Pred Path: $DPC_PRED_PATH"
echo "Output Path: $OUTPUT_PATH"
echo "Workers: $NUM_WORKERS"
echo ""

# Build command arguments
CMD_ARGS=(
    --candidate_path "$CANDIDATE_PATH"
    --sc_pred_path "$SC_PRED_PATH"
    --dpc_pred_path "$DPC_PRED_PATH"
    --dataset_type "$DATASET_TYPE"
    --data_path "$DATA_PATH"
    --db_root_path "$DB_ROOT_PATH"
    --output_path "$OUTPUT_PATH"
    --timeout "$TIMEOUT"
    --num_workers "$NUM_WORKERS"
)

# Run the analysis script
"$PYTHON_BIN" baseline/run_majority_analysis.py "${CMD_ARGS[@]}"
