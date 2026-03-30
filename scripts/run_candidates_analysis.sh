#!/bin/bash

# =============================================================================
# Batch Analysis Script: Compare DPC vs SC across different candidate counts
# =============================================================================
# This script runs experiments for different numbers of SQL candidates (N)
# and compares DPC vs SC performance for each N.
#
# Usage:
#   bash scripts/run_candidates_analysis.sh
#
# Environment Variables (can be overridden):
#   - MODEL_NAME: Model name (default: "Qwen2.5-Coder-7B-Instruct")
#   - DATASET_TYPE: Dataset type (default: "bird")
#   - DATA_PATH: Path to dataset file (default: "data/bird/dev/mini_dev.json")
#   - DB_ROOT_PATH: Path to database root (default: "data/bird/dev/dev_databases")
#   - DATASET_NAME: Dataset name for output files (default: inferred from DATA_PATH)
#                    e.g., "BIRD_Mini_Dev" or "SPIDER_Test"
#   - NUM_CANDIDATES_LIST: Space-separated list of candidate counts (default: "3 5 7 9 11")
#   - NUM_WORKERS: Number of parallel workers (default: 8)
#   - API_KEY: LLM API key (if needed)
#   - BASE_URL: LLM API base URL (if needed)
# =============================================================================

set -e  # Exit on error

# --- Configuration ---
ARTIFACT_ROOT=${ARTIFACT_ROOT:-"artifacts"}
MODEL_NAME=${MODEL_NAME:-"qwen2.5-coder-7b-instruct"}
DATASET_TYPE=${DATASET_TYPE:-"bird"}
DATA_PATH=${DATA_PATH:-"data/bird/dev/mini_dev.json"}
DB_ROOT_PATH=${DB_ROOT_PATH:-"data/bird/dev/dev_databases"}
NUM_CANDIDATES_LIST=${NUM_CANDIDATES_LIST:-"3 5 7 9 11"}
NUM_WORKERS=${NUM_WORKERS:-8}

# Extract dataset name from DATA_PATH for naming
# If DATASET_NAME is not provided, infer it from DATA_PATH
# e.g., "data/bird/dev/mini_dev.json" -> "BIRD_Mini_Dev"
if [ -z "$DATASET_NAME" ]; then
    # Extract dataset type (bird/spider) and convert to uppercase
    DATASET_TYPE_UPPER=$(echo "$DATASET_TYPE" | tr '[:lower:]' '[:upper:]')
    
    # Extract file name without extension and convert to proper case
    FILE_NAME=$(basename "$DATA_PATH" .json)
    # Convert snake_case to Title_Case (e.g., mini_dev -> Mini_Dev)
    FILE_NAME_TITLE=$(echo "$FILE_NAME" | sed 's/_/ /g' | awk '{for(i=1;i<=NF;i++) $i=toupper(substr($i,1,1)) substr($i,2)}1' | sed 's/ /_/g')
    
    DATASET_NAME="${DATASET_TYPE_UPPER}_${FILE_NAME_TITLE}"
fi

# Create analysis output directory
ANALYSIS_DIR="${ARTIFACT_ROOT}/analysis"
mkdir -p "$ANALYSIS_DIR"

echo "=============================================================================="
echo "Batch Candidates Analysis: DPC vs SC Comparison"
echo "=============================================================================="
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_TYPE"
echo "Data Path: $DATA_PATH"
echo "Dataset Name: $DATASET_NAME"
echo "Candidate Counts: $NUM_CANDIDATES_LIST"
echo "Workers: $NUM_WORKERS"
echo "Output Directory: $ANALYSIS_DIR"
echo "=============================================================================="
echo ""

# Process each candidate count
for N in $NUM_CANDIDATES_LIST; do
    echo ""
    echo "========================================================================"
    echo "Processing N = $N candidates"
    echo "========================================================================"
    
    # Step 1: Generate baseline candidates with N candidates
    echo "[Step 1/3] Generating $N SQL candidates..."
    TEMP_CANDIDATES_FILE="${ARTIFACT_ROOT}/candidates/${MODEL_NAME}_${DATASET_NAME}_N${N}.json"
    
    NUM_CANDIDATES=$N \
    DATASET_TYPE="$DATASET_TYPE" \
    DATA_PATH="$DATA_PATH" \
    DB_ROOT_PATH="$DB_ROOT_PATH" \
    OUTPUT_PATH="$TEMP_CANDIDATES_FILE" \
    MODEL_NAME="$MODEL_NAME" \
    NUM_WORKERS="$NUM_WORKERS" \
    API_KEY="${API_KEY:-}" \
    BASE_URL="${BASE_URL:-}" \
    bash scripts/run_gen_baseline.sh
    
    if [ ! -f "$TEMP_CANDIDATES_FILE" ]; then
        echo "ERROR: Failed to generate candidates file: $TEMP_CANDIDATES_FILE"
        exit 1
    fi
    echo "✓ Generated candidates: $TEMP_CANDIDATES_FILE"
    
    # Step 2: Run SC Selection
    echo "[Step 2/3] Running SC Selection..."
    SC_OUTPUT_FILE="${ANALYSIS_DIR}/${MODEL_NAME}_${DATASET_NAME}_SC_N-Candidates-${N}.json"
    
    DATASET_TYPE="$DATASET_TYPE" \
    DATA_PATH="$DATA_PATH" \
    DB_ROOT_PATH="$DB_ROOT_PATH" \
    PRED_PATH="$TEMP_CANDIDATES_FILE" \
    OUTPUT_PATH="$SC_OUTPUT_FILE" \
    NUM_WORKERS="$NUM_WORKERS" \
    bash scripts/run_sc_selection.sh
    
    if [ ! -f "$SC_OUTPUT_FILE" ]; then
        echo "ERROR: Failed to generate SC output: $SC_OUTPUT_FILE"
        exit 1
    fi
    echo "✓ SC Selection completed: $SC_OUTPUT_FILE"
    
    # Step 3: Run DPC Selection
    echo "[Step 3/3] Running DPC Selection..."
    DPC_OUTPUT_FILE="${ANALYSIS_DIR}/${MODEL_NAME}_${DATASET_NAME}_DPC_N-Candidates-${N}.json"
    
    DATASET_TYPE="$DATASET_TYPE" \
    DATA_PATH="$DATA_PATH" \
    DB_ROOT_PATH="$DB_ROOT_PATH" \
    PRED_SQLS_PATH="$TEMP_CANDIDATES_FILE" \
    OUTPUT_PATH="$DPC_OUTPUT_FILE" \
    MODEL_NAME="$MODEL_NAME" \
    NUM_WORKERS="$NUM_WORKERS" \
    API_KEY="${API_KEY:-}" \
    BASE_URL="${BASE_URL:-}" \
    bash scripts/run_dpc_selection.sh
    
    if [ ! -f "$DPC_OUTPUT_FILE" ]; then
        echo "ERROR: Failed to generate DPC output: $DPC_OUTPUT_FILE"
        exit 1
    fi
    echo "✓ DPC Selection completed: $DPC_OUTPUT_FILE"
    
    echo ""
    echo "✓ Completed N = $N:"
    echo "  - Candidates: $TEMP_CANDIDATES_FILE"
    echo "  - SC Results:  $SC_OUTPUT_FILE"
    echo "  - DPC Results: $DPC_OUTPUT_FILE"
done

echo ""
echo "=============================================================================="
echo "All experiments completed!"
echo "=============================================================================="
echo "Results saved in: $ANALYSIS_DIR"
echo ""
echo "Generated files:"
for N in $NUM_CANDIDATES_LIST; do
    echo "  N=$N:"
    echo "    - ${ANALYSIS_DIR}/${MODEL_NAME}_${DATASET_NAME}_SC_N-Candidates-${N}.json"
    echo "    - ${ANALYSIS_DIR}/${MODEL_NAME}_${DATASET_NAME}_DPC_N-Candidates-${N}.json"
done
echo "=============================================================================="
