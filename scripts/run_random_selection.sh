#!/bin/bash

# --- Configuration ---
CANDIDATES_PATH=${CANDIDATES_PATH:-"results/candidates/Qwen2.5-Coder-7B-Instruct_BIRD_Mini_Dev.json"}
OUTPUT_PATH=${OUTPUT_PATH:-"results/selected/Qwen2.5-Coder-7B-Instruct_BIRD_Mini_Dev_Random.json"}
SEED=${SEED:-42}

echo "Starting Random Selection Baseline..."
echo "Input: $CANDIDATES_PATH"
echo "Output: $OUTPUT_PATH"
echo "Seed: $SEED"

# Run the script
python baseline/run_random_selection.py \
    --pred_path "$CANDIDATES_PATH" \
    --output_path "$OUTPUT_PATH" \
    --seed "$SEED"

