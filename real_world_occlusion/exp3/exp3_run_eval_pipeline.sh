#!/bin/bash

# ===== Configuration =====
OUT_PATH="/home/rsvargh2/Research/occlusion/real_world_occlusion/runs/exp3/run10/"
DATA_PATH="/scratch/rsvargh2/occlusion_images/exp3"
LOG_FILE="errors_run10.log"

MODEL_NAMES=(
    # "Qwen2.5-VL"
    # "FT-Qwen2.5-VL"
    # "Gemma3"
    # "FT-Gemma3"
    # "Llama-3.2"
    # "FT-Llama-3.2"
    # "Qwen3-VL"
    # "FT-Qwen3-VL"
    # "InternVL3.5"
    # "Molmo2"
    "COCO-FT-Llama-3.2"
    "COCO-FT-Gemma3"
    "COCO-FT-Qwen2.5-VL"
    "COCO-FT-Qwen3-VL"
)

# ===== Helpers =====
> "$LOG_FILE"  # Clear log at start

run_cmd() {
    local model="$1"
    local cmd="$2"
 
    echo "Running: $cmd"
    error_output=$(eval "$cmd" 2>&1 >/dev/tty)
    if [ $? -ne 0 ]; then
        echo "[ERROR] Model: $model" >> "$LOG_FILE"
        echo "  Command: $cmd" >> "$LOG_FILE"
        echo "  Error: $error_output" >> "$LOG_FILE"
        echo "" >> "$LOG_FILE"
    fi
}

# ===== Main Loop =====
for MODEL in "${MODEL_NAMES[@]}"; do
    echo ""
    echo "===== Processing: $MODEL ====="

    # ===== Activate Conda Environment =====
    if [ "$MODEL" == "InternVL3.5" ]; then
        CONDA_ENV="internvl_venv"
    elif [ "$MODEL" == "Molmo2" ]; then
        CONDA_ENV="molmo_venv"
    else
        CONDA_ENV="vqa_clutter_venv"
    fi

    echo "Activating conda environment: $CONDA_ENV"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"

    run_cmd "$MODEL" "python call_vlm.py --model_name \"$MODEL\" --out_path \"$OUT_PATH\" --data_path \"$DATA_PATH\""
    run_cmd "$MODEL" "python exp3_calculate_alias_results.py --model_name \"$MODEL\" --out_path \"$OUT_PATH\""
done

# ===== Summary =====
echo ""
if [ -s "$LOG_FILE" ]; then
    echo "===== Errors were logged to $LOG_FILE ====="
    cat "$LOG_FILE"
else
    echo "===== All commands completed successfully ====="
fi