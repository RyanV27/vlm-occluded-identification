#!/usr/bin/env bash

# =========================
# Configuration (edit here)
# =========================
MODEL_NAMES=(
    # "Qwen3-VL" 
    # "FT-Qwen3-VL" 
    # "Qwen2.5-VL" 
    # "FT-Qwen2.5-VL" 
    # "Llama-3.2" 
    # "FT-Llama-3.2" 
    # "Gemma3" 
    # "FT-Gemma3"
    "Molmo2"
    # "InternVL3.5"
) 
OUT_PATH="./runs/run2"
OCC_LEVELS=("no" "low")     # "no" "low" "medium" "high"
ALL_OCC_LEVELS=("no" "low" "medium" "high")

# =========================
# Execution
# =========================

for model in "${MODEL_NAMES[@]}"; do
  
  # ===== Activate Conda Environment =====
  if [ "$model" == "InternVL3.5" ]; then
    CONDA_ENV="internvl_venv"
  elif [ "$model" == "Molmo2" ]; then
    CONDA_ENV="molmo_venv"
  else
    CONDA_ENV="vqa_clutter_venv"
  fi
    
  echo "Activating conda environment: $CONDA_ENV"
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"

  # Calling VLM
  for level in "${OCC_LEVELS[@]}"; do
    echo "Model name=${model}"
    echo "Running with occ_level=${level}"
    python call_vlm.py \
      --model_name "${model}" \
      --occ_level "${level}" \
      --out_path "${OUT_PATH}"
  done

  # Calculating results
  python calculate_alias_results.py \
    --model_name "${model}" \
    --occ_level "${ALL_OCC_LEVELS[@]}" \
    --out_path "${OUT_PATH}"

done
