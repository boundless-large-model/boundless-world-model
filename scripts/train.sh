#!/bin/bash
set -euo pipefail

export PYTHONPATH="${PWD}:${PYTHONPATH}"

# ===========================================
# Environment Configuration (Machine-specific)
# ===========================================

export CUDA_VISIBLE_DEVICES="0"

MODEL_DIR="/path/to/wan2.1/Wan2.1-Fun-V1.1-1.3B-InP"
DATASET_DIR="/path/to/dataset"

TAG="exp_001"

# ===========================================
# Config Selection (Experiment-specific)
# ===========================================

CONFIG_FILE="configs/train/train_action_noise.yaml"

# Optional: override output path from config
OUTPUT_OVERRIDE=""  # Leave empty to use config value

# RESUME_CKPT="Ckpt/exp_001/epoch-19.safetensors"
RESUME_CKPT=""

# ===========================================
# Launch Training
# ===========================================

echo "Starting training..."
echo "  Config: ${CONFIG_FILE}"
echo "  Model: ${MODEL_DIR}"
echo "  Dataset: ${DATASET_DIR}"
echo "  Tag: ${TAG}"

# Build command
CMD="python scripts/train.py \
  --config ${CONFIG_FILE} \
  --model_paths ${MODEL_DIR} \
  --dataset_base_path ${DATASET_DIR}"

# Add optional overrides
if [ -n "${OUTPUT_OVERRIDE}" ]; then
  CMD="${CMD} --output_path ${OUTPUT_OVERRIDE}"
fi

if [ -n "${RESUME_CKPT}" ]; then
  CMD="${CMD} --ckpt_path ${RESUME_CKPT}"
fi

echo "Command: ${CMD}"
echo ""
eval "accelerate launch ${CMD}"