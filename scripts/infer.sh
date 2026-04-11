#!/bin/bash
set -euo pipefail

# ===========================================
# Environment Configuration (Machine-specific)
# ===========================================

export CUDA_VISIBLE_DEVICES="0"

MODEL_DIR="/path/to/wan2.1/Wan2.1-Fun-V1.1-1.3B-InP"
DATASET_DIR="/path/to/dataset"

TAG="exp_001"
EPOCH=9

# ===========================================
# Config Selection (Experiment-specific)
# ===========================================

CONFIG_FILE="configs/infer_noise_base.yaml"

# Checkpoint path (derived from TAG and EPOCH by default)
CKPT_OVERRIDE=""  # Leave empty to use default: Ckpt/${TAG}/epoch-${EPOCH}.safetensors

# Optional: metrics, chunk inference
ENABLE_METRICS=1  # 1=enable, 0=disable
CHUNK_INFER=0     # 1=enable, 0=disable

# ===========================================
# Launch Inference
# ===========================================

# Determine checkpoint path
if [ -n "${CKPT_OVERRIDE}" ]; then
  CKPT_PATH="${CKPT_OVERRIDE}"
else
  CKPT_PATH="Ckpt/${TAG}/epoch-${EPOCH}/epoch-${EPOCH}.safetensors"
fi

echo "Starting inference..."
echo "  Config: ${CONFIG_FILE}"
echo "  Model: ${MODEL_DIR}"
echo "  Dataset: ${DATASET_DIR}"
echo "  Checkpoint: ${CKPT_PATH}"

# Build command
CMD="python scripts/infer.py \
  --config ${CONFIG_FILE} \
  --model_paths ${MODEL_DIR} \
  --dataset_base_path ${DATASET_DIR} \
  --ckpt_path ${CKPT_PATH}"

# Add optional flags
if [ "${ENABLE_METRICS}" -eq 1 ]; then
  CMD="${CMD} --enable_metrics"
fi

if [ "${CHUNK_INFER}" -eq 1 ]; then
  CMD="${CMD} --chunk_infer"
fi

echo "Command: ${CMD}"
echo ""
eval "${CMD}"
