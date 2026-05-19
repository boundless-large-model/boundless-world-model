#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCAL_CONFIG="${LOCAL_CONFIG:-${SCRIPT_DIR}/local.sh}"

if [[ -f "${LOCAL_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${LOCAL_CONFIG}"
fi

cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CONFIG_PATH="${CONFIG_PATH:-configs/infer/robotwin_ti2v_720p.yaml}"
MODEL_PATHS="${MODEL_PATHS:-models/Wan2.2-TI2V-5B}"
CKPT_PATH="${CKPT_PATH:-checkpoints/action_model.safetensors}"
DATASET_BASE_PATH="${DATASET_BASE_PATH:-data/RoboTwin2.0_lerobot}"
DATASET_METADATA_PATH="${DATASET_METADATA_PATH:-data/RoboTwin2.0_lerobot/metadata/episodes_val.jsonl}"
ACTION_STAT_PATH="${ACTION_STAT_PATH:-data/RoboTwin2.0_lerobot/metadata/stat.json}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/infer}"
MAX_SAMPLES="${MAX_SAMPLES:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" scripts/infer.py \
  --config "${CONFIG_PATH}" \
  --model_paths "${MODEL_PATHS}" \
  --ckpt_path "${CKPT_PATH}" \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${DATASET_METADATA_PATH}" \
  --action_stat_path "${ACTION_STAT_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --max_samples "${MAX_SAMPLES}"
