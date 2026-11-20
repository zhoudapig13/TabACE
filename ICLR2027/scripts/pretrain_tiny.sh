#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
RUN_NAME="${RUN_NAME:-pretrain_tiny}"
PRIOR_DIR="${PRIOR_DIR:-results/${RUN_NAME}_prior_data}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-results/${RUN_NAME}}"
RESET_OUTPUTS="${RESET_OUTPUTS:-true}"
SKIP_PRIOR="${SKIP_PRIOR:-false}"

PRIOR_TYPE="${PRIOR_TYPE:-mlp_scm}"
MAX_STEPS="${MAX_STEPS:-100}"
NUM_BATCHES="${NUM_BATCHES:-${MAX_STEPS}}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-256}"
MAX_FEATURES="${MAX_FEATURES:-30}"
MAX_CLASSES="${MAX_CLASSES:-5}"

if [[ "${RESET_OUTPUTS}" == "true" ]]; then
  rm -rf "${PRIOR_DIR}" "${CHECKPOINT_DIR}"
fi
mkdir -p "${PRIOR_DIR}" "${CHECKPOINT_DIR}"

if [[ "${SKIP_PRIOR}" != "true" ]]; then
  "${PYTHON_BIN}" -m tabicl.prior \
    --save_dir "${PRIOR_DIR}" \
    --num_batches "${NUM_BATCHES}" \
    --batch_size "${BATCH_SIZE}" \
    --batch_size_per_gp 1 \
    --min_features 2 \
    --max_features "${MAX_FEATURES}" \
    --max_classes "${MAX_CLASSES}" \
    --max_seq_len "${MAX_SEQ_LEN}" \
    --min_train_size 0.5 \
    --max_train_size 0.9 \
    --prior_type "${PRIOR_TYPE}" \
    --n_jobs 1 \
    --num_threads_per_generate 1 \
    --device cpu
fi

"${PYTHON_BIN}" -m tabicl.train \
  --wandb_log False \
  --device cuda \
  --dtype float32 \
  --max_steps "${MAX_STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  --micro_batch_size "${MICRO_BATCH_SIZE}" \
  --lr 1e-4 \
  --scheduler cosine_warmup \
  --warmup_proportion 0.05 \
  --gradient_clipping 1.0 \
  --prior_dir "${PRIOR_DIR}" \
  --load_prior_start 0 \
  --delete_after_load False \
  --prior_device cpu \
  --embed_dim 64 \
  --col_num_blocks 2 \
  --col_nhead 4 \
  --col_num_inds 32 \
  --col_feature_group False \
  --row_num_blocks 2 \
  --row_nhead 4 \
  --row_num_cls 2 \
  --icl_num_blocks 2 \
  --icl_nhead 4 \
  --ff_factor 2 \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --save_temp_every 25 \
  --save_perm_every "${MAX_STEPS}" \
  --max_checkpoints 4
