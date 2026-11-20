#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
PRIOR_DIR="${PRIOR_DIR:-results/pretrain_prior_data}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-results/pretrain_from_prior}"
RESET_OUTPUTS="${RESET_OUTPUTS:-true}"
SKIP_PRIOR="${SKIP_PRIOR:-false}"

if [[ "${RESET_OUTPUTS}" == "true" ]]; then
  rm -rf "${PRIOR_DIR}" "${CHECKPOINT_DIR}"
fi
mkdir -p "${PRIOR_DIR}" "${CHECKPOINT_DIR}"

if [[ "${SKIP_PRIOR}" != "true" ]]; then
  "${PYTHON_BIN}" -m tabicl.prior \
    --save_dir "${PRIOR_DIR}" \
    --num_batches 2 \
    --batch_size 4 \
    --batch_size_per_gp 1 \
    --min_features 2 \
    --max_features 4 \
    --max_classes 3 \
    --max_seq_len 64 \
    --min_train_size 0.5 \
    --max_train_size 0.8 \
    --prior_type mlp_scm \
    --n_jobs 1 \
    --num_threads_per_generate 1 \
    --device cpu
fi

"${PYTHON_BIN}" -m tabicl.train \
  --wandb_log False \
  --device cuda \
  --dtype float32 \
  --max_steps 2 \
  --batch_size 4 \
  --micro_batch_size 1 \
  --lr 1e-4 \
  --scheduler constant \
  --prior_dir "${PRIOR_DIR}" \
  --load_prior_start 0 \
  --delete_after_load False \
  --prior_device cpu \
  --embed_dim 32 \
  --col_num_blocks 1 \
  --col_nhead 2 \
  --col_num_inds 16 \
  --col_feature_group False \
  --row_num_blocks 1 \
  --row_nhead 2 \
  --row_num_cls 2 \
  --icl_num_blocks 1 \
  --icl_nhead 2 \
  --ff_factor 2 \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --save_temp_every 1 \
  --save_perm_every 2 \
  --max_checkpoints 2
