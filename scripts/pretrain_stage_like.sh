#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
RUN_NAME="${RUN_NAME:-pretrain_stage_like}"
PRIOR_DIR="${PRIOR_DIR:-results/${RUN_NAME}_prior_data}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-results/${RUN_NAME}}"
RESET_OUTPUTS="${RESET_OUTPUTS:-true}"
SKIP_PRIOR="${SKIP_PRIOR:-false}"

PRIOR_TYPE="${PRIOR_TYPE:-mix_scm}"
MAX_STEPS="${MAX_STEPS:-5000}"
NUM_BATCHES="${NUM_BATCHES:-${MAX_STEPS}}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
MAX_FEATURES="${MAX_FEATURES:-100}"
MAX_CLASSES="${MAX_CLASSES:-10}"
PRIOR_BATCH_SIZE_PER_GP="${PRIOR_BATCH_SIZE_PER_GP:-4}"
N_JOBS="${N_JOBS:-1}"
NUM_THREADS_PER_GENERATE="${NUM_THREADS_PER_GENERATE:-1}"
RESUME_FROM="${RESUME_FROM:-0}"
SAVE_TEMP_EVERY="${SAVE_TEMP_EVERY:-250}"
SAVE_PERM_EVERY="${SAVE_PERM_EVERY:-1000}"
MAX_CHECKPOINTS="${MAX_CHECKPOINTS:-5}"

if [[ "${RESET_OUTPUTS}" == "true" ]]; then
  rm -rf "${PRIOR_DIR}" "${CHECKPOINT_DIR}"
fi
mkdir -p "${PRIOR_DIR}" "${CHECKPOINT_DIR}"

if [[ "${SKIP_PRIOR}" != "true" ]]; then
  "${PYTHON_BIN}" -m tabicl.prior \
    --save_dir "${PRIOR_DIR}" \
    --num_batches "${NUM_BATCHES}" \
    --resume_from "${RESUME_FROM}" \
    --batch_size "${BATCH_SIZE}" \
    --batch_size_per_gp "${PRIOR_BATCH_SIZE_PER_GP}" \
    --min_features 2 \
    --max_features "${MAX_FEATURES}" \
    --max_classes "${MAX_CLASSES}" \
    --max_seq_len "${MAX_SEQ_LEN}" \
    --min_train_size 0.1 \
    --max_train_size 0.9 \
    --prior_type "${PRIOR_TYPE}" \
    --n_jobs "${N_JOBS}" \
    --num_threads_per_generate "${NUM_THREADS_PER_GENERATE}" \
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
  --warmup_proportion 0.02 \
  --gradient_clipping 1.0 \
  --prior_dir "${PRIOR_DIR}" \
  --load_prior_start 0 \
  --delete_after_load False \
  --prior_device cpu \
  --embed_dim 128 \
  --col_num_blocks 3 \
  --col_nhead 4 \
  --col_num_inds 128 \
  --col_feature_group False \
  --row_num_blocks 3 \
  --row_nhead 8 \
  --row_num_cls 4 \
  --row_rope_base 100000 \
  --icl_num_blocks 12 \
  --icl_nhead 4 \
  --ff_factor 2 \
  --norm_first True \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --save_temp_every "${SAVE_TEMP_EVERY}" \
  --save_perm_every "${SAVE_PERM_EVERY}" \
  --max_checkpoints "${MAX_CHECKPOINTS}"
