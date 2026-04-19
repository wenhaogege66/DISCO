#!/usr/bin/env bash
# GRU4Rec – Yelp training with DDBC evaluation protocol
# Run from workspace root:  bash GRU4Rec/scripts/train_yelp.sh

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WSPACE_DIR="$(dirname "$SCRIPT_DIR")"   # GRU4Rec/
ROOT_DIR="$(dirname "$WSPACE_DIR")"     # wenhao/

# ── Hyperparameters ────────────────────────────────────────────────────────
LAYERS="64"
LOSS="cross-entropy"
EPOCHS=50
BATCH_SIZE=512
LR=0.05
MOMENTUM=0.0
N_SAMPLE=2048
SAMPLE_ALPHA=0.5
DROPOUT_P_EMBED=0.0
DROPOUT_P_HIDDEN=0.0
CONSTRAINED_EMBEDDING=1

# ── Evaluation ─────────────────────────────────────────────────────────────
EVAL_FREQ=5
PREDICT_NUMS="3"
CANDIDATE_MULTIPLIERS="19"
PREDICT_MODE="single"   # single (fast) during training; change to ar for final eval
TOPK=1
VAL_SEED=100

# ── Device ─────────────────────────────────────────────────────────────────
DEVICE="cuda:0"
SEED=42

# ── Launch ─────────────────────────────────────────────────────────────────
cd "$ROOT_DIR"

conda run -n DDBC python GRU4Rec/train_yelp.py \
    --layers              "$LAYERS" \
    --loss                "$LOSS" \
    --epochs              "$EPOCHS" \
    --batch_size          "$BATCH_SIZE" \
    --lr                  "$LR" \
    --momentum            "$MOMENTUM" \
    --n_sample            "$N_SAMPLE" \
    --sample_alpha        "$SAMPLE_ALPHA" \
    --dropout_p_embed     "$DROPOUT_P_EMBED" \
    --dropout_p_hidden    "$DROPOUT_P_HIDDEN" \
    --constrained_embedding "$CONSTRAINED_EMBEDDING" \
    --eval_freq           "$EVAL_FREQ" \
    --predict_nums        "$PREDICT_NUMS" \
    --candidate_multipliers "$CANDIDATE_MULTIPLIERS" \
    --predict_mode        "$PREDICT_MODE" \
    --topk                "$TOPK" \
    --val_seed            "$VAL_SEED" \
    --device              "$DEVICE" \
    --seed                "$SEED"
