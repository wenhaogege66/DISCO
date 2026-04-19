#!/usr/bin/env bash
# GRU4Rec – evaluate best checkpoint against DDBC protocol
# Usage: bash GRU4Rec/scripts/eval_best.sh [/path/to/best_model.pt]
# Default checkpoint: GRU4Rec/outputs/yelp/best_model.pt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WSPACE_DIR="$(dirname "$SCRIPT_DIR")"   # GRU4Rec/
ROOT_DIR="$(dirname "$WSPACE_DIR")"     # wenhao/

CKPT="${1:-$WSPACE_DIR/outputs/yelp/best_model.pt}"

# Evaluation parameters (match DISCO default eval config)
PREDICT_NUMS="3"
CANDIDATE_MULTIPLIERS="19"
PREDICT_MODE="ar"   # ar for final eval (matches DreamRec eval_best.sh)
TOPK=1
VAL_SEED=100
DEVICE="cuda:0"

echo "=================================================="
echo "  GRU4Rec Eval — DDBC Protocol"
echo "  Checkpoint: $CKPT"
echo "  predict_nums=$PREDICT_NUMS  multipliers=$CANDIDATE_MULTIPLIERS"
echo "  predict_mode=$PREDICT_MODE  topk=$TOPK"
echo "=================================================="

cd "$ROOT_DIR"

conda run -n DDBC python GRU4Rec/eval_best.py \
    --ckpt                  "$CKPT"                   \
    --predict_nums          "$PREDICT_NUMS"            \
    --candidate_multipliers "$CANDIDATE_MULTIPLIERS"   \
    --predict_mode          "$PREDICT_MODE"            \
    --topk                  "$TOPK"                    \
    --val_seed              "$VAL_SEED"                \
    --device                "$DEVICE"

echo "=================================================="
echo "  评估完成"
echo "=================================================="
