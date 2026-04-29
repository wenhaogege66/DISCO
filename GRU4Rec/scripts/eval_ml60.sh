#!/usr/bin/env bash
# GRU4Rec – evaluate best ML-60 checkpoint against DDBC protocol
# Usage: bash GRU4Rec/scripts/eval_ml60.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WSPACE_DIR="$(dirname "$SCRIPT_DIR")"
ROOT_DIR="$(dirname "$WSPACE_DIR")"

CKPT="$WSPACE_DIR/outputs/ml60/start-ml60/best_model.pt"

# ML-60 eval config
PREDICT_NUMS="30"
CANDIDATE_MULTIPLIERS="19"
PREDICT_MODE="ar"
TOPK=1
VAL_SEED=42
DEVICE="cuda:0"

echo "=================================================="
echo "  GRU4Rec ML-60 Eval — DDBC Protocol"
echo "  Checkpoint: $CKPT"
echo "  predict_nums=$PREDICT_NUMS  multipliers=$CANDIDATE_MULTIPLIERS"
echo "  predict_mode=$PREDICT_MODE  topk=$TOPK"
echo "=================================================="

cd "$ROOT_DIR"

conda run -n DDBC python GRU4Rec/eval_ml60.py \
    --ckpt                  "$CKPT"                   \
    --data_dir              "$WSPACE_DIR/data/ml60"    \
    --disco_dir             "$ROOT_DIR/DISCO/datasets/MovieLens-20M/len60" \
    --dreamrec_dir          "$ROOT_DIR/DreamRec/data/ml60" \
    --predict_nums          "$PREDICT_NUMS"            \
    --candidate_multipliers "$CANDIDATE_MULTIPLIERS"   \
    --predict_mode          "$PREDICT_MODE"            \
    --topk                  "$TOPK"                    \
    --val_seed              "$VAL_SEED"                \
    --device                "$DEVICE"

echo "=================================================="
echo "  ML-60 评估完成"
echo "=================================================="
