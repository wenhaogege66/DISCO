#!/bin/bash
#
# test_all_ml60.sh — DDBC test evaluation for ML-60 baselines
#
# Usage:
#   bash test_all_ml60.sh tiger
#   bash test_all_ml60.sh letter
#   bash test_all_ml60.sh tiger letter
#

set -euo pipefail

ROOT="/home/sjj/wenhao"
LOG_DIR="$ROOT/logs/ml60"
mkdir -p "$LOG_DIR"

RUN_TAG="start-ml60-v3"
DATASET="MovieLens-20M"
TIGER_CKPT="$ROOT/TIGER/ckpt/tiger-ml60-start-ml60-v3-Apr-30-2026_06-30-73b817.pth"
LETTER_CKPT_DIR="$ROOT/LETTER/LETTER-TIGER/ckpt/MovieLens-20M_${RUN_TAG}"
LETTER_CKPT_NAME="checkpoint-353070"

# ── TIGER test ────────────────────────────────────────────────────────────────
test_tiger() {
    local LOG_FILE="$LOG_DIR/test_tiger_$(date +%Y%m%d_%H%M%S).log"
    echo "──────────────────────────────────────────────────────────────"
    echo "  [TIGER] DDBC Test Evaluation"
    echo "  Log: $LOG_FILE"
    echo "──────────────────────────────────────────────────────────────"

    export WANDB_DISABLED=true
    (cd "$ROOT/TIGER" && \
     CUDA_VISIBLE_DEVICES=0 python -u main.py \
        --model=TIGER \
        --dataset=MovieLens20M \
        --category=len60 \
        --mode=test \
        --run_id="tiger-ml60-${RUN_TAG}" \
        --ckpt_path="$TIGER_CKPT" \
        --ddbc_eval=True \
        --ddbc_predict_nums="[30]" \
        --ddbc_multipliers="[19]" \
        --ddbc_seed=100 \
        --ddbc_predict_mode=single \
        --ddbc_dataset="MovieLens-20M/len60" \
        --ddbc_item_num=17188 \
        --eval_batch_size=8 \
        --use_fp16=True \
        --train_batch_size=256 \
    ) 2>&1 | tee "$LOG_FILE"

    echo "  [TIGER] 测试完成，log: $LOG_FILE"
}

# ── LETTER test ───────────────────────────────────────────────────────────────
test_letter() {
    local LOG_FILE="$LOG_DIR/test_letter_$(date +%Y%m%d_%H%M%S).log"
    echo "──────────────────────────────────────────────────────────────"
    echo "  [LETTER] DDBC Test Evaluation"
    echo "  Log: $LOG_FILE"
    echo "──────────────────────────────────────────────────────────────"

    CUDA_VISIBLE_DEVICES=0 \
    python -u "$ROOT/LETTER/LETTER-TIGER/test_letter.py" \
        2>&1 | tee "$LOG_FILE"

    echo "  [LETTER] 测试完成，log: $LOG_FILE"
}

# ── Main ──────────────────────────────────────────────────────────────────────

MODELS=("${@}")

if [ ${#MODELS[@]} -eq 0 ]; then
    echo "Usage: bash test_all_ml60.sh [tiger] [letter]"
    echo "  e.g.  bash test_all_ml60.sh tiger letter"
    exit 1
fi

for MODEL in "${MODELS[@]}"; do
    case "${MODEL,,}" in
        tiger)
            test_tiger
            ;;
        letter)
            test_letter
            ;;
        *)
            echo "Unknown model: $MODEL (supported: tiger, letter)"
            ;;
    esac
done

echo ""
echo "Done."
