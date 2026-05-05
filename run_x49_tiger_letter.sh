#!/bin/bash
#
# run_x49_tiger_letter.sh — Run TIGER and LETTER DDBC test evaluation with multiplier=49
#
# Usage:
#   bash run_x49_tiger_letter.sh
#

set -euo pipefail

ROOT="/home/sjj/wenhao"
LOG_DIR="$ROOT/logs/ml60/canditate49"
mkdir -p "$LOG_DIR"

TIGER_CKPT="$ROOT/TIGER/ckpt/tiger-ml60-start-ml60-v3-Apr-30-2026_06-30-73b817.pth"

# ── TIGER x49 ───────────────────────────────────────────────────────────────────
run_tiger_x49() {
    local LOG_FILE="$LOG_DIR/tiger_x49_$(date +%Y%m%d_%H%M%S).log"
    echo "=============================================================="
    echo "  [TIGER] DDBC x49 Test Evaluation"
    echo "  Log: $LOG_FILE"
    echo "=============================================================="

    export WANDB_DISABLED=true
    (cd "$ROOT/TIGER" && \
     CUDA_VISIBLE_DEVICES=0 python -u main.py \
        --model=TIGER \
        --dataset=MovieLens20M \
        --category=len60 \
        --mode=test \
        --run_id="tiger-ml60-start-ml60-v3" \
        --ckpt_path="$TIGER_CKPT" \
        --ddbc_eval=True \
        --ddbc_predict_nums="[30]" \
        --ddbc_multipliers="[49]" \
        --ddbc_seed=100 \
        --ddbc_predict_mode=single \
        --ddbc_dataset="MovieLens-20M/len60" \
        --ddbc_item_num=17188 \
        --eval_batch_size=8 \
        --use_fp16=True \
        --train_batch_size=256 \
    ) 2>&1 | tee "$LOG_FILE"

    echo "  [TIGER x49] Done. Log: $LOG_FILE"
}

# ── LETTER x49 ──────────────────────────────────────────────────────────────────
run_letter_x49() {
    local LOG_FILE="$LOG_DIR/letter_x49_$(date +%Y%m%d_%H%M%S).log"
    echo "=============================================================="
    echo "  [LETTER] DDBC x49 Test Evaluation"
    echo "  Log: $LOG_FILE"
    echo "=============================================================="

    CUDA_VISIBLE_DEVICES=0 \
    python -u "$ROOT/LETTER/LETTER-TIGER/test_letter_x49.py" \
        2>&1 | tee "$LOG_FILE"

    echo "  [LETTER x49] Done. Log: $LOG_FILE"
}

# ── Main ────────────────────────────────────────────────────────────────────────

echo ""
echo "=== Running TIGER x49 ==="
run_tiger_x49

echo ""
echo "=== Running LETTER x49 ==="
run_letter_x49

echo ""
echo "All x49 tests done."
