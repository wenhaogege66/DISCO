#!/usr/bin/env bash
# =============================================================================
# test_all.sh — 一键测试脚本（DISCO + 7 baselines）
#
# 用法:
#   RUN_TAG=exp_yelp_v1 bash test_all.sh [模型列表]
#
# 示例:
#   RUN_TAG=exp_yelp_v1 bash test_all.sh           # 测试全部
#   RUN_TAG=exp_yelp_v1 bash test_all.sh disco sasrec
#
# 可选模型名: disco  dreamrec  difurec  gru4rec  sasrec  bert4rec  tiger  letter
#
# RUN_TAG 必须与训练时一致，用于定位各模型的 ckpt 默认路径和 log 文件名。
# 也可在下方 "Checkpoint 路径" 区域手动覆盖每个模型的 ckpt 路径。
#
# =============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 批次标签（必须与训练时一致）──────────────────────────────────────────────
if [ -z "${RUN_TAG:-}" ]; then
    echo "[ERROR] 请指定 RUN_TAG，与训练时保持一致："
    echo "  RUN_TAG=exp_yelp_v1 bash test_all.sh"
    exit 1
fi
echo "================================================================"
echo "  RUN_TAG = $RUN_TAG"
echo "================================================================"

# ── 选择要测试的模型 ──────────────────────────────────────────────────────────
if [ $# -eq 0 ]; then
    MODELS=(disco dreamrec difurec gru4rec sasrec bert4rec tiger letter)
else
    MODELS=("$@")
fi

# =============================================================================
# Checkpoint 路径（留空则使用各模型默认路径）
# =============================================================================

# DISCO: outputs/yelp/<date>/<time>/checkpoints/best.ckpt
DISCO_CKPT=""

# DreamRec: outputs/yelp/<run_name>/best_model.pt
DREAMREC_CKPT=""

# DiffuRec: outputs/yelp/<desc>/best_model.pt
DIFUREC_CKPT=""

# GRU4Rec: outputs/yelp/<run_tag>/best_model.pt
GRU4REC_CKPT=""

# SASRec: SASRec/python/Yelp_<train_dir>/best_model_ep*.pth
# 需要指定 train_dir（与训练时 --train_dir 一致）
SASREC_TRAIN_DIR=""   # 例: "run_20260419_120000"

# BERT4Rec: experiments/bert4rec-yelp-<run_tag>_<date>/models/best_model.pth
BERT4REC_CKPT=""

# TIGER: ckpt/<run_id>.pth（run_id = tiger-yelp-<run_tag>）
TIGER_CKPT=""

# LETTER: LETTER-TIGER/ckpt/Yelp_<run_tag>/
LETTER_CKPT_DIR=""

# =============================================================================
# 测试参数（与训练保持一致）
# =============================================================================

PREDICT_NUMS="3"
MULTIPLIERS="19"
SEED=100

DISCO_GPU="0,1"
DREAMREC_GPU=0
DIFUREC_GPU=1
GRU4REC_GPU="cuda:0"
SASREC_GPU=1
BERT4REC_GPU=1
TIGER_GPU=0
LETTER_GPU=1

# =============================================================================
# 测试函数
# =============================================================================

test_disco() {
    echo ""
    echo "──────────────────────────────────────────────────────────────"
    echo "  [DISCO] 开始测试  RUN_TAG=$RUN_TAG"
    echo "──────────────────────────────────────────────────────────────"
    local LOG_DIR="$ROOT/DISCO/logs"
    local LOG_FILE="$LOG_DIR/test_${RUN_TAG}.log"
    mkdir -p "$LOG_DIR"

    if [ -z "$DISCO_CKPT" ]; then
        echo "  [DISCO] 未指定 DISCO_CKPT，请在脚本中设置后重试"
        return 1
    fi

    export CUDA_VISIBLE_DEVICES=$DISCO_GPU
    export PYTHONPATH="$ROOT/DISCO:$PYTHONPATH"

    python "$ROOT/DISCO/main.py" \
        mode=rec_eval \
        evaluator.candidate_multiplier=$MULTIPLIERS \
        evaluator.allow_duplicate_items=true \
        training.layer_loss_weights.enabled=false \
        loader.batch_size=32 \
        loader.eval_batch_size=1 \
        dataset=Yelp \
        model=small \
        model.hidden_size=64 \
        parameterization=subs \
        backbone=dit \
        rq_n_codebooks=3 \
        rq_codebook_size=256 \
        model.length=52 \
        seq_len=10 \
        swap_ratio=0 \
        evaluator.dataset=Yelp \
        eval.predict_num_items=$PREDICT_NUMS \
        eval.checkpoint_path="$DISCO_CKPT" \
        sampling.cfg_enabled=true \
        sampling.cfg_encoder=true \
        sampling.cfg_w=2.0 \
        2>&1 | tee "$LOG_FILE"
    echo "  [DISCO] 完成，log: $LOG_FILE"
}

test_dreamrec() {
    echo ""
    echo "──────────────────────────────────────────────────────────────"
    echo "  [DreamRec] 开始测试  RUN_TAG=$RUN_TAG"
    echo "──────────────────────────────────────────────────────────────"
    local LOG_DIR="$ROOT/DreamRec/logs"
    local LOG_FILE="$LOG_DIR/test_${RUN_TAG}.log"
    mkdir -p "$LOG_DIR"

    # DreamRec 没有独立 test 脚本，训练结束后自动在 best_model 上跑 test
    # 若需要单独重测，需手动调用 DreamRec.py --mode test（如已支持）
    echo "  [DreamRec] 训练结束时已自动完成 test 评估，结果见训练 log"
    echo "  若需重测，请手动指定 ckpt 并调用 DreamRec.py"
    if [ -n "$DREAMREC_CKPT" ]; then
        export CUDA_VISIBLE_DEVICES=$DREAMREC_GPU
        python -u "$ROOT/DreamRec/DreamRec.py" \
            --mode         test \
            --data         yelp \
            --predict_nums "$PREDICT_NUMS" \
            --candidate_multipliers "$MULTIPLIERS" \
            --topk         1 \
            --ddbc_data_dir "$ROOT/DreamRec/data/yelp" \
            --save_dir     "$(dirname "$DREAMREC_CKPT")" \
            --descri       "dreamrec-yelp-${RUN_TAG}" \
            2>&1 | tee "$LOG_FILE"
    fi
}

test_difurec() {
    echo ""
    echo "──────────────────────────────────────────────────────────────"
    echo "  [DiffuRec] 开始测试  RUN_TAG=$RUN_TAG"
    echo "──────────────────────────────────────────────────────────────"
    local LOG_DIR="$ROOT/DiffuRec/logs"
    local LOG_FILE="$LOG_DIR/test_${RUN_TAG}.log"
    mkdir -p "$LOG_DIR"

    local DESC="difurec-yelp-${RUN_TAG}"
    local SAVE_DIR="${DIFUREC_CKPT:-$ROOT/DiffuRec/outputs/yelp/$DESC}"

    CUDA_VISIBLE_DEVICES=$DIFUREC_GPU python "$ROOT/DiffuRec/src/main.py" \
        --mode              test \
        --dataset           yelp \
        --data_path         "$ROOT/DiffuRec/data/yelp/dataset.pkl" \
        --log_file          "$ROOT/DiffuRec/log/" \
        --max_len           10 \
        --hidden_size       64 \
        --num_blocks        4 \
        --diffusion_steps   32 \
        --noise_schedule    trunc_lin \
        --schedule_sampler_name lossaware \
        --lambda_uncertainty 0.001 \
        --predict_nums      "$PREDICT_NUMS" \
        --candidate_multipliers "$MULTIPLIERS" \
        --topk              1 \
        --ddbc_data_dir     "$ROOT/DreamRec/data/yelp" \
        --save_dir          "$SAVE_DIR" \
        --description       "$DESC" \
        --random_seed       1997 \
        2>&1 | tee "$LOG_FILE"
    echo "  [DiffuRec] 完成，log: $LOG_FILE"
}

test_gru4rec() {
    echo ""
    echo "──────────────────────────────────────────────────────────────"
    echo "  [GRU4Rec] 开始测试  RUN_TAG=$RUN_TAG"
    echo "──────────────────────────────────────────────────────────────"
    local LOG_DIR="$ROOT/GRU4Rec/logs"
    local LOG_FILE="$LOG_DIR/test_${RUN_TAG}.log"
    mkdir -p "$LOG_DIR"

    local OUTPUT_DIR="${GRU4REC_CKPT:-$ROOT/GRU4Rec/outputs/yelp/$RUN_TAG}"

    conda run -n DDBC python "$ROOT/GRU4Rec/train_yelp.py" \
        --mode              test \
        --predict_nums      "$PREDICT_NUMS" \
        --candidate_multipliers "$MULTIPLIERS" \
        --predict_mode      ar \
        --topk              1 \
        --val_seed          $SEED \
        --device            "$GRU4REC_GPU" \
        --output_dir        "$OUTPUT_DIR" \
        --log_file          "$LOG_FILE" \
        2>&1 | tee "$LOG_FILE"
    echo "  [GRU4Rec] 完成，log: $LOG_FILE"
}

test_sasrec() {
    echo ""
    echo "──────────────────────────────────────────────────────────────"
    echo "  [SASRec] 开始测试  RUN_TAG=$RUN_TAG"
    echo "──────────────────────────────────────────────────────────────"
    local LOG_DIR="$ROOT/SASRec/logs"
    local LOG_FILE="$LOG_DIR/test_${RUN_TAG}.log"
    mkdir -p "$LOG_DIR"

    local TRAIN_DIR="${SASREC_TRAIN_DIR:-$RUN_TAG}"

    CUDA_VISIBLE_DEVICES=$SASREC_GPU \
    python "$ROOT/SASRec/python/main_disco.py" \
        --mode=test \
        --dataset=Yelp \
        --train_dir="$TRAIN_DIR" \
        --maxlen=10 \
        --hidden_units=64 \
        --num_blocks=2 \
        --num_heads=1 \
        --ddbc_predict_nums="[$PREDICT_NUMS]" \
        --ddbc_multipliers="[$MULTIPLIERS]" \
        --ddbc_seed=$SEED \
        2>&1 | tee "$LOG_FILE"
    echo "  [SASRec] 完成，log: $LOG_FILE"
}

test_bert4rec() {
    echo ""
    echo "──────────────────────────────────────────────────────────────"
    echo "  [BERT4Rec] 开始测试  RUN_TAG=$RUN_TAG"
    echo "──────────────────────────────────────────────────────────────"
    local LOG_DIR="$ROOT/BERT4Rec/logs"
    local LOG_FILE="$LOG_DIR/test_${RUN_TAG}.log"
    mkdir -p "$LOG_DIR"

    local CKPT_ARG=""
    [ -n "$BERT4REC_CKPT" ] && CKPT_ARG="--test_model_path $BERT4REC_CKPT"

    CUDA_VISIBLE_DEVICES=$BERT4REC_GPU \
    python "$ROOT/BERT4Rec/main.py" \
        --mode                 test \
        --template             train_bert_yelp \
        --dataset_code         yelp \
        --device               cuda \
        --device_idx           $BERT4REC_GPU \
        --bert_max_len         10 \
        --bert_hidden_units    64 \
        --bert_num_blocks      2 \
        --bert_num_heads       4 \
        --bert_dropout         0.1 \
        --bert_mask_prob       0.15 \
        --model_init_seed      0 \
        --predict_nums         "$PREDICT_NUMS" \
        --candidate_multipliers "$MULTIPLIERS" \
        --topk                 1 \
        --random_seed          $SEED \
        --ddbc_data_dir        "$ROOT/DreamRec/data/yelp" \
        --experiment_dir       "$ROOT/BERT4Rec/experiments" \
        --experiment_description "bert4rec-yelp-${RUN_TAG}" \
        $CKPT_ARG \
        2>&1 | tee "$LOG_FILE"
    echo "  [BERT4Rec] 完成，log: $LOG_FILE"
}

test_tiger() {
    echo ""
    echo "──────────────────────────────────────────────────────────────"
    echo "  [TIGER] 开始测试  RUN_TAG=$RUN_TAG"
    echo "──────────────────────────────────────────────────────────────"
    local LOG_DIR="$ROOT/TIGER/logs"
    local LOG_FILE="$LOG_DIR/test_${RUN_TAG}.log"
    mkdir -p "$LOG_DIR"

    local CKPT_ARG=""
    [ -n "$TIGER_CKPT" ] && CKPT_ARG="--ckpt_path=$TIGER_CKPT"

    CUDA_VISIBLE_DEVICES=$TIGER_GPU \
    python "$ROOT/TIGER/main.py" \
        --mode=test \
        --model=TIGER \
        --dataset=Yelp \
        --run_id="tiger-yelp-${RUN_TAG}" \
        --ddbc_eval=True \
        --ddbc_predict_nums="[$PREDICT_NUMS]" \
        --ddbc_multipliers="[$MULTIPLIERS]" \
        --ddbc_seed=$SEED \
        $CKPT_ARG \
        2>&1 | tee "$LOG_FILE"
    echo "  [TIGER] 完成，log: $LOG_FILE"
}

test_letter() {
    echo ""
    echo "──────────────────────────────────────────────────────────────"
    echo "  [LETTER] 开始测试  RUN_TAG=$RUN_TAG"
    echo "──────────────────────────────────────────────────────────────"
    local LOG_DIR="$ROOT/LETTER/logs"
    local LOG_FILE="$LOG_DIR/test_${RUN_TAG}.log"
    mkdir -p "$LOG_DIR"

    local CKPT_DIR="${LETTER_CKPT_DIR:-$ROOT/LETTER/LETTER-TIGER/ckpt/Yelp_${RUN_TAG}}"

    CUDA_VISIBLE_DEVICES=$LETTER_GPU \
    python "$ROOT/LETTER/LETTER-TIGER/test_disco.py" \
        --dataset Yelp \
        --data_path "$ROOT/LETTER/data" \
        --base_model "$ROOT/LETTER/LETTER-TIGER/ckpt/TIGER" \
        --ckpt_path "$CKPT_DIR" \
        --index_file .index.json \
        --max_his_len 20 \
        --ddbc_predict_nums $PREDICT_NUMS \
        --ddbc_multipliers $MULTIPLIERS \
        --ddbc_seed $SEED \
        2>&1 | tee "$LOG_FILE"
    echo "  [LETTER] 完成，log: $LOG_FILE"
}

# =============================================================================
# 主流程
# =============================================================================

SUMMARY=()

for MODEL in "${MODELS[@]}"; do
    case "${MODEL,,}" in
        disco)     test_disco    && SUMMARY+=("  disco     OK") || SUMMARY+=("  disco     FAILED") ;;
        dreamrec)  test_dreamrec && SUMMARY+=("  dreamrec  OK") || SUMMARY+=("  dreamrec  FAILED") ;;
        difurec|diffurec) test_difurec && SUMMARY+=("  difurec   OK") || SUMMARY+=("  difurec   FAILED") ;;
        gru4rec)   test_gru4rec  && SUMMARY+=("  gru4rec   OK") || SUMMARY+=("  gru4rec   FAILED") ;;
        sasrec)    test_sasrec   && SUMMARY+=("  sasrec    OK") || SUMMARY+=("  sasrec    FAILED") ;;
        bert4rec)  test_bert4rec && SUMMARY+=("  bert4rec  OK") || SUMMARY+=("  bert4rec  FAILED") ;;
        tiger)     test_tiger    && SUMMARY+=("  tiger     OK") || SUMMARY+=("  tiger     FAILED") ;;
        letter)    test_letter   && SUMMARY+=("  letter    OK") || SUMMARY+=("  letter    FAILED") ;;
        *) echo "  [WARN] 未知模型: $MODEL，跳过" ;;
    esac
done

echo ""
echo "================================================================"
echo "  测试完成汇总  RUN_TAG=$RUN_TAG"
for line in "${SUMMARY[@]}"; do echo "$line"; done
echo "================================================================"
