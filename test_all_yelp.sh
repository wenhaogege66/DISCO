#!/usr/bin/env bash
# =============================================================================
# test_all.sh — 一键测试脚本（DISCO + 7 baselines）
#
# 用法:
#   DATASET=yelp RUN_TAG=exp_yelp_v1 bash test_all.sh [模型列表]
#
# 示例:
#   DATASET=yelp RUN_TAG=exp_yelp_v1 bash test_all.sh           # 测试全部
#   DATASET=yelp RUN_TAG=exp_yelp_v1 bash test_all.sh disco sasrec
#
# 可选模型名: disco  dreamrec  difurec  gru4rec  sasrec  bert4rec  tiger  letter
#
# 命名规范（与 train_all.sh 一致）：
#   log  : <Model>/logs/test_<DATASET>_<RUN_TAG>.log
#   ckpt : <Model>/outputs/<DATASET>/<RUN_TAG>/best_model.*
#
# RUN_TAG 必须与训练时一致，用于定位各模型的 ckpt 默认路径和 log 文件名。
# 也可在下方 "Checkpoint 路径" 区域手动覆盖每个模型的 ckpt 路径。
#
# =============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 数据集 ────────────────────────────────────────────────────────────────────
DATASET="${DATASET:-yelp}"
DATASET_CAP="${DATASET^}"

# ── 批次标签（必须与训练时一致）──────────────────────────────────────────────
if [ -z "${RUN_TAG:-}" ]; then
    echo "[ERROR] 请指定 RUN_TAG，与训练时保持一致："
    echo "  DATASET=$DATASET RUN_TAG=exp_yelp_v1 bash test_all.sh"
    exit 1
fi

# ── 统一日志目录 ────────────────────────────────────────────────────────────
LOG_DIR="$ROOT/logs/$DATASET"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "  DATASET  = $DATASET"
echo "  RUN_TAG  = $RUN_TAG"
echo "  LOG_DIR  = $LOG_DIR"
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

# DISCO: outputs/<dataset>/<date>/<time>/checkpoints/best.ckpt
DISCO_CKPT="${DISCO_CKPT:-}"

# DreamRec: outputs/<dataset>/<run_tag>/best_model.pt
DREAMREC_CKPT="${DREAMREC_CKPT:-}"

# DiffuRec: outputs/<dataset>/<run_tag>/best_model.pt
DIFUREC_CKPT=""

# GRU4Rec: outputs/<dataset>/<run_tag>/best_model.pt
GRU4REC_CKPT=""

# SASRec: SASRec/python/<Dataset>_<train_dir>/SASRec.epoch=*.pth
# train_dir is just RUN_TAG (no dataset prefix) for runs before train_all.sh was updated
SASREC_TRAIN_DIR="${SASREC_TRAIN_DIR:-${RUN_TAG}}"

# BERT4Rec: experiments/bert4rec-<dataset>-<run_tag>/models/best_model.pth
BERT4REC_CKPT="${BERT4REC_CKPT:-}"

# TIGER: ckpt/tiger-<dataset>-<run_tag>-*.pth
TIGER_CKPT="${TIGER_CKPT:-}"

# LETTER: LETTER-TIGER/ckpt/<Dataset>_<run_tag>/
LETTER_CKPT_DIR=""

# =============================================================================
# 测试参数（与训练保持一致）
# =============================================================================

PREDICT_NUMS="3"
MULTIPLIERS="19"
SEED=100
LOG_STAMP="${LOG_STAMP:-$(date +%Y%m%d_%H%M%S)}"

DISCO_GPU="0,1"
DREAMREC_GPU=0
DIFUREC_GPU=1
GRU4REC_GPU="cuda:0"
SASREC_GPU=1
BERT4REC_GPU=1
TIGER_GPU=0
LETTER_GPU=1

declare -A MODEL_LOG_FILES

# =============================================================================
# 测试函数
# =============================================================================

test_disco() {
    echo ""
    echo "──────────────────────────────────────────────────────────────"
    echo "  [DISCO] 开始测试  DATASET=$DATASET  RUN_TAG=$RUN_TAG"
    echo "──────────────────────────────────────────────────────────────"
    local LOG_FILE="$LOG_DIR/test_disco_${RUN_TAG}_${LOG_STAMP}.log"

    if [ -z "$DISCO_CKPT" ]; then
        echo "  [DISCO] 未指定 DISCO_CKPT，请通过环境变量 DISCO_CKPT=<path> 设置后重试"
        return 1
    fi

    export CUDA_VISIBLE_DEVICES=$DISCO_GPU
    export PYTHONPATH="$ROOT/DISCO:${PYTHONPATH:-}"

    conda run -n DDBC python "$ROOT/DISCO/main.py" \
        mode=rec_eval \
        evaluator.candidate_multiplier=$MULTIPLIERS \
        evaluator.allow_duplicate_items=true \
        training.layer_loss_weights.enabled=false \
        loader.batch_size=32 \
        loader.eval_batch_size=1 \
        data=$DATASET \
        dataset=$DATASET_CAP \
        model=small \
        model.hidden_size=64 \
        parameterization=subs \
        backbone=dit \
        rq_n_codebooks=3 \
        rq_codebook_size=256 \
        model.length=52 \
        seq_len=10 \
        swap_ratio=0 \
        evaluator.dataset=$DATASET_CAP \
        eval.predict_num_items=$PREDICT_NUMS \
        eval.checkpoint_path="$DISCO_CKPT" \
        evaluator.topk=[1,2,3] \
        sampling.cfg_enabled=true \
        sampling.cfg_encoder=true \
        sampling.cfg_w=2.0 \
        2>&1 | tee "$LOG_FILE"
    echo "  [DISCO] 完成，log: $LOG_FILE"
}

test_dreamrec() {
    echo ""
    echo "──────────────────────────────────────────────────────────────"
    echo "  [DreamRec] 开始测试  DATASET=$DATASET  RUN_TAG=$RUN_TAG"
    echo "──────────────────────────────────────────────────────────────"
    local LOG_FILE="$LOG_DIR/test_dreamrec_${RUN_TAG}_${LOG_STAMP}.log"
    MODEL_LOG_FILES[dreamrec]="$LOG_FILE"

    # DreamRec 没有独立 test 脚本，训练结束后自动在 best_model 上跑 test
    # 若需要单独重测，需手动调用 DreamRec.py --mode test（如已支持）
    echo "  [DreamRec] 训练结束时已自动完成 test 评估，结果见训练 log"
    echo "  若需重测，请手动指定 ckpt 并调用 DreamRec.py"
    if [ -n "$DREAMREC_CKPT" ]; then
        (
            cd "$ROOT/DreamRec"
            export CUDA_VISIBLE_DEVICES=$DREAMREC_GPU
            conda run -n DDBC python -u DreamRec.py \
                --data         $DATASET \
                --epoch        0 \
                --predict_nums "$PREDICT_NUMS" \
                --candidate_multipliers "$MULTIPLIERS" \
                --topk         1 \
                --predict_mode ar \
                --save_dir     "$(dirname "$DREAMREC_CKPT")" \
                --descri       "dreamrec-${DATASET}-${RUN_TAG}"
        ) 2>&1 | tee "$LOG_FILE"
    fi
}

test_difurec() {
    echo ""
    echo "──────────────────────────────────────────────────────────────"
    echo "  [DiffuRec] 开始测试  DATASET=$DATASET  RUN_TAG=$RUN_TAG"
    echo "──────────────────────────────────────────────────────────────"
    local LOG_FILE="$LOG_DIR/test_difurec_${RUN_TAG}_${LOG_STAMP}.log"
    MODEL_LOG_FILES[difurec]="$LOG_FILE"

    local DESC="difurec-${DATASET}-${RUN_TAG}"
    local SAVE_DIR="${DIFUREC_CKPT:-$ROOT/DiffuRec/outputs/${DATASET}/${RUN_TAG}}"

    CUDA_VISIBLE_DEVICES=$DIFUREC_GPU python "$ROOT/DiffuRec/src/main.py" \
        --mode              test \
        --dataset           $DATASET \
        --data_path         "$ROOT/DiffuRec/data/${DATASET}/dataset.pkl" \
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
        --ddbc_data_dir     "$ROOT/DreamRec/data/${DATASET}" \
        --save_dir          "$SAVE_DIR" \
        --description       "$DESC" \
        --random_seed       1997 \
        2>&1 | tee "$LOG_FILE"
    echo "  [DiffuRec] 完成，log: $LOG_FILE"
}

test_gru4rec() {
    echo ""
    echo "──────────────────────────────────────────────────────────────"
    echo "  [GRU4Rec] 开始测试  DATASET=$DATASET  RUN_TAG=$RUN_TAG"
    echo "──────────────────────────────────────────────────────────────"
    local LOG_FILE="$LOG_DIR/test_gru4rec_${RUN_TAG}_${LOG_STAMP}.log"
    MODEL_LOG_FILES[gru4rec]="$LOG_FILE"

    local OUTPUT_DIR="${GRU4REC_CKPT:-$ROOT/GRU4Rec/outputs/${DATASET}/${RUN_TAG}}"

    conda run -n DDBC python "$ROOT/GRU4Rec/train_yelp.py" \
        --epochs            0 \
        --predict_nums      "$PREDICT_NUMS" \
        --candidate_multipliers "$MULTIPLIERS" \
        --predict_mode      ar \
        --topk              1 \
        --val_seed          $SEED \
        --device            "$GRU4REC_GPU" \
        --output_dir        "$OUTPUT_DIR" \
        --log_file          "$LOG_FILE" \
        2>&1 | tee "$LOG_FILE"

    # Print DISCO-style output_results line for GRU4Rec
    conda run -n DDBC python "$ROOT/GRU4Rec/eval_best.py" \
        --ckpt "$OUTPUT_DIR/best_model.pt" \
        --predict_nums "$PREDICT_NUMS" \
        --candidate_multipliers "$MULTIPLIERS" \
        --predict_mode ar \
        --topk 1 \
        --device "$GRU4REC_GPU" \
        2>&1 | tee -a "$LOG_FILE"
    echo "  [GRU4Rec] 完成，log: $LOG_FILE"
}

test_sasrec() {
    echo ""
    echo "──────────────────────────────────────────────────────────────"
    echo "  [SASRec] 开始测试  DATASET=$DATASET  RUN_TAG=$RUN_TAG"
    echo "──────────────────────────────────────────────────────────────"
    local LOG_FILE="$LOG_DIR/test_sasrec_${RUN_TAG}_${LOG_STAMP}.log"
    MODEL_LOG_FILES[sasrec]="$LOG_FILE"

    local TRAIN_DIR="${SASREC_TRAIN_DIR:-${RUN_TAG}}"

    (
        cd "$ROOT/SASRec/python"
        CUDA_VISIBLE_DEVICES=$SASREC_GPU \
        python main_disco.py \
            --mode=test \
            --dataset=$DATASET_CAP \
            --train_dir="$TRAIN_DIR" \
            --maxlen=10 \
            --hidden_units=64 \
            --num_blocks=2 \
            --num_heads=1 \
            --ddbc_predict_nums="[$PREDICT_NUMS]" \
            --ddbc_multipliers="[$MULTIPLIERS]" \
            --ddbc_seed=$SEED \
            --item_num=20033
    ) 2>&1 | tee "$LOG_FILE"
    echo "  [SASRec] 完成，log: $LOG_FILE"
}

test_bert4rec() {
    echo ""
    echo "──────────────────────────────────────────────────────────────"
    echo "  [BERT4Rec] 开始测试  DATASET=$DATASET  RUN_TAG=$RUN_TAG"
    echo "──────────────────────────────────────────────────────────────"
    local LOG_FILE="$LOG_DIR/test_bert4rec_${RUN_TAG}_${LOG_STAMP}.log"
    MODEL_LOG_FILES[bert4rec]="$LOG_FILE"

    local CKPT_ARG=""
    [ -n "$BERT4REC_CKPT" ] && CKPT_ARG="--test_model_path $BERT4REC_CKPT"

    CUDA_VISIBLE_DEVICES=$BERT4REC_GPU \
    conda run -n DDBC python "$ROOT/BERT4Rec/main.py" \
        --mode                 test \
        --template             train_bert_yelp \
        --dataset_code         $DATASET \
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
        --ddbc_data_dir        "$ROOT/DreamRec/data/${DATASET}" \
        --experiment_dir       "$ROOT/BERT4Rec/experiments" \
        --experiment_description "bert4rec-${DATASET}-${RUN_TAG}" \
        $CKPT_ARG \
        2>&1 | tee "$LOG_FILE"
    echo "  [BERT4Rec] 完成，log: $LOG_FILE"
}

test_tiger() {
    echo ""
    echo "──────────────────────────────────────────────────────────────"
    echo "  [TIGER] 开始测试  DATASET=$DATASET  RUN_TAG=$RUN_TAG"
    echo "──────────────────────────────────────────────────────────────"
    local LOG_FILE="$LOG_DIR/test_tiger_${RUN_TAG}_${LOG_STAMP}.log"
    MODEL_LOG_FILES[tiger]="$LOG_FILE"

    local CKPT_ARG=""
    [ -n "$TIGER_CKPT" ] && CKPT_ARG="--ckpt_path=$TIGER_CKPT"

    (
        cd "$ROOT/TIGER"
        CUDA_VISIBLE_DEVICES=$TIGER_GPU python main.py \
            --mode=test \
            --model=TIGER \
            --dataset=$DATASET_CAP \
            --run_id="tiger-${DATASET}-${RUN_TAG}" \
            --ddbc_eval=True \
            --ddbc_predict_nums="[$PREDICT_NUMS]" \
            --ddbc_multipliers="[$MULTIPLIERS]" \
            --ddbc_seed=$SEED \
            $CKPT_ARG
    ) 2>&1 | tee "$LOG_FILE"
    echo "  [TIGER] 完成，log: $LOG_FILE"
}

test_letter() {
    echo ""
    echo "──────────────────────────────────────────────────────────────"
    echo "  [LETTER] 开始测试  DATASET=$DATASET  RUN_TAG=$RUN_TAG"
    echo "──────────────────────────────────────────────────────────────"
    local LOG_FILE="$LOG_DIR/test_letter_${RUN_TAG}_${LOG_STAMP}.log"
    MODEL_LOG_FILES[letter]="$LOG_FILE"

    local CKPT_DIR="${LETTER_CKPT_DIR:-$ROOT/LETTER/LETTER-TIGER/ckpt/${DATASET_CAP}_${RUN_TAG}}"

    CUDA_VISIBLE_DEVICES=$LETTER_GPU \
    python "$ROOT/LETTER/LETTER-TIGER/test_disco.py" \
        --dataset $DATASET_CAP \
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
echo "  测试完成汇总  DATASET=$DATASET  RUN_TAG=$RUN_TAG"
for line in "${SUMMARY[@]}"; do echo "$line"; done
echo "================================================================"

echo ""
echo "================================================================"
echo "  统一指标汇总 (output_results, x19/items3, topk=1)"
for MODEL in "${MODELS[@]}"; do
    key="${MODEL,,}"
    log_file="${MODEL_LOG_FILES[$key]:-}"
    if [ -z "$log_file" ] || [ ! -f "$log_file" ]; then
        echo "  [$key] MISSING_LOG"
        continue
    fi
    metric_line="$(grep "output_results OrderedDict" "$log_file" | tail -1 || true)"
    if [ -n "$metric_line" ]; then
        echo "  [$key] $metric_line"
    else
        echo "  [$key] MISSING_output_results (log: $log_file)"
    fi
done
echo "================================================================"
