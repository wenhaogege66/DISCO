#!/usr/bin/env bash
# =============================================================================
# train_all.sh — 一键训练脚本（DISCO + 7 baselines）
#
# 用法:
#   bash train_all.sh [模型列表]
#
# 示例:
#   bash train_all.sh                          # 训练全部
#   bash train_all.sh disco dreamrec           # 只训练 DISCO 和 DreamRec
#   bash train_all.sh sasrec bert4rec difurec  # 只训练指定模型
#
# 可选模型名: disco  dreamrec  difurec  gru4rec  sasrec  bert4rec  tiger  letter
#
# 命名规范：所有 log 和 ckpt 均包含 DATASET 和 RUN_TAG，格式：
#   log  : <Model>/logs/<DATASET>_<RUN_TAG>.log
#   ckpt : <Model>/outputs/<DATASET>/<RUN_TAG>/best_model.*
#
# 强烈建议手动指定 RUN_TAG，训练和测试使用同一个名字：
#   DATASET=yelp RUN_TAG=exp_yelp_v1 bash train_all.sh
#   DATASET=yelp RUN_TAG=exp_yelp_v1 bash test_all.sh
# 不指定时自动生成时间戳（仅用于一次性实验，测试时需手动对应）。
# =============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 数据集（影响数据路径和输出目录）─────────────────────────────────────────
DATASET="${DATASET:-yelp}"
DATASET_CAP="${DATASET^}"   # yelp → Yelp，用于需要首字母大写的参数

# ── 统一批次标签（所有模型共享，写入 log/ckpt 名称）──────────────────────────
if [ -z "${RUN_TAG:-}" ]; then
    RUN_TAG="run_$(date +%Y%m%d_%H%M%S)"
    echo "[WARN] RUN_TAG 未指定，自动生成: $RUN_TAG"
    echo "[WARN] 测试时请用: DATASET=$DATASET RUN_TAG=$RUN_TAG bash test_all.sh"
fi
echo "================================================================"
echo "  DATASET  = $DATASET"
echo "  RUN_TAG  = $RUN_TAG"
echo "  log/ckpt 命名: <Model>/logs/${DATASET}_${RUN_TAG}_$(date +%Y%m%d_%H%M%S).log"
echo "                 <Model>/outputs/${DATASET}/${RUN_TAG}/"
echo "================================================================"

# ── 选择要训练的模型 ──────────────────────────────────────────────────────────
if [ $# -eq 0 ]; then
    MODELS=(disco dreamrec difurec gru4rec sasrec bert4rec tiger letter)
else
    MODELS=("$@")
fi

# =============================================================================
# 各模型超参（在此处统一调整）
# =============================================================================

# ── DISCO ─────────────────────────────────────────────────────────────────────
DISCO_GPUS="0,1"
DISCO_BATCH_SIZE=256
DISCO_MAX_STEPS=20000
DISCO_HIDDEN_SIZE=64
DISCO_RQ_N_CODEBOOKS=3
DISCO_RQ_CODEBOOK_SIZE=256
DISCO_MODEL_LENGTH=52          # 1 + item_num*(n_codebooks+2) + 1 → 保持与 rq 配置一致
DISCO_SEQ_LEN=10
DISCO_SAMPLING_STEPS=25
DISCO_CFG_ENABLED=true
DISCO_CFG_ENCODER=true
DISCO_CFG_P_DROP=0.1
DISCO_CFG_W=2.0

# ── DreamRec ──────────────────────────────────────────────────────────────────
DREAMREC_GPU=0
DREAMREC_EPOCH=75
DREAMREC_BATCH_SIZE=256
DREAMREC_HIDDEN_FACTOR=64
DREAMREC_DIFFUSER_TYPE="mlp1"
DREAMREC_DROPOUT=0.15
DREAMREC_L2_DECAY=1e-4
DREAMREC_OPTIMIZER="adamw"
DREAMREC_LR_LIST=(0.001)       # 可改为多个 lr 串行: (0.01 0.001 0.0001)
DREAMREC_TIMESTEPS=500
DREAMREC_BETA_SCHE="exp"
DREAMREC_W=10
DREAMREC_P=0.1
DREAMREC_PREDICT_NUMS="3"
DREAMREC_MULTIPLIERS="19"
DREAMREC_EVAL_FREQ=5
DREAMREC_PREDICT_MODE="single"
DREAMREC_TOPK=1
DREAMREC_SEED=100

# ── DiffuRec ──────────────────────────────────────────────────────────────────
DIFUREC_GPU=1
DIFUREC_HIDDEN_SIZE=64
DIFUREC_BATCH_SIZE=512
DIFUREC_EPOCHS=500
DIFUREC_LR=0.001
DIFUREC_NUM_BLOCKS=4
DIFUREC_DIFFUSION_STEPS=32
DIFUREC_NOISE_SCHEDULE="trunc_lin"
DIFUREC_LAMBDA_UNCERTAINTY=0.001
DIFUREC_PREDICT_NUMS="3"
DIFUREC_MULTIPLIERS="19"
DIFUREC_EVAL_INTERVAL=20
DIFUREC_PATIENCE=5
DIFUREC_TOPK=1
DIFUREC_SEED=1997

# ── GRU4Rec ───────────────────────────────────────────────────────────────────
GRU4REC_GPU="cuda:0"
GRU4REC_LAYERS="64"
GRU4REC_LOSS="cross-entropy"
GRU4REC_EPOCHS=50
GRU4REC_BATCH_SIZE=512
GRU4REC_LR=0.05
GRU4REC_N_SAMPLE=2048
GRU4REC_PREDICT_NUMS="3"
GRU4REC_MULTIPLIERS="19"
GRU4REC_EVAL_FREQ=5
GRU4REC_PREDICT_MODE="single"
GRU4REC_TOPK=1
GRU4REC_SEED=42

# ── SASRec ────────────────────────────────────────────────────────────────────
SASREC_GPU=1
SASREC_MAXLEN=10
SASREC_HIDDEN_UNITS=64
SASREC_NUM_BLOCKS=2
SASREC_NUM_HEADS=1
SASREC_DROPOUT=0.2
SASREC_BATCH_SIZE=256
SASREC_LR=0.001
SASREC_EPOCHS=200
SASREC_EVAL_INTERVAL=5
SASREC_PATIENCE=25
SASREC_PREDICT_NUMS="[3]"
SASREC_MULTIPLIERS="[19]"
SASREC_SEED=100

# ── BERT4Rec ──────────────────────────────────────────────────────────────────
BERT4REC_GPU=1
BERT4REC_BATCH_SIZE=256
BERT4REC_EPOCHS=200
BERT4REC_LR=0.001
BERT4REC_HIDDEN_UNITS=64
BERT4REC_NUM_BLOCKS=2
BERT4REC_NUM_HEADS=4
BERT4REC_MAX_LEN=10
BERT4REC_DROPOUT=0.1
BERT4REC_MASK_PROB=0.15
BERT4REC_PREDICT_NUMS="3"
BERT4REC_MULTIPLIERS="19"
BERT4REC_EVAL_FREQ=5
BERT4REC_PATIENCE=10
BERT4REC_SEED=100

# ── TIGER ─────────────────────────────────────────────────────────────────────
TIGER_GPU=0
TIGER_EPOCHS=200
TIGER_BATCH_SIZE=256
TIGER_EVAL_BATCH_SIZE=8
TIGER_EVAL_INTERVAL=5
TIGER_PATIENCE=25
TIGER_PREDICT_NUMS="[3]"
TIGER_MULTIPLIERS="[19]"
TIGER_SEED=100

# ── LETTER ────────────────────────────────────────────────────────────────────
LETTER_GPU=1
LETTER_MAX_HIS_LEN=20
LETTER_BATCH_SIZE=256
LETTER_LR=5e-4
LETTER_EPOCHS=200
LETTER_WEIGHT_DECAY=0.01
LETTER_PATIENCE=20
LETTER_PREDICT_NUMS=3
LETTER_MULTIPLIERS=19
LETTER_SEED=100

# =============================================================================
# 训练函数
# =============================================================================

train_disco() {
    echo ""
    echo "──────────────────────────────────────────────────────────────"
    echo "  [DISCO] 开始训练  DATASET=$DATASET  RUN_TAG=$RUN_TAG"
    echo "──────────────────────────────────────────────────────────────"
    local LOG_DIR="$ROOT/DISCO/logs"
    mkdir -p "$LOG_DIR"
    local LOG_FILE="$LOG_DIR/${DATASET}_${RUN_TAG}_$(date +%Y%m%d_%H%M%S).log"

    export CUDA_VISIBLE_DEVICES=$DISCO_GPUS
    export PYTHONPATH="$ROOT/DISCO:$PYTHONPATH"
    export HF_ENDPOINT=https://hf-mirror.com
    export TRANSFORMERS_OFFLINE=1

    python "$ROOT/DISCO/main.py" \
        training.layer_loss_weights.enabled=false \
        loader.batch_size=$DISCO_BATCH_SIZE \
        loader.eval_batch_size=$DISCO_BATCH_SIZE \
        trainer.max_steps=$DISCO_MAX_STEPS \
        model=small \
        model.hidden_size=$DISCO_HIDDEN_SIZE \
        data=$DATASET \
        dataset=$DATASET_CAP \
        run_name="disco-${DATASET}-${RUN_TAG}" \
        parameterization=subs \
        seq_len=$DISCO_SEQ_LEN \
        rq_n_codebooks=$DISCO_RQ_N_CODEBOOKS \
        rq_codebook_size=$DISCO_RQ_CODEBOOK_SIZE \
        model.length=$DISCO_MODEL_LENGTH \
        swap_ratio=0 \
        eval.compute_generative_perplexity=False \
        sampling.steps=$DISCO_SAMPLING_STEPS \
        sampling.cfg_enabled=$DISCO_CFG_ENABLED \
        sampling.cfg_encoder=$DISCO_CFG_ENCODER \
        sampling.cfg_p_drop=$DISCO_CFG_P_DROP \
        sampling.cfg_w=$DISCO_CFG_W \
        use_tensorboard=true \
        checkpointing.resume_from_ckpt=false \
        2>&1 | tee "$LOG_FILE"
    echo "  [DISCO] 完成，log: $LOG_FILE"
}

train_dreamrec() {
    echo ""
    echo "──────────────────────────────────────────────────────────────"
    echo "  [DreamRec] 开始训练  DATASET=$DATASET  RUN_TAG=$RUN_TAG"
    echo "──────────────────────────────────────────────────────────────"
    export CUDA_VISIBLE_DEVICES=$DREAMREC_GPU

    for LR in "${DREAMREC_LR_LIST[@]}"; do
        local RUN_NAME="dreamrec-${DATASET}-lr${LR}-${RUN_TAG}"
        local SAVE_DIR="$ROOT/DreamRec/outputs/${DATASET}/${RUN_TAG}"
        local LOG_FILE="$SAVE_DIR/train.log"
        local TB_DIR="$ROOT/DreamRec/tensorboard/${DATASET}/${RUN_TAG}"
        mkdir -p "$SAVE_DIR"

        echo "  LR=$LR  ->  $SAVE_DIR"
        python -u "$ROOT/DreamRec/DreamRec.py" \
            --data         $DATASET \
            --epoch        $DREAMREC_EPOCH \
            --batch_size   $DREAMREC_BATCH_SIZE \
            --random_seed  $DREAMREC_SEED \
            --hidden_factor $DREAMREC_HIDDEN_FACTOR \
            --diffuser_type "$DREAMREC_DIFFUSER_TYPE" \
            --dropout_rate  $DREAMREC_DROPOUT \
            --l2_decay      $DREAMREC_L2_DECAY \
            --optimizer     "$DREAMREC_OPTIMIZER" \
            --lr            $LR \
            --timesteps     $DREAMREC_TIMESTEPS \
            --beta_sche     "$DREAMREC_BETA_SCHE" \
            --w             $DREAMREC_W \
            --p             $DREAMREC_P \
            --predict_nums  "$DREAMREC_PREDICT_NUMS" \
            --candidate_multipliers "$DREAMREC_MULTIPLIERS" \
            --eval_freq     $DREAMREC_EVAL_FREQ \
            --predict_mode  "$DREAMREC_PREDICT_MODE" \
            --topk          $DREAMREC_TOPK \
            --tb_log_dir    "$TB_DIR" \
            --save_dir      "$SAVE_DIR" \
            --cuda          $DREAMREC_GPU \
            --descri        "$RUN_NAME" \
            2>&1 | tee "$LOG_FILE"
    done
    echo "  [DreamRec] 完成"
}

train_difurec() {
    echo ""
    echo "──────────────────────────────────────────────────────────────"
    echo "  [DiffuRec] 开始训练  DATASET=$DATASET  RUN_TAG=$RUN_TAG"
    echo "──────────────────────────────────────────────────────────────"
    local SAVE_DIR="$ROOT/DiffuRec/outputs/${DATASET}/${RUN_TAG}"
    local LOG_FILE="$ROOT/DiffuRec/logs/${DATASET}_${RUN_TAG}_$(date +%Y%m%d_%H%M%S).log"
    local TB_DIR="$ROOT/DiffuRec/tensorboard/${DATASET}/${RUN_TAG}"
    mkdir -p "$ROOT/DiffuRec/logs" "$SAVE_DIR"

    CUDA_VISIBLE_DEVICES=$DIFUREC_GPU python "$ROOT/DiffuRec/src/main.py" \
        --dataset           $DATASET \
        --data_path         "$ROOT/DiffuRec/data/${DATASET}/dataset.pkl" \
        --log_file          "$ROOT/DiffuRec/log/" \
        --max_len           10 \
        --hidden_size       $DIFUREC_HIDDEN_SIZE \
        --batch_size        $DIFUREC_BATCH_SIZE \
        --epochs            $DIFUREC_EPOCHS \
        --lr                $DIFUREC_LR \
        --weight_decay      0 \
        --dropout           0.1 \
        --emb_dropout       0.3 \
        --num_blocks        $DIFUREC_NUM_BLOCKS \
        --diffusion_steps   $DIFUREC_DIFFUSION_STEPS \
        --noise_schedule    "$DIFUREC_NOISE_SCHEDULE" \
        --schedule_sampler_name lossaware \
        --lambda_uncertainty $DIFUREC_LAMBDA_UNCERTAINTY \
        --eval_interval     $DIFUREC_EVAL_INTERVAL \
        --patience          $DIFUREC_PATIENCE \
        --predict_nums      "$DIFUREC_PREDICT_NUMS" \
        --candidate_multipliers "$DIFUREC_MULTIPLIERS" \
        --topk              $DIFUREC_TOPK \
        --ddbc_data_dir     "$ROOT/DreamRec/data/${DATASET}" \
        --tb_log_dir        "$TB_DIR" \
        --save_dir          "$SAVE_DIR" \
        --description       "difurec-${DATASET}-${RUN_TAG}" \
        --random_seed       $DIFUREC_SEED \
        2>&1 | tee "$LOG_FILE"
    echo "  [DiffuRec] 完成，log: $LOG_FILE"
}

train_gru4rec() {
    echo ""
    echo "──────────────────────────────────────────────────────────────"
    echo "  [GRU4Rec] 开始训练  DATASET=$DATASET  RUN_TAG=$RUN_TAG"
    echo "──────────────────────────────────────────────────────────────"
    local OUTPUT_DIR="$ROOT/GRU4Rec/outputs/${DATASET}/${RUN_TAG}"
    local LOG_FILE="$ROOT/GRU4Rec/logs/${DATASET}_${RUN_TAG}_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "$ROOT/GRU4Rec/logs"

    conda run -n DDBC python "$ROOT/GRU4Rec/train_yelp.py" \
        --layers              "$GRU4REC_LAYERS" \
        --loss                "$GRU4REC_LOSS" \
        --epochs              "$GRU4REC_EPOCHS" \
        --batch_size          "$GRU4REC_BATCH_SIZE" \
        --lr                  "$GRU4REC_LR" \
        --n_sample            "$GRU4REC_N_SAMPLE" \
        --eval_freq           "$GRU4REC_EVAL_FREQ" \
        --predict_nums        "$GRU4REC_PREDICT_NUMS" \
        --candidate_multipliers "$GRU4REC_MULTIPLIERS" \
        --predict_mode        "$GRU4REC_PREDICT_MODE" \
        --topk                "$GRU4REC_TOPK" \
        --val_seed            "$GRU4REC_SEED" \
        --device              "$GRU4REC_GPU" \
        --seed                "$GRU4REC_SEED" \
        --output_dir          "$OUTPUT_DIR" \
        --log_file            "$LOG_FILE" \
        2>&1 | tee "$LOG_FILE"
    echo "  [GRU4Rec] 完成，log: $LOG_FILE"
}

train_sasrec() {
    echo ""
    echo "──────────────────────────────────────────────────────────────"
    echo "  [SASRec] 开始训练  DATASET=$DATASET  RUN_TAG=$RUN_TAG"
    echo "──────────────────────────────────────────────────────────────"
    local LOG_FILE="$ROOT/SASRec/logs/${DATASET}_${RUN_TAG}_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "$ROOT/SASRec/logs"

    (
        cd "$ROOT/SASRec/python"
        CUDA_VISIBLE_DEVICES=$SASREC_GPU python main_disco.py \
            --dataset=$DATASET_CAP \
            --train_dir="${DATASET}_${RUN_TAG}" \
            --maxlen=$SASREC_MAXLEN \
            --hidden_units=$SASREC_HIDDEN_UNITS \
            --num_blocks=$SASREC_NUM_BLOCKS \
            --num_heads=$SASREC_NUM_HEADS \
            --dropout_rate=$SASREC_DROPOUT \
            --batch_size=$SASREC_BATCH_SIZE \
            --lr=$SASREC_LR \
            --num_epochs=$SASREC_EPOCHS \
            --eval_interval=$SASREC_EVAL_INTERVAL \
            --patience=$SASREC_PATIENCE \
            --ddbc_predict_nums="$SASREC_PREDICT_NUMS" \
            --ddbc_multipliers="$SASREC_MULTIPLIERS" \
            --ddbc_seed=$SASREC_SEED \
            --item_num=20033 \
            --tb_log_dir="$ROOT/SASRec/tensorboard/${DATASET}_${RUN_TAG}"
    ) 2>&1 | tee "$LOG_FILE"
    echo "  [SASRec] 完成，log: $LOG_FILE"
}

train_bert4rec() {
    echo ""
    echo "──────────────────────────────────────────────────────────────"
    echo "  [BERT4Rec] 开始训练  DATASET=$DATASET  RUN_TAG=$RUN_TAG"
    echo "──────────────────────────────────────────────────────────────"
    local LOG_FILE="$ROOT/BERT4Rec/logs/${DATASET}_${RUN_TAG}_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "$ROOT/BERT4Rec/logs"

    CUDA_VISIBLE_DEVICES=$BERT4REC_GPU \
    python "$ROOT/BERT4Rec/main.py" \
        --template             train_bert_yelp \
        --dataset_code         $DATASET \
        --device               cuda \
        --device_idx           $BERT4REC_GPU \
        --train_batch_size     $BERT4REC_BATCH_SIZE \
        --val_batch_size       $BERT4REC_BATCH_SIZE \
        --test_batch_size      $BERT4REC_BATCH_SIZE \
        --num_epochs           $BERT4REC_EPOCHS \
        --lr                   $BERT4REC_LR \
        --weight_decay         0 \
        --enable_lr_schedule   True \
        --decay_step           25 \
        --gamma                1.0 \
        --bert_max_len         $BERT4REC_MAX_LEN \
        --bert_hidden_units    $BERT4REC_HIDDEN_UNITS \
        --bert_num_blocks      $BERT4REC_NUM_BLOCKS \
        --bert_num_heads       $BERT4REC_NUM_HEADS \
        --bert_dropout         $BERT4REC_DROPOUT \
        --bert_mask_prob       $BERT4REC_MASK_PROB \
        --model_init_seed      0 \
        --predict_nums         "$BERT4REC_PREDICT_NUMS" \
        --candidate_multipliers "$BERT4REC_MULTIPLIERS" \
        --topk                 1 \
        --eval_freq            $BERT4REC_EVAL_FREQ \
        --patience             $BERT4REC_PATIENCE \
        --random_seed          $BERT4REC_SEED \
        --ddbc_data_dir        "$ROOT/DreamRec/data/${DATASET}" \
        --experiment_dir       "$ROOT/BERT4Rec/experiments" \
        --experiment_description "bert4rec-${DATASET}-${RUN_TAG}" \
        2>&1 | tee "$LOG_FILE"
    echo "  [BERT4Rec] 完成，log: $LOG_FILE"
}

train_tiger() {
    echo ""
    echo "──────────────────────────────────────────────────────────────"
    echo "  [TIGER] 开始训练  DATASET=$DATASET  RUN_TAG=$RUN_TAG"
    echo "──────────────────────────────────────────────────────────────"
    local LOG_FILE="$ROOT/TIGER/logs/${DATASET}_${RUN_TAG}_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "$ROOT/TIGER/logs"

    (
        cd "$ROOT/TIGER"
        CUDA_VISIBLE_DEVICES=$TIGER_GPU python main.py \
            --model=TIGER \
            --dataset=$DATASET_CAP \
            --run_id="tiger-${DATASET}-${RUN_TAG}" \
            --ddbc_eval=True \
            --ddbc_predict_nums="$TIGER_PREDICT_NUMS" \
            --ddbc_multipliers="$TIGER_MULTIPLIERS" \
            --ddbc_seed=$TIGER_SEED \
            --eval_interval=$TIGER_EVAL_INTERVAL \
            --epochs=$TIGER_EPOCHS \
            --patience=$TIGER_PATIENCE \
            --train_batch_size=$TIGER_BATCH_SIZE \
            --eval_batch_size=$TIGER_EVAL_BATCH_SIZE
    ) 2>&1 | tee "$LOG_FILE"
    echo "  [TIGER] 完成，log: $LOG_FILE"
}

train_letter() {
    echo ""
    echo "──────────────────────────────────────────────────────────────"
    echo "  [LETTER] 开始训练  DATASET=$DATASET  RUN_TAG=$RUN_TAG"
    echo "──────────────────────────────────────────────────────────────"
    local OUTPUT_DIR="$ROOT/LETTER/LETTER-TIGER/ckpt/${DATASET_CAP}_${RUN_TAG}"
    local LOG_FILE="$ROOT/LETTER/logs/${DATASET}_${RUN_TAG}_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "$ROOT/LETTER/logs"

    export WANDB_DISABLED=true
    CUDA_VISIBLE_DEVICES=$LETTER_GPU \
    torchrun --nproc_per_node=1 --master_port=2315 \
        "$ROOT/LETTER/LETTER-TIGER/finetune_disco.py" \
        --dataset $DATASET_CAP \
        --data_path "$ROOT/LETTER/data" \
        --base_model "$ROOT/LETTER/LETTER-TIGER/ckpt/TIGER" \
        --output_dir "$OUTPUT_DIR" \
        --index_file .index.json \
        --max_his_len $LETTER_MAX_HIS_LEN \
        --per_device_batch_size $LETTER_BATCH_SIZE \
        --gradient_accumulation_steps 1 \
        --learning_rate $LETTER_LR \
        --epochs $LETTER_EPOCHS \
        --weight_decay $LETTER_WEIGHT_DECAY \
        --warmup_ratio 0.01 \
        --lr_scheduler_type cosine \
        --logging_step 50 \
        --optim adamw_torch \
        --save_and_eval_strategy epoch \
        --temperature 1.0 \
        --ddbc_predict_nums $LETTER_PREDICT_NUMS \
        --ddbc_multipliers $LETTER_MULTIPLIERS \
        --ddbc_seed $LETTER_SEED \
        --patience $LETTER_PATIENCE \
        2>&1 | tee "$LOG_FILE"
    echo "  [LETTER] 完成，log: $LOG_FILE"
}

# =============================================================================
# 主流程
# =============================================================================

SUMMARY=()

for MODEL in "${MODELS[@]}"; do
    case "${MODEL,,}" in
        disco)
            if train_disco; then
                SUMMARY+=("  disco     OK")
            else
                SUMMARY+=("  disco     FAILED")
            fi
            ;;
        dreamrec)
            if train_dreamrec; then
                SUMMARY+=("  dreamrec  OK")
            else
                SUMMARY+=("  dreamrec  FAILED")
            fi
            ;;
        difurec|diffurec)
            if train_difurec; then
                SUMMARY+=("  difurec   OK")
            else
                SUMMARY+=("  difurec   FAILED")
            fi
            ;;
        gru4rec)
            if train_gru4rec; then
                SUMMARY+=("  gru4rec   OK")
            else
                SUMMARY+=("  gru4rec   FAILED")
            fi
            ;;
        sasrec)
            if train_sasrec; then
                SUMMARY+=("  sasrec    OK")
            else
                SUMMARY+=("  sasrec    FAILED")
            fi
            ;;
        bert4rec)
            if train_bert4rec; then
                SUMMARY+=("  bert4rec  OK")
            else
                SUMMARY+=("  bert4rec  FAILED")
            fi
            ;;
        tiger)
            if train_tiger; then
                SUMMARY+=("  tiger     OK")
            else
                SUMMARY+=("  tiger     FAILED")
            fi
            ;;
        letter)
            if train_letter; then
                SUMMARY+=("  letter    OK")
            else
                SUMMARY+=("  letter    FAILED")
            fi
            ;;
        *) echo "  [WARN] 未知模型: $MODEL，跳过" ;;
    esac
done

echo ""
echo "================================================================"
echo "  训练完成汇总  RUN_TAG=$RUN_TAG"
for line in "${SUMMARY[@]}"; do echo "$line"; done
echo "================================================================"
