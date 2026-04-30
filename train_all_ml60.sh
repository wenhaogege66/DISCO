#!/usr/bin/env bash
# =============================================================================
# train_all_ml60.sh — 一键训练脚本（DISCO + 7 baselines on MovieLens-60）
#
# 用法:
#   bash train_all_ml60.sh [模型列表]
#
# 示例:
#   bash train_all_ml60.sh                          # 训练全部
#   bash train_all_ml60.sh disco dreamrec           # 只训练 DISCO 和 DreamRec
#   bash train_all_ml60.sh sasrec bert4rec difurec  # 只训练指定模型
#
# 可选模型名: disco  dreamrec  difurec  gru4rec  sasrec  bert4rec  tiger  letter
#
# 命名规范：所有 log 和 ckpt 均包含 DATASET 和 RUN_TAG，格式：
#   log  : <Model>/logs/${DATASET}_${RUN_TAG}.log
#   ckpt : <Model>/outputs/${DATASET}/${RUN_TAG}/best_model.*
#
# 强烈建议手动指定 RUN_TAG：
#   DATASET=ml60 RUN_TAG=exp_ml60_v1 bash train_all_ml60.sh
#   DATASET=ml60 RUN_TAG=exp_ml60_v1 bash test_all_ml60.sh
# =============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 数据集 ──────────────────────────────────────────────────────────────────
DATASET="${DATASET:-ml60}"
DISCO_DATA_NAME="movielens20m_len60"   # DISCO config data name
DISCO_DATASET_NAME="MovieLens-20M"     # DISCO dataset folder name

# Data paths
DISCO_DATA_DIR="$ROOT/DISCO/datasets/$DISCO_DATASET_NAME/len60"
DREAMREC_DATA_DIR="$ROOT/DreamRec/data/${DATASET}"
DIFUREC_DATA_DIR="$ROOT/DiffuRec/data/${DATASET}"
GRU4REC_DATA_DIR="$ROOT/GRU4Rec/data/${DATASET}"
SASREC_DATA_FILE="$ROOT/SASRec/python/data/MovieLens60.txt"
BERT4REC_DATA_DIR="$ROOT/BERT4Rec/Data/preprocessed/${DATASET}_min_rating0-min_uc0-min_sc0-splitleave_one_out"

# ML-60 constants
ITEM_NUM=17188
SEQ_SIZE=60

# ── 统一批次标签 ────────────────────────────────────────────────────────────
if [ -z "${RUN_TAG:-}" ]; then
    RUN_TAG="run_$(date +%Y%m%d_%H%M%S)"
    echo "[WARN] RUN_TAG 未指定，自动生成: $RUN_TAG"
fi

# ── 统一日志目录 ────────────────────────────────────────────────────────────
LOG_DIR="$ROOT/logs/$DATASET"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "  DATASET  = $DATASET  (MovieLens-60)"
echo "  RUN_TAG  = $RUN_TAG"
echo "  ITEM_NUM = $ITEM_NUM"
echo "  SEQ_SIZE = $SEQ_SIZE"
echo "  LOG_DIR  = $LOG_DIR"
echo "================================================================"

# ── 选择要训练的模型 ────────────────────────────────────────────────────────
if [ $# -eq 0 ]; then
    MODELS=(disco dreamrec difurec gru4rec sasrec bert4rec tiger letter)
else
    MODELS=("$@")
fi

# =============================================================================
# 各模型超参（ML-60 适配）
# =============================================================================

# ── DISCO ─────────────────────────────────────────────────────────────────────
DISCO_GPUS="0,1"
DISCO_BATCH_SIZE=256
DISCO_MAX_STEPS=20000
DISCO_HIDDEN_SIZE=64
DISCO_RQ_N_CODEBOOKS=3
DISCO_RQ_CODEBOOK_SIZE=256
DISCO_MODEL_LENGTH=302         # 1 + 60*(3+2) + 1 = 302 (full 60 items)
DISCO_SEQ_LEN=60
DISCO_SAMPLING_STEPS=256
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
DREAMREC_LR_LIST=(0.001)
DREAMREC_TIMESTEPS=500
DREAMREC_BETA_SCHE="exp"
DREAMREC_W=10
DREAMREC_P=0.1
DREAMREC_PREDICT_NUMS="30"
DREAMREC_MULTIPLIERS="19"
DREAMREC_EVAL_FREQ=5
DREAMREC_PREDICT_MODE="single"
DREAMREC_TOPK=1
DREAMREC_SEED=100

# ── DiffuRec ──────────────────────────────────────────────────────────────────
DIFUREC_GPU=1
DIFUREC_HIDDEN_SIZE=64
DIFUREC_BATCH_SIZE=512
DIFUREC_EPOCHS=80
DIFUREC_LR=0.001
DIFUREC_NUM_BLOCKS=4
DIFUREC_DIFFUSION_STEPS=32
DIFUREC_NOISE_SCHEDULE="trunc_lin"
DIFUREC_LAMBDA_UNCERTAINTY=0.001
DIFUREC_PREDICT_NUMS="30"
DIFUREC_MULTIPLIERS="19"
DIFUREC_EVAL_INTERVAL=20
DIFUREC_EVAL_START_EPOCH=50
DIFUREC_PREDICT_MODE="single"
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
GRU4REC_PREDICT_NUMS="30"
GRU4REC_MULTIPLIERS="19"
GRU4REC_EVAL_FREQ=5
GRU4REC_PREDICT_MODE="single"
GRU4REC_TOPK=1
GRU4REC_SEED=42

# ── SASRec ────────────────────────────────────────────────────────────────────
SASREC_GPU=1
SASREC_MAXLEN=60
SASREC_HIDDEN_UNITS=64
SASREC_NUM_BLOCKS=2
SASREC_NUM_HEADS=1
SASREC_DROPOUT=0.2
SASREC_BATCH_SIZE=256
SASREC_LR=0.001
SASREC_EPOCHS=200
SASREC_EVAL_INTERVAL=5
SASREC_PATIENCE=25
SASREC_PREDICT_NUMS="[30]"
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
BERT4REC_MAX_LEN=60
BERT4REC_DROPOUT=0.1
BERT4REC_MASK_PROB=0.15
BERT4REC_PREDICT_NUMS="30"
BERT4REC_MULTIPLIERS="19"
BERT4REC_EVAL_FREQ=5
BERT4REC_PATIENCE=10
BERT4REC_SEED=100

# ── TIGER ─────────────────────────────────────────────────────────────────────
TIGER_GPU=0
TIGER_EPOCHS=150
TIGER_BATCH_SIZE=256
TIGER_EVAL_BATCH_SIZE=8
TIGER_EVAL_START_EPOCH=30
TIGER_EVAL_INTERVAL=20
TIGER_PATIENCE=25
TIGER_PREDICT_NUMS="[30]"
TIGER_MULTIPLIERS="[19]"
TIGER_PREDICT_MODE="single"
TIGER_SEED=100

# ── LETTER ────────────────────────────────────────────────────────────────────
LETTER_GPU=1
LETTER_MAX_HIS_LEN=60
LETTER_BATCH_SIZE=256
LETTER_LR=5e-4
LETTER_EPOCHS=150
LETTER_WEIGHT_DECAY=0.01
LETTER_PATIENCE=20
LETTER_PREDICT_NUMS=30
LETTER_MULTIPLIERS=19
LETTER_EVAL_START_EPOCH=30
LETTER_EVAL_INTERVAL=20
LETTER_PREDICT_MODE="single"
LETTER_SEED=100

# =============================================================================
# 数据转换检查
# =============================================================================
check_and_convert_data() {
    echo ""
    echo "──────────────────────────────────────────────────────────────"
    echo "  检查 ML-60 数据转换状态"
    echo "──────────────────────────────────────────────────────────────"

    local need_convert=false

    # DreamRec data
    if [ ! -f "$DREAMREC_DATA_DIR/data_statis.df" ]; then
        echo "  [MISSING] DreamRec .df data"
        need_convert=true
    else
        echo "  [OK] DreamRec .df data"
    fi

    # DiffuRec data
    if [ ! -f "$DIFUREC_DATA_DIR/dataset.pkl" ]; then
        echo "  [MISSING] DiffuRec dataset.pkl"
        need_convert=true
    else
        echo "  [OK] DiffuRec dataset.pkl"
    fi

    # BERT4Rec data
    if [ ! -f "$BERT4REC_DATA_DIR/dataset.pkl" ]; then
        echo "  [MISSING] BERT4Rec dataset.pkl"
        need_convert=true
    else
        echo "  [OK] BERT4Rec dataset.pkl"
    fi

    # SASRec data
    if [ ! -f "$SASREC_DATA_FILE" ]; then
        echo "  [MISSING] SASRec MovieLens60.txt"
        need_convert=true
    else
        echo "  [OK] SASRec MovieLens60.txt"
    fi

    # Test candidates (check predict_n=30, multiplier=19)
    if [ ! -f "$DISCO_DATA_DIR/test_candidates_seed1_x19_items30.pkl" ]; then
        echo "  [MISSING] test_candidates_seed1_x19_items30.pkl"
        need_convert=true
    else
        echo "  [OK] test_candidates_seed1_x19_items30.pkl"
    fi

    if [ "$need_convert" = true ]; then
        echo ""
        echo "  正在运行数据转换..."
        conda run -n DDBC python "$DISCO_DATA_DIR/convert_disco_to_dreamrec.py"
        conda run -n DDBC python "$DISCO_DATA_DIR/convert_disco_to_diffurec.py"
        conda run -n DDBC python "$DISCO_DATA_DIR/convert_disco_to_bert4rec.py"
        conda run -n DDBC python "$DISCO_DATA_DIR/convert_disco_to_sasrec.py"
        echo "  数据转换完成。"
    fi

    # Always re-derive TIGER/LETTER training data from DISCO train.txt
    # (DISCO is source of truth for the train/val/test split)
    echo ""
    echo "  正在从 DISCO train.txt 派生 TIGER/LETTER 训练数据..."
    conda run -n DDBC python "$DISCO_DATA_DIR/convert_disco_to_tiger_letter.py"
    echo "  TIGER/LETTER 数据派生完成（仅使用 DISCO 训练集用户）。"
}

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
        data=$DISCO_DATA_NAME \
        dataset=$DISCO_DATASET_NAME \
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
        (
            cd "$ROOT/DreamRec"
            python -u DreamRec.py \
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
        )
    done
    echo "  [DreamRec] 完成"
}

train_difurec() {
    echo ""
    echo "──────────────────────────────────────────────────────────────"
    echo "  [DiffuRec] 开始训练  DATASET=$DATASET  RUN_TAG=$RUN_TAG"
    echo "──────────────────────────────────────────────────────────────"
    local SAVE_DIR="$ROOT/DiffuRec/outputs/${DATASET}/${RUN_TAG}"
    local LOG_FILE="$LOG_DIR/difurec_${RUN_TAG}_$(date +%Y%m%d_%H%M%S).log"
    local TB_DIR="$ROOT/DiffuRec/tensorboard/${DATASET}/${RUN_TAG}"
    mkdir -p "$SAVE_DIR"

    CUDA_VISIBLE_DEVICES=$DIFUREC_GPU python "$ROOT/DiffuRec/src/main.py" \
        --dataset           $DATASET \
        --data_path         "$DIFUREC_DATA_DIR/dataset.pkl" \
        --log_file          "$ROOT/DiffuRec/log/" \
        --max_len           $SEQ_SIZE \
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
        --eval_start_epoch  $DIFUREC_EVAL_START_EPOCH \
        --predict_mode      "$DIFUREC_PREDICT_MODE" \
        --patience          $DIFUREC_PATIENCE \
        --predict_nums      "$DIFUREC_PREDICT_NUMS" \
        --candidate_multipliers "$DIFUREC_MULTIPLIERS" \
        --topk              $DIFUREC_TOPK \
        --ddbc_data_dir     "$DREAMREC_DATA_DIR" \
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

    conda run -n DDBC python "$ROOT/GRU4Rec/train_movielens60.py" \
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
            --dataset=MovieLens60 \
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
            --item_num=$ITEM_NUM \
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
    conda run -n BERT4Rec python "$ROOT/BERT4Rec/main.py" \
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
        --ddbc_data_dir        "$DREAMREC_DATA_DIR" \
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
    local LOG_FILE="$LOG_DIR/tiger_${RUN_TAG}_$(date +%Y%m%d_%H%M%S).log"

    (
        cd "$ROOT/TIGER"
        CUDA_VISIBLE_DEVICES=$TIGER_GPU python main.py \
            --model=TIGER \
            --dataset=MovieLens20M \
            --category=len60 \
            --run_id="tiger-${DATASET}-${RUN_TAG}" \
            --ddbc_eval=True \
            --ddbc_predict_nums="$TIGER_PREDICT_NUMS" \
            --ddbc_multipliers="$TIGER_MULTIPLIERS" \
            --ddbc_seed=$TIGER_SEED \
            --ddbc_predict_mode="$TIGER_PREDICT_MODE" \
            --ddbc_dataset="MovieLens-20M/len60" \
            --ddbc_item_num=$ITEM_NUM \
            --eval_interval=$TIGER_EVAL_INTERVAL \
            --eval_start_epoch=$TIGER_EVAL_START_EPOCH \
            --epochs=$TIGER_EPOCHS \
            --patience=$TIGER_PATIENCE \
            --train_batch_size=$TIGER_BATCH_SIZE \
            --eval_batch_size=$TIGER_EVAL_BATCH_SIZE \
            --use_fp16=True
    ) 2>&1 | tee "$LOG_FILE"
    echo "  [TIGER] 完成，log: $LOG_FILE"
}

train_letter() {
    echo ""
    echo "──────────────────────────────────────────────────────────────"
    echo "  [LETTER] 开始训练  DATASET=$DATASET  RUN_TAG=$RUN_TAG"
    echo "──────────────────────────────────────────────────────────────"
    local OUTPUT_DIR="$ROOT/LETTER/LETTER-TIGER/ckpt/MovieLens-20M_${RUN_TAG}"
    local LOG_FILE="$LOG_DIR/letter_${RUN_TAG}_$(date +%Y%m%d_%H%M%S).log"

    export WANDB_DISABLED=true
    CUDA_VISIBLE_DEVICES=$LETTER_GPU \
    torchrun --nproc_per_node=1 --master_port=2315 \
        "$ROOT/LETTER/LETTER-TIGER/finetune_disco.py" \
        --dataset MovieLens-20M \
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
        --eval_start_epoch $LETTER_EVAL_START_EPOCH \
        --eval_interval $LETTER_EVAL_INTERVAL \
        --predict_mode $LETTER_PREDICT_MODE \
        --fp16 \
        2>&1 | tee "$LOG_FILE"
    echo "  [LETTER] 完成，log: $LOG_FILE"
}

# =============================================================================
# 主流程
# =============================================================================

# 先检查和转换数据
check_and_convert_data

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
echo "  训练完成汇总  DATASET=$DATASET  RUN_TAG=$RUN_TAG"
for line in "${SUMMARY[@]}"; do echo "$line"; done
echo "================================================================"
