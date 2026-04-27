#!/bin/bash
# Parameterized TIGER test-only script for DISCO-style evaluation.
# Example:
#   CKPT=/home/sjj/wenhao/TIGER/ckpt/tiger_ml60_disco-xxx.pth \
#   bash scripts/test_ml_disco.sh DDBC_DATASET=MovieLens-20M/len60 SEQ_LEN=60 PREDICT_NUMS="[30]" MULTIPLIERS="[19]" ITEM_NUM=17188

set -e
cd /home/sjj/wenhao/TIGER

mkdir -p logs

GPU=${GPU:-1}
DATASET=${DATASET:-MovieLens20M}
CATEGORY=${CATEGORY:-MovieLens20M}
DDBC_DATASET=${DDBC_DATASET:-MovieLens-20M/len60}
SEQ_LEN=${SEQ_LEN:-60}
RUN_ID=${RUN_ID:-tiger_ml60_disco}
PREDICT_NUMS=${PREDICT_NUMS:-"[30]"}
MULTIPLIERS=${MULTIPLIERS:-"[19]"}
ITEM_NUM=${ITEM_NUM:-17188}
SEED=${SEED:-100}
TRAIN_BS=${TRAIN_BS:-256}
EVAL_BS=${EVAL_BS:-32}

CUDA_VISIBLE_DEVICES=${GPU} python main.py \
    --model=TIGER \
    --dataset=${DATASET} \
    --category=${CATEGORY} \
    --seq_len=${SEQ_LEN} \
    --max_item_seq_len=${SEQ_LEN} \
    --run_id=${RUN_ID} \
    --mode=test \
    ${CKPT:+--ckpt_path="${CKPT}"} \
    --ddbc_eval=True \
    --ddbc_dataset=${DDBC_DATASET} \
    --ddbc_item_num=${ITEM_NUM} \
    --ddbc_predict_nums="${PREDICT_NUMS}" \
    --ddbc_multipliers="${MULTIPLIERS}" \
    --ddbc_seed=${SEED} \
    --train_batch_size=${TRAIN_BS} \
    --eval_batch_size=${EVAL_BS} \
    2>&1 | tee logs/${RUN_ID}_test_$(date +%Y%m%d_%H%M%S).log
