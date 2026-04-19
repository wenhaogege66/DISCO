#!/bin/bash
# Train TIGER on Yelp_2020 with DDBC-compatible evaluation.
# Checkpoint selection: val_recall@3_x19 (overrides default ndcg@10).
# Test evaluation runs automatically after training with the best checkpoint.

set -e
cd /home/sjj/wenhao/TIGER

CUDA_VISIBLE_DEVICES=0 python main.py \
    --model=TIGER \
    --dataset=Yelp \
    --run_id=tiger_yelp_disco \
    --ddbc_eval=True \
    --ddbc_predict_nums="[3]" \
    --ddbc_multipliers="[19]" \
    --ddbc_seed=100 \
    --eval_interval=5 \
    --epochs=200 \
    --patience=25 \
    --train_batch_size=256 \
    --eval_batch_size=32 \
    2>&1 | tee logs/yelp_disco_$(date +%Y%m%d_%H%M%S).log
