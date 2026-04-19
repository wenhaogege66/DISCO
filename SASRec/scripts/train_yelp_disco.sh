#!/bin/bash
# Train SASRec on Yelp_2020 with DDBC-compatible evaluation.
# Checkpoint selection: val_recall@3_x19.
# Test evaluation runs automatically after training with the best checkpoint.
#
# Prerequisites:
#   bash SASRec/scripts/convert_yelp.sh   (run once to generate data/Yelp.txt)

set -e
cd /home/sjj/wenhao/SASRec/python

mkdir -p ../../logs

CUDA_VISIBLE_DEVICES=1 python main_disco.py \
    --dataset=Yelp \
    --train_dir=disco \
    --maxlen=10 \
    --hidden_units=64 \
    --num_blocks=2 \
    --num_heads=1 \
    --dropout_rate=0.2 \
    --l2_emb=0.0 \
    --batch_size=256 \
    --lr=0.001 \
    --num_epochs=200 \
    --eval_interval=5 \
    --patience=25 \
    --ddbc_predict_nums="[3]" \
    --ddbc_multipliers="[19]" \
    --ddbc_seed=100 \
    2>&1 | tee ../../logs/sasrec_yelp_disco_$(date +%Y%m%d_%H%M%S).log
