#!/usr/bin/env bash
# Train DiffuRec on Yelp dataset with DDBC-aligned evaluation
# Usage: bash DiffuRec/scripts/train_yelp.sh

set -e

CUDA_VISIBLE_DEVICES=1 python /home/sjj/wenhao/DiffuRec/src/main.py \
    --dataset           yelp \
    --data_path         /home/sjj/wenhao/DiffuRec/data/yelp/dataset.pkl \
    --log_file          /home/sjj/wenhao/DiffuRec/log/ \
    --max_len           10 \
    --hidden_size       64 \
    --batch_size        512 \
    --epochs            500 \
    --lr                0.001 \
    --weight_decay      0 \
    --dropout           0.1 \
    --emb_dropout       0.3 \
    --num_blocks        4 \
    --diffusion_steps   32 \
    --noise_schedule    trunc_lin \
    --schedule_sampler_name lossaware \
    --lambda_uncertainty 0.001 \
    --eval_interval     20 \
    --patience          5 \
    --predict_nums      "3" \
    --candidate_multipliers "19" \
    --topk              1 \
    --ddbc_data_dir     /home/sjj/wenhao/DreamRec/data/yelp \
    --tb_log_dir        /home/sjj/wenhao/DiffuRec/tensorboard/yelp \
    --save_dir          /home/sjj/wenhao/DiffuRec/outputs/yelp \
    --description       DiffuRec_yelp \
    --random_seed       1997
