#!/usr/bin/env bash
# Evaluate DiffuRec best checkpoint on Yelp test set
# Usage: bash DiffuRec/scripts/test_yelp.sh [/path/to/best_model.pt]
# If no path given, loads from default save_dir (outputs/yelp/DiffuRec_yelp/best_model.pt).

set -e

CKPT_PATH=${1:-""}

CUDA_VISIBLE_DEVICES=1 python /home/sjj/wenhao/DiffuRec/src/main.py \
    --mode              test \
    --dataset           yelp \
    --data_path         /home/sjj/wenhao/DiffuRec/data/yelp/dataset.pkl \
    --log_file          /home/sjj/wenhao/DiffuRec/log/ \
    --max_len           10 \
    --hidden_size       64 \
    --num_blocks        4 \
    --diffusion_steps   32 \
    --noise_schedule    trunc_lin \
    --schedule_sampler_name lossaware \
    --lambda_uncertainty 0.001 \
    --predict_nums      "3,5" \
    --candidate_multipliers "9,19,49,99" \
    --topk              1 \
    --ddbc_data_dir     /home/sjj/wenhao/DreamRec/data/yelp \
    --save_dir          /home/sjj/wenhao/DiffuRec/outputs/yelp/DiffuRec_yelp \
    --description       DiffuRec_yelp \
    --random_seed       1997 \
    ${CKPT_PATH:+--ckpt_path "$CKPT_PATH"}
