#!/bin/bash
# Run DDBC test evaluation on a saved LETTER-TIGER checkpoint.
#
# Usage:
#   bash scripts/test_yelp_disco.sh [ckpt_dir]
#
# If ckpt_dir is not provided, defaults to ./ckpt/Yelp_disco (best model saved there).

set -e
cd /home/sjj/wenhao/LETTER/LETTER-TIGER

CKPT_DIR=${1:-./ckpt/Yelp_disco}

CUDA_VISIBLE_DEVICES=1 python test_disco.py \
    --dataset Yelp \
    --data_path ../data \
    --base_model ./ckpt/TIGER \
    --ckpt_path "$CKPT_DIR" \
    --index_file .index.json \
    --max_his_len 20 \
    --ddbc_predict_nums 3 \
    --ddbc_multipliers 19 \
    --ddbc_seed 100 \
    2>&1 | tee ../../logs/letter_yelp_test_$(date +%Y%m%d_%H%M%S).log
