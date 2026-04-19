#!/bin/bash
# Train LETTER-TIGER on Yelp_2020 with DDBC-compatible evaluation.
# Checkpoint selection: val_recall@3_x19.
# Test evaluation runs automatically after training with the best checkpoint.
#
# Prerequisites:
#   - Yelp.inter.json and Yelp.index.json already exist in LETTER/data/Yelp/
#   - Base T5 model in LETTER/LETTER-TIGER/ckpt/TIGER/

set -e
cd /home/sjj/wenhao/LETTER/LETTER-TIGER

export WANDB_DISABLED=true
mkdir -p ../../logs

CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=2315 finetune_disco.py \
    --dataset Yelp \
    --data_path ../data \
    --base_model ./ckpt/TIGER \
    --output_dir ./ckpt/Yelp_disco \
    --index_file .index.json \
    --max_his_len 20 \
    --per_device_batch_size 256 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-4 \
    --epochs 200 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type cosine \
    --logging_step 50 \
    --optim adamw_torch \
    --save_and_eval_strategy epoch \
    --temperature 1.0 \
    --ddbc_predict_nums 3 \
    --ddbc_multipliers 19 \
    --ddbc_seed 100 \
    --patience 20 \
    2>&1 | tee ../../logs/letter_yelp_disco_$(date +%Y%m%d_%H%M%S).log
