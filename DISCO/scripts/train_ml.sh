#!/bin/bash

# DISCO MovieLens-20M (len60) 训练脚本
# seq_len=60, 3 codebooks × 256 entries, model.length=302

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=/home/sjj/wenhao/DISCO:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
export TRANSFORMERS_OFFLINE=1

RUN_NAME="disco-ml60-$(date +%Y%m%d-%H%M%S)"

# RESUME_CKPT=""

if [ -f "$RESUME_CKPT" ]; then
  echo "恢复训练从: $RESUME_CKPT"
  python /home/sjj/wenhao/DISCO/main.py \
    training.layer_loss_weights.enabled=false \
    loader.batch_size=256 \
    loader.eval_batch_size=256 \
    trainer.max_steps=20000 \
    model=small \
    model.hidden_size=64 \
    data=movielens20m_len60 \
    dataset=MovieLens-20M/len60 \
    +run_name=${RUN_NAME} \
    parameterization=subs \
    seq_len=60 \
    rq_n_codebooks=3 \
    rq_codebook_size=256 \
    model.length=302 \
    swap_ratio=0 \
    eval.compute_generative_perplexity=False \
    sampling.steps=25 \
    sampling.cfg_enabled=true \
    sampling.cfg_encoder=true \
    sampling.cfg_p_drop=0.1 \
    sampling.cfg_w=2.0 \
    use_tensorboard=true \
    checkpointing.resume_ckpt_path="$RESUME_CKPT"
else
  echo "从头开始训练"
  python /home/sjj/wenhao/DISCO/main.py \
    training.layer_loss_weights.enabled=false \
    loader.batch_size=256 \
    loader.eval_batch_size=256 \
    trainer.max_steps=20000 \
    model=small \
    model.hidden_size=64 \
    data=movielens20m_len60 \
    dataset=MovieLens-20M/len60 \
    +run_name=${RUN_NAME} \
    parameterization=subs \
    seq_len=60 \
    rq_n_codebooks=3 \
    rq_codebook_size=256 \
    model.length=302 \
    swap_ratio=0 \
    eval.compute_generative_perplexity=False \
    sampling.steps=25 \
    sampling.cfg_enabled=true \
    sampling.cfg_encoder=true \
    sampling.cfg_p_drop=0.1 \
    sampling.cfg_w=2.0 \
    use_tensorboard=true \
    checkpointing.resume_from_ckpt=false
fi
