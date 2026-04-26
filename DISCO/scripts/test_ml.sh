export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=/home/sjj/wenhao/DISCO:$PYTHONPATH

# CKPT="/home/sjj/wenhao/DISCO/outputs/movielens20m_len60/2026.04.24/231520/checkpoints/best.ckpt"
CKPT="/home/sjj/wenhao/DISCO/outputs/movielens20m_len60/2026.04.25/231512/checkpoints/best.ckpt"

echo "========================================"
echo "Testing MovieLens-20M len60: ${CKPT}"
echo "========================================"

python /home/sjj/wenhao/DISCO/main.py \
  mode=rec_eval \
  evaluator.candidate_multiplier=19 \
  evaluator.allow_duplicate_items=true \
  training.layer_loss_weights.enabled=false \
  loader.batch_size=32 \
  loader.eval_batch_size=1 \
  data=movielens20m_len60 \
  dataset=MovieLens-20M/len60 \
  evaluator.dataset=MovieLens-20M/len60 \
  model=small \
  model.hidden_size=64 \
  parameterization=subs \
  backbone=dit \
  rq_n_codebooks=3 \
  rq_codebook_size=256 \
  model.length=302 \
  seq_len=60 \
  swap_ratio=0 \
  sampling.cfg_enabled=true \
  sampling.cfg_encoder=true \
  sampling.cfg_w=2.0 \
  eval.predict_num_items=30 \
  eval.checkpoint_path="${CKPT}"

echo "========================================"
echo "测试完成"
echo "========================================"
