export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=/home/sjj/wenhao/DISCO:$PYTHONPATH

CKPT="/home/sjj/wenhao/DISCO/outputs/yelp/2026.04.27/204346/checkpoints/best.ckpt"

echo "========================================"
echo "Testing yelp latest: ${CKPT}"
echo "========================================"

python /home/sjj/wenhao/DISCO/main.py \
  mode=rec_eval \
  evaluator.candidate_multiplier=19 \
  evaluator.allow_duplicate_items=true \
  training.layer_loss_weights.enabled=false \
  loader.batch_size=32 \
  loader.eval_batch_size=1 \
  dataset=Yelp \
  evaluator.dataset=Yelp \
  model=small \
  model.hidden_size=64 \
  parameterization=subs \
  backbone=dit \
  rq_n_codebooks=3 \
  rq_codebook_size=256 \
  model.length=52 \
  seq_len=10 \
  swap_ratio=0 \
  eval.predict_num_items=3 \
  eval.checkpoint_path="${CKPT}" \
  sampling.cfg_enabled=true \
  sampling.cfg_encoder=true \
  sampling.cfg_w=2.0

echo "========================================"
echo "测试完成"
echo "========================================"
