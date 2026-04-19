# CUDA_VISIBLE_DEVICES=0,1 \
# accelerate launch \
#     --main_process_port 12348 \
#     main.py \
#     --model=TIGER \
#     --dataset=AmazonReviews2014 \
#     --category=Sports_and_Outdoors

# ========== Only Process Data (No Training) ==========
# Set proxy for downloads if needed
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
CUDA_VISIBLE_DEVICES=0 python main.py --model=TIGER --dataset=AmazonReviews2014 --category=Sports_and_Outdoors --process_only
