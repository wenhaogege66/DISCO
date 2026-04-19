# CUDA_VISIBLE_DEVICES=0,1 \
# accelerate launch \
#     --main_process_port 12345 \
#     main.py \
#     --model=TIGER \
#     --dataset=AmazonReviews2023 \
#     --category=Industrial_and_Scientific \

# ========== Only Process Data (No Training) ==========
# Uncomment the following line to only download and process data without training:
# Set proxy for HuggingFace downloads
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
CUDA_VISIBLE_DEVICES=0 python main.py --model=TIGER --dataset=AmazonReviews2023 --category=Industrial_and_Scientific --process_only
