# CUDA_VISIBLE_DEVICES=0,1 \
# accelerate launch \
#     --main_process_port 12347 \
#     main.py \
#     --model=TIGER \
#     --dataset=Yelp \
#     --category=Yelp \

# ========== Only Process Data (No Training) ==========
# Uncomment the following line to only download and process data without training:
CUDA_VISIBLE_DEVICES=0 python main.py --model=TIGER --dataset=Yelp --category=Yelp --process_only
