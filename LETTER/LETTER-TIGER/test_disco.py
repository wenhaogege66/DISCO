"""
Standalone DDBC test evaluation for a saved LETTER-TIGER checkpoint.

Usage:
  python test_disco.py \
      --dataset Yelp \
      --data_path ../data \
      --base_model ./ckpt/TIGER \
      --ckpt_path ./ckpt/Yelp_disco \
      --index_file .index.json \
      --ddbc_predict_nums 3 \
      --ddbc_multipliers 19 \
      --ddbc_seed 100
"""

import argparse
import os
import torch
from transformers import T5Tokenizer, T5Config
from modeling_letter import LETTER
from utils import set_seed, load_datasets, parse_global_args, parse_dataset_args
from evaluate_ddbc import build_item_token_map, evaluate_ddbc_letter


def parse_args():
    parser = argparse.ArgumentParser()
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser.add_argument('--ckpt_path',         type=str, required=True)
    parser.add_argument('--gpu_id',            type=int, default=0)
    parser.add_argument('--ddbc_predict_nums', type=int, nargs='+', default=[3])
    parser.add_argument('--ddbc_multipliers',  type=int, nargs='+', default=[19])
    parser.add_argument('--ddbc_seed',         type=int, default=100)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    set_seed(42)

    device = torch.device('cuda', args.gpu_id)

    tokenizer = T5Tokenizer.from_pretrained(args.base_model, model_max_length=512)

    # Load datasets just to get new tokens
    train_data, _ = load_datasets(args)
    tokenizer.add_tokens(train_data.datasets[0].get_new_tokens())

    config = T5Config.from_pretrained(args.ckpt_path)
    config.vocab_size = len(tokenizer)

    model = LETTER(config)
    model.resize_token_embeddings(len(tokenizer))

    # Load saved weights
    import glob
    ckpt_files = glob.glob(os.path.join(args.ckpt_path, 'pytorch_model*.bin'))
    if not ckpt_files:
        # Try safetensors
        from safetensors.torch import load_file
        st_files = glob.glob(os.path.join(args.ckpt_path, 'model*.safetensors'))
        state_dict = {}
        for f in st_files:
            state_dict.update(load_file(f))
    else:
        import torch
        state_dict = {}
        for f in ckpt_files:
            state_dict.update(torch.load(f, map_location='cpu'))

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    item2token_ids = build_item_token_map(tokenizer, dataset=args.dataset)

    print(f'[Test] Running DDBC test evaluation from {args.ckpt_path}')
    evaluate_ddbc_letter(
        model, tokenizer, item2token_ids, device,
        predict_nums = args.ddbc_predict_nums,
        multipliers  = args.ddbc_multipliers,
        seed         = args.ddbc_seed,
        split        = 'test',
    )
