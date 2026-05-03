"""
LETTER test-only evaluation. Loads checkpoint → runs DDBC test eval.
Usage: python test_letter.py
"""

import os
import torch
from transformers import T5Tokenizer, T5Config

from modeling_letter import LETTER
from evaluate_ddbc import build_item_token_map, evaluate_ddbc_letter

CKPT_DIR  = "/home/sjj/wenhao/LETTER/LETTER-TIGER/ckpt/MovieLens-20M_start-ml60-v3"
CKPT_NAME = "checkpoint-353070"
DATASET   = "MovieLens-20M"
PREDICT_N = 30
MULTIPLIER = 19
SEED       = 100


def main():
    device = torch.device("cuda")

    # Tokenizer from saved output directory (already has full vocab)
    tokenizer = T5Tokenizer.from_pretrained(CKPT_DIR)

    # Config and model
    config = T5Config.from_pretrained(os.path.join(CKPT_DIR, CKPT_NAME))
    config.vocab_size = len(tokenizer)
    model = LETTER(config)
    model.set_hyper(1.0)
    model.resize_token_embeddings(len(tokenizer))

    # Load checkpoint weights
    from safetensors.torch import load_file
    ckpt_path = os.path.join(CKPT_DIR, CKPT_NAME, "model.safetensors")
    state_dict = load_file(ckpt_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")

    model.to(device)
    model.eval()

    item2token_ids = build_item_token_map(tokenizer, dataset=DATASET)

    evaluate_ddbc_letter(
        model, tokenizer, item2token_ids, device,
        predict_nums=[PREDICT_N],
        multipliers=[MULTIPLIER],
        seed=SEED,
        split='test',
        predict_mode='single',
        dataset=DATASET,
    )


if __name__ == "__main__":
    main()
