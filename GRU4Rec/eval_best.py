#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GRU4Rec – evaluate a saved checkpoint against the DDBC evaluation protocol.

Usage (from workspace root):
    conda activate DDBC
    python GRU4Rec/eval_best.py --ckpt GRU4Rec/outputs/yelp/best_model.pt

Output format matches DISCO:
    output_results OrderedDict([('recall@1', ...), ...])
"""

import argparse
import logging
import os
import sys
from collections import OrderedDict

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WSPACE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from gru4rec_pytorch import GRU4RecModel
from train_yelp import ITEM_NUM, evaluate_ddbc, setup_logging


def parse_args():
    p = argparse.ArgumentParser(description='GRU4Rec eval_best — DDBC protocol')
    p.add_argument('--ckpt',                  required=True,
                   help='Path to best_model.pt')
    p.add_argument('--data_dir',
                   default=os.path.join(SCRIPT_DIR, 'data/yelp'))
    p.add_argument('--disco_dir',
                   default=os.path.join(WSPACE_DIR, 'DISCO/datasets/Yelp'))
    p.add_argument('--dreamrec_dir',
                   default=os.path.join(WSPACE_DIR, 'DreamRec/data/yelp'))
    p.add_argument('--predict_nums',          default='3')
    p.add_argument('--candidate_multipliers', default='19')
    p.add_argument('--predict_mode',          default='ar',
                   choices=['single', 'ar'])
    p.add_argument('--topk',     type=int, default=1)
    p.add_argument('--val_seed', type=int, default=100)
    p.add_argument('--device',   default='cuda:0')
    return p.parse_args()


def main():
    args = parse_args()
    logger = setup_logging(os.path.join(SCRIPT_DIR, 'logs/eval_best.log'))

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    if not os.path.exists(args.ckpt):
        logger.error(f'Checkpoint not found: {args.ckpt}')
        sys.exit(1)

    ckpt = torch.load(args.ckpt, map_location=device)
    saved_args = ckpt['args']
    val_recall_saved = ckpt.get('val_recall', float('nan'))
    logger.info(
        f'Loaded checkpoint from epoch {ckpt["epoch"]} '
        f'(val_recall@3_x19={val_recall_saved:.4f})'
    )

    # Reconstruct model from saved args
    layers      = [int(x) for x in saved_args['layers'].split('/')]
    constrained = bool(saved_args.get('constrained_embedding', 1))
    model = GRU4RecModel(
        ITEM_NUM, layers,
        dropout_p_embed=saved_args.get('dropout_p_embed', 0.0),
        dropout_p_hidden=saved_args.get('dropout_p_hidden', 0.0),
        embedding=0,
        constrained_embedding=constrained,
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    logger.info(f'Model: layers={layers}, constrained_embedding={constrained}')

    predict_nums = [int(x) for x in args.predict_nums.split(',')]
    multipliers  = [int(x) for x in args.candidate_multipliers.split(',')]

    logger.info(
        f'Evaluating: predict_nums={predict_nums}, multipliers={multipliers}, '
        f'predict_mode={args.predict_mode}, topk={args.topk}'
    )

    metrics, _ = evaluate_ddbc(
        model, device,
        predict_nums, multipliers, args.val_seed,
        args.data_dir, args.disco_dir, args.dreamrec_dir,
        writer=None, epoch=None, split='test',
        predict_mode=args.predict_mode,
        topk=args.topk,
        logger=logger,
    )

    # Format output to match DISCO:
    # output_results OrderedDict([('recall@1', ...), ...])
    # Primary config: predict_n=predict_nums[0], multiplier=multipliers[0]
    predict_n  = predict_nums[0]
    multiplier = multipliers[0]
    tk         = args.topk
    sfx        = f'@{predict_n}_x{multiplier}'

    key_map = [
        (f'test_recall{sfx}',    f'recall@{tk}'),
        (f'test_precision{sfx}', f'precision@{tk}'),
        (f'test_hit_1{sfx}',     f'hit_1@{tk}'),
        (f'test_hit_2{sfx}',     f'hit_2@{tk}'),
        (f'test_hit_3{sfx}',     f'hit_3@{tk}'),
        (f'test_hit_4{sfx}',     f'hit_4@{tk}'),
        (f'test_hit_5{sfx}',     f'hit_5@{tk}'),
        (f'test_hit_full{sfx}',  f'hit_full@{tk}'),
        (f'test_sm{sfx}',        f'sm@{tk}'),
        (f'test_sh{sfx}',        f'sh@{tk}'),
        (f'test_sn{sfx}',        f'sn@{tk}'),
    ]

    output = OrderedDict(
        (dst, round(metrics.get(src, 0.0), 4))
        for src, dst in key_map
    )

    print(f'output_results {output}')
    logger.info(f'output_results {output}')


if __name__ == '__main__':
    main()
