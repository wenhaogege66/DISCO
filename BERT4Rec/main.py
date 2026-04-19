import os
import sys

# Run from BERT4Rec/ directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch

from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import setup_train


def train():
    export_root = setup_train(args)
    train_loader, val_loader, test_loader = dataloader_factory(args)

    # args.num_items is set by BertDataloader (= len(smap) = 20033 for Yelp, 1-based)
    # Store 0-based count for DDBC candidate evaluation
    args.item_num_0based = args.num_items   # 20033

    model = model_factory(args)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    trainer.train()


def test():
    from trainers.bert import BERTTrainer, evaluate_ddbc
    from config import STATE_DICT_KEY
    import torch

    export_root = setup_train(args)
    train_loader, val_loader, test_loader = dataloader_factory(args)
    args.item_num_0based = args.num_items

    model = model_factory(args)
    ckpt_path = args.test_model_path or os.path.join(export_root, 'models', 'best_model.pth')
    state = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(state[STATE_DICT_KEY])
    model = model.to(args.device)

    predict_nums = [int(x) for x in args.predict_nums.split(',')]
    multipliers  = [int(x) for x in args.candidate_multipliers.split(',')]
    evaluate_ddbc(model, args, predict_nums, multipliers, args.random_seed,
                  split='test', topk=args.topk, ddbc_data_dir=args.ddbc_data_dir)


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
    else:
        raise ValueError('Invalid mode')
