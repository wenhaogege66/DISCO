"""
LETTER-TIGER training with DDBC-compatible evaluation.

Checkpoint selection: val_recall@3_x19 (replaces default HuggingFace eval loss).
Test evaluation runs automatically after training with the best checkpoint.

Usage:
  torchrun --nproc_per_node=1 --master_port=2315 finetune_disco.py \
      --dataset Yelp --output_dir ./ckpt/Yelp_disco ...
"""

import argparse
import os
import sys

import torch
import transformers
from transformers import (
    T5Tokenizer, T5Config, T5ForConditionalGeneration,
    EarlyStoppingCallback, TrainerCallback, TrainingArguments, Trainer,
)
from modeling_letter import LETTER
from utils import (
    set_seed, ensure_dir, load_datasets,
    parse_global_args, parse_train_args, parse_dataset_args,
)
from collator import Collator
from evaluate_ddbc import build_item_token_map, evaluate_ddbc_letter


# ── DDBC callback ─────────────────────────────────────────────────────────────
class DDBCEvalCallback(TrainerCallback):
    """
    Runs DDBC evaluation at the end of each eval epoch and injects
    'eval_ddbc_recall' into trainer.state so HuggingFace Trainer can use it
    for best-model selection and early stopping.
    """

    def __init__(self, tokenizer, item2token_ids, device,
                 predict_nums, multipliers, seed):
        self.tokenizer      = tokenizer
        self.item2token_ids = item2token_ids
        self.device         = device
        self.predict_nums   = predict_nums
        self.multipliers    = multipliers
        self.seed           = seed

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if model is None:
            return
        # Only run on main process
        if args.local_rank not in (-1, 0):
            return

        _, val_recall = evaluate_ddbc_letter(
            model, self.tokenizer, self.item2token_ids, self.device,
            predict_nums=self.predict_nums,
            multipliers=self.multipliers,
            seed=self.seed,
            split='val',
        )

        # Inject into metrics dict so Trainer's best-model selection sees it.
        # metric_for_best_model='ddbc_recall' → Trainer looks for 'eval_ddbc_recall'.
        if metrics is not None:
            metrics['eval_ddbc_recall'] = val_recall
        if state.log_history:
            state.log_history[-1]['eval_ddbc_recall'] = val_recall


# ── Training ──────────────────────────────────────────────────────────────────
def train(args):
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp        = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    device     = torch.device("cuda", local_rank)

    if local_rank == 0:
        print(vars(args))

    config    = T5Config.from_pretrained(args.base_model)
    tokenizer = T5Tokenizer.from_pretrained(args.base_model, model_max_length=512)
    args.deepspeed = None

    train_data, valid_data = load_datasets(args)
    add_num = tokenizer.add_tokens(train_data.datasets[0].get_new_tokens())
    config.vocab_size = len(tokenizer)

    if local_rank == 0:
        print(f"Added {add_num} new tokens. Vocab size: {len(tokenizer)}")
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)

    # Build item -> token ID map for DDBC scoring
    item2token_ids = build_item_token_map(tokenizer, dataset=args.dataset)

    collator = Collator(args, tokenizer)
    model    = LETTER(config)
    model.set_hyper(args.temperature)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    ddbc_callback = DDBCEvalCallback(
        tokenizer      = tokenizer,
        item2token_ids = item2token_ids,
        device         = device,
        predict_nums   = args.ddbc_predict_nums,
        multipliers    = args.ddbc_multipliers,
        seed           = args.ddbc_seed,
    )

    trainer = Trainer(
        model         = model,
        train_dataset = train_data,
        eval_dataset  = valid_data,
        args          = TrainingArguments(
            seed                        = args.seed,
            per_device_train_batch_size = args.per_device_batch_size,
            per_device_eval_batch_size  = args.per_device_batch_size,
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            warmup_ratio                = args.warmup_ratio,
            num_train_epochs            = args.epochs,
            learning_rate               = args.learning_rate,
            weight_decay                = args.weight_decay,
            lr_scheduler_type           = args.lr_scheduler_type,
            logging_steps               = args.logging_step,
            optim                       = args.optim,
            eval_strategy               = args.save_and_eval_strategy,
            save_strategy               = args.save_and_eval_strategy,
            eval_steps                  = args.save_and_eval_steps,
            save_steps                  = args.save_and_eval_steps,
            output_dir                  = args.output_dir,
            save_total_limit            = 1,
            load_best_model_at_end      = True,
            metric_for_best_model       = 'ddbc_recall',
            greater_is_better           = True,
            ddp_find_unused_parameters  = False if ddp else None,
            eval_delay                  = 1 if args.save_and_eval_strategy == "epoch" else 2000,
            report_to                   = 'tensorboard',
            logging_dir                 = os.path.join('/home/sjj/wenhao/LETTER/tensorboard',
                                                       args.dataset),
        ),
        tokenizer    = tokenizer,
        data_collator= collator,
        callbacks    = [
            ddbc_callback,
            EarlyStoppingCallback(early_stopping_patience=args.patience),
        ],
    )
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(output_dir=args.output_dir)

    # ── Post-training: test evaluation with best checkpoint ───────────────────
    if local_rank == 0:
        print('\n[Test] Running DDBC test evaluation with best checkpoint...')
        evaluate_ddbc_letter(
            trainer.model, tokenizer, item2token_ids, device,
            predict_nums = args.ddbc_predict_nums,
            multipliers  = args.ddbc_multipliers,
            seed         = args.ddbc_seed,
            split        = 'test',
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LETTER-TIGER DDBC')
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)

    # DDBC-specific args
    parser.add_argument('--ddbc_predict_nums', type=int, nargs='+', default=[3])
    parser.add_argument('--ddbc_multipliers',  type=int, nargs='+', default=[19])
    parser.add_argument('--ddbc_seed',         type=int, default=100)
    parser.add_argument('--patience',          type=int, default=20,
                        help='EarlyStopping patience (eval rounds)')

    args = parser.parse_args()
    train(args)
