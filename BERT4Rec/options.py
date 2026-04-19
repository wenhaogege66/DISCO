from templates import set_template
from datasets import DATASETS
from dataloaders import DATALOADERS
from models import MODELS
from trainers import TRAINERS

import argparse


parser = argparse.ArgumentParser(description='RecPlay')

parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--template', type=str, default=None)
parser.add_argument('--test_model_path', type=str, default=None)

# Dataset
parser.add_argument('--dataset_code', type=str, default='ml-20m', choices=DATASETS.keys())
parser.add_argument('--min_rating', type=int, default=4)
parser.add_argument('--min_uc', type=int, default=5)
parser.add_argument('--min_sc', type=int, default=0)
parser.add_argument('--split', type=str, default='leave_one_out')
parser.add_argument('--dataset_split_seed', type=int, default=98765)
parser.add_argument('--eval_set_size', type=int, default=500)

# Dataloader
parser.add_argument('--dataloader_code', type=str, default='bert', choices=DATALOADERS.keys())
parser.add_argument('--dataloader_random_seed', type=float, default=0.0)
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=64)

# NegativeSampler
parser.add_argument('--train_negative_sampler_code', type=str, default='random', choices=['popular', 'random'])
parser.add_argument('--train_negative_sample_size', type=int, default=100)
parser.add_argument('--train_negative_sampling_seed', type=int, default=None)
parser.add_argument('--test_negative_sampler_code', type=str, default='random', choices=['popular', 'random'])
parser.add_argument('--test_negative_sample_size', type=int, default=100)
parser.add_argument('--test_negative_sampling_seed', type=int, default=None)

# Trainer
parser.add_argument('--trainer_code', type=str, default='bert', choices=TRAINERS.keys())
parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
parser.add_argument('--num_gpu', type=int, default=1)
parser.add_argument('--device_idx', type=str, default='0')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'])
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--momentum', type=float, default=None)
parser.add_argument('--decay_step', type=int, default=15)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--log_period_as_iter', type=int, default=12800)
parser.add_argument('--metric_ks', nargs='+', type=int, default=[10, 20, 50])
parser.add_argument('--best_metric', type=str, default='NDCG@10')
parser.add_argument('--enable_lr_schedule', type=bool, default=True)
parser.add_argument('--find_best_beta', type=bool, default=False)
parser.add_argument('--total_anneal_steps', type=int, default=2000)
parser.add_argument('--anneal_cap', type=float, default=0.2)

# Model
parser.add_argument('--model_code', type=str, default='bert', choices=MODELS.keys())
parser.add_argument('--model_init_seed', type=int, default=None)
parser.add_argument('--bert_max_len', type=int, default=None)
parser.add_argument('--bert_num_items', type=int, default=None)
parser.add_argument('--bert_hidden_units', type=int, default=None)
parser.add_argument('--bert_num_blocks', type=int, default=None)
parser.add_argument('--bert_num_heads', type=int, default=None)
parser.add_argument('--bert_dropout', type=float, default=None)
parser.add_argument('--bert_mask_prob', type=float, default=None)
parser.add_argument('--dae_num_items', type=int, default=None)
parser.add_argument('--dae_num_hidden', type=int, default=0)
parser.add_argument('--dae_hidden_dim', type=int, default=600)
parser.add_argument('--dae_latent_dim', type=int, default=200)
parser.add_argument('--dae_dropout', type=float, default=0.5)
parser.add_argument('--vae_num_items', type=int, default=None)
parser.add_argument('--vae_num_hidden', type=int, default=0)
parser.add_argument('--vae_hidden_dim', type=int, default=600)
parser.add_argument('--vae_latent_dim', type=int, default=200)
parser.add_argument('--vae_dropout', type=float, default=0.5)

# Experiment
parser.add_argument('--experiment_dir', type=str, default='experiments')
parser.add_argument('--experiment_description', type=str, default='test')

# DDBC-aligned evaluation
parser.add_argument('--predict_nums', type=str, default='3,5',
                    help='Comma-sep predict_n values, e.g. "3,5"')
parser.add_argument('--candidate_multipliers', type=str, default='9,19,49,99',
                    help='Comma-sep candidate multipliers')
parser.add_argument('--topk', type=int, default=1,
                    help='Per-position top-K for SM@K metric')
parser.add_argument('--ddbc_data_dir', type=str,
                    default='/home/sjj/wenhao/DreamRec/data/yelp',
                    help='Dir with {valid,test}_data_items{n}.df (from DreamRec)')
parser.add_argument('--eval_freq', type=int, default=5,
                    help='Run DDBC eval every this many epochs')
parser.add_argument('--patience', type=int, default=10,
                    help='Early stopping patience in eval rounds')
parser.add_argument('--random_seed', type=int, default=100,
                    help='Seed for valid candidate pool construction')

args = parser.parse_args()
set_template(args)
