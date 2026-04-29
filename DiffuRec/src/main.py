import os
import random
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import logging
import time
import pickle
from utils import Data_Train, Data_Val, Data_Test, Data_CHLS
from model import create_model_diffu, Att_Diffuse_model
from trainer import model_train, LSHT_inference
from collections import Counter


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                    help='train: full training; test: load checkpoint and evaluate on test set')
parser.add_argument('--ckpt_path', type=str, default='',
                    help='Checkpoint path for --mode test (default: save_dir/best_model.pt)')
parser.add_argument('--dataset', default='yelp', help='Dataset name; yelp uses DISCO/DreamRec data')
parser.add_argument('--data_path', default='../data/yelp/dataset.pkl', help='Path to dataset.pkl')
parser.add_argument('--log_file', default='../log/', help='log dir path')
parser.add_argument('--random_seed', type=int, default=1997, help='Random seed')
parser.add_argument('--max_len', type=int, default=10, help='The max length of sequence (Yelp: 10)')
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--num_gpu', type=int, default=1, help='Number of GPU')
parser.add_argument('--batch_size', type=int, default=512, help='Batch Size')
parser.add_argument("--hidden_size", default=64, type=int, help="hidden size of model (DreamRec uses 64)")
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout of representation')
parser.add_argument('--emb_dropout', type=float, default=0.3, help='Dropout of item embedding')
parser.add_argument("--hidden_act", default="gelu", type=str)
parser.add_argument('--num_blocks', type=int, default=4, help='Number of Transformer blocks')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs for training')
parser.add_argument('--decay_step', type=int, default=100, help='Decay step for StepLR')
parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for StepLR')
parser.add_argument('--metric_ks', nargs='+', type=int, default=[5, 10, 20], help='ks for original HR/NDCG@k (unused in DDBC eval)')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'])
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--loss_lambda', type=float, default=0.001, help='loss weight for diffusion')
parser.add_argument('--weight_decay', type=float, default=0, help='L2 regularization')
parser.add_argument('--momentum', type=float, default=None, help='SGD momentum')
parser.add_argument('--schedule_sampler_name', type=str, default='lossaware', help='Diffusion for t generation')
parser.add_argument('--diffusion_steps', type=int, default=32, help='Diffusion step')
parser.add_argument('--lambda_uncertainty', type=float, default=0.001, help='uncertainty weight')
parser.add_argument('--noise_schedule', default='trunc_lin', help='Beta generation')
parser.add_argument('--rescale_timesteps', default=True, help='rescal timesteps')
parser.add_argument('--eval_interval', type=int, default=20, help='the number of epoch to eval')
parser.add_argument('--eval_start_epoch', type=int, default=0, help='Skip eval before this epoch')
parser.add_argument('--patience', type=int, default=5, help='the number of epoch to wait before early stop')
parser.add_argument('--description', type=str, default='DiffuRec_yelp', help='Model brief introduction')
parser.add_argument('--predict_mode', type=str, default='ar', choices=['single', 'ar'],
                    help='single: score once, take top-N (fast); ar: step-by-step (allows duplicates)')
# DDBC-aligned evaluation arguments
parser.add_argument('--predict_nums', type=str, default='3,5', help='Comma-sep predict_n values, e.g. "3,5"')
parser.add_argument('--candidate_multipliers', type=str, default='9,19,49,99', help='Comma-sep multipliers')
parser.add_argument('--topk', type=int, default=1, help='Per-position top-K for SM@K metric')
parser.add_argument('--ddbc_data_dir', type=str, default='/home/sjj/wenhao/DreamRec/data/yelp',
                    help='Dir containing {valid,test}_data_items{n}.df (reused from DreamRec)')
parser.add_argument('--tb_log_dir', type=str, default='', help='TensorBoard log dir (empty=auto)')
parser.add_argument('--save_dir', type=str, default='', help='Checkpoint save dir (empty=auto)')
parser.add_argument('--long_head', default=False, help='Long and short sequence, head and long-tail items analysis')
args = parser.parse_args()

print(args)

if not os.path.exists(args.log_file):
    os.makedirs(args.log_file)
dataset_log_dir = os.path.join(args.log_file, args.dataset)
if not os.path.exists(dataset_log_dir):
    os.makedirs(dataset_log_dir)

logging.basicConfig(level=logging.INFO,
                    filename=os.path.join(dataset_log_dir, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '.log'),
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s',
                    filemode='w')
logger = logging.getLogger(__name__)
logger.info(args)


def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def item_num_create(args, item_num):
    args.item_num = item_num
    return args


def cold_hot_long_short(data_raw, dataset_name):
    item_list = []
    len_list = []
    target_item = []

    for id_temp in data_raw['train']:
        temp_list = data_raw['train'][id_temp] + data_raw['val'][id_temp] + data_raw['test'][id_temp]
        len_list.append(len(temp_list))
        target_item.append(data_raw['test'][id_temp][0])
        item_list += temp_list
    item_num_count = Counter(item_list)
    split_num = np.percentile(list(item_num_count.values()), 80)
    cold_item, hot_item = [], []
    for item_num_temp in item_num_count.items():
        if item_num_temp[1] < split_num:
            cold_item.append(item_num_temp[0])
        else:
            hot_item.append(item_num_temp[0])
    cold_ids, hot_ids = [], []
    cold_list, hot_list = [], []
    for id_temp, item_temp in enumerate(data_raw['test'].values()):
        if item_temp[0] in hot_item:
            hot_ids.append(id_temp)
            hot_list.append(data_raw['train'][id_temp] + data_raw['val'][id_temp] + data_raw['test'][id_temp])
        else:
            cold_ids.append(id_temp)
            cold_list.append(data_raw['train'][id_temp] + data_raw['val'][id_temp] + data_raw['test'][id_temp])
    cold_hot_dict = {'hot': hot_list, 'cold': cold_list}

    len_short    = np.percentile(len_list, 20)
    len_midshort = np.percentile(len_list, 40)
    len_midlong  = np.percentile(len_list, 60)
    len_long     = np.percentile(len_list, 80)

    len_seq_dict = {'short': [], 'mid_short': [], 'mid': [], 'mid_long': [], 'long': []}
    for id_temp, len_temp in enumerate(len_list):
        temp_seq = data_raw['train'][id_temp] + data_raw['val'][id_temp] + data_raw['test'][id_temp]
        if len_temp <= len_short:
            len_seq_dict['short'].append(temp_seq)
        elif len_short < len_temp <= len_midshort:
            len_seq_dict['mid_short'].append(temp_seq)
        elif len_midshort < len_temp <= len_midlong:
            len_seq_dict['mid'].append(temp_seq)
        elif len_midlong < len_temp <= len_long:
            len_seq_dict['mid_long'].append(temp_seq)
        else:
            len_seq_dict['long'].append(temp_seq)
    return cold_hot_dict, len_seq_dict, split_num, [len_short, len_midshort, len_midlong, len_long], len_list, list(item_num_count.values())


def main(args):
    fix_random_seed_as(args.random_seed)

    with open(args.data_path, 'rb') as f:
        data_raw = pickle.load(f)

    args = item_num_create(args, len(data_raw['smap']))  # item_num = 20033

    # Auto-set tb_log_dir and save_dir if not provided
    if not args.tb_log_dir:
        args.tb_log_dir = f'../tensorboard/{args.dataset}/{args.description}'
    if not args.save_dir:
        args.save_dir = f'../outputs/{args.dataset}/{args.description}'

    if args.mode == 'test':
        from trainer import evaluate_ddbc
        diffu_rec = create_model_diffu(args)
        model = Att_Diffuse_model(diffu_rec, args)
        ckpt_path = args.ckpt_path or os.path.join(args.save_dir, 'best_model.pt')
        state = torch.load(ckpt_path, map_location=args.device)
        model.load_state_dict(state['model_state_dict'])
        model = model.to(args.device)
        model.eval()

        predict_nums = [int(x) for x in args.predict_nums.split(',')]
        multipliers  = [int(x) for x in args.candidate_multipliers.split(',')]
        evaluate_ddbc(model, args, predict_nums, multipliers, args.random_seed,
                      split='test', topk=args.topk, ddbc_data_dir=args.ddbc_data_dir,
                      predict_mode='ar')
        return

    tra_data      = Data_Train(data_raw['train'], args)
    val_data      = Data_Val(data_raw['train'], data_raw['val'], args)
    test_data     = Data_Test(data_raw['train'], data_raw['val'], data_raw['test'], args)
    tra_data_loader  = tra_data.get_pytorch_dataloaders()
    val_data_loader  = val_data.get_pytorch_dataloaders()
    test_data_loader = test_data.get_pytorch_dataloaders()

    diffu_rec = create_model_diffu(args)
    rec_diffu_joint_model = Att_Diffuse_model(diffu_rec, args)

    best_model, test_results = model_train(
        tra_data_loader, val_data_loader, test_data_loader,
        rec_diffu_joint_model, args, logger
    )

    if args.long_head:
        cold_hot_dict, len_seq_dict, split_hotcold, split_length, list_len, list_num = cold_hot_long_short(data_raw, args.dataset)
        for label, data_list in [('Cold', cold_hot_dict['cold']), ('Hot', cold_hot_dict['hot']),
                                  ('Short', len_seq_dict['short']), ('Long', len_seq_dict['long'])]:
            d = Data_CHLS(data_list, args)
            print(f'--------------{label}-----------------------')
            LSHT_inference(best_model, args, d.get_pytorch_dataloaders())


if __name__ == '__main__':
    main(args)
