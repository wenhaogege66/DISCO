"""
SASRec training with DDBC-compatible evaluation.

Checkpoint selection: val_recall@3_x19 (replaces default NDCG@10).
Test evaluation runs automatically after training with the best checkpoint.
"""

import os
import time
import torch
import argparse
import ast
from torch.utils.tensorboard import SummaryWriter

from model import SASRec
from utils import data_partition, WarpSampler
from evaluate_ddbc import evaluate_ddbc_sasrec


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset',          required=True)
parser.add_argument('--train_dir',        required=True)
parser.add_argument('--batch_size',       default=256,   type=int)
parser.add_argument('--lr',               default=0.001, type=float)
parser.add_argument('--maxlen',           default=10,    type=int)
parser.add_argument('--hidden_units',     default=64,    type=int)
parser.add_argument('--num_blocks',       default=2,     type=int)
parser.add_argument('--num_epochs',       default=200,   type=int)
parser.add_argument('--num_heads',        default=1,     type=int)
parser.add_argument('--dropout_rate',     default=0.2,   type=float)
parser.add_argument('--l2_emb',           default=0.0,   type=float)
parser.add_argument('--device',           default='cuda', type=str)
parser.add_argument('--norm_first',       action='store_true', default=False)
parser.add_argument('--state_dict_path',  default=None,  type=str)
parser.add_argument('--item_num',         default=0,     type=int,
                    help='Override itemnum (0 = auto-detect from data file)')
# DDBC-specific
parser.add_argument('--eval_interval',    default=5,     type=int,
                    help='Evaluate every N epochs')
parser.add_argument('--patience',         default=25,    type=int,
                    help='Early stopping patience (in eval rounds)')
parser.add_argument('--ddbc_predict_nums', default='[3]', type=str,
                    help='List of predict_n values, e.g. "[3,5]"')
parser.add_argument('--ddbc_multipliers',  default='[19]', type=str,
                    help='List of candidate multipliers, e.g. "[9,19,49,99]"')
parser.add_argument('--ddbc_seed',         default=100,   type=int)
parser.add_argument('--tb_log_dir',        default='',    type=str,
                    help='TensorBoard log dir (default: <folder>/tensorboard)')
parser.add_argument('--mode',              default='train', type=str,
                    help='train or test')

args = parser.parse_args()
args.ddbc_predict_nums = ast.literal_eval(args.ddbc_predict_nums)
args.ddbc_multipliers  = ast.literal_eval(args.ddbc_multipliers)

folder = args.dataset + '_' + args.train_dir
if not os.path.isdir(folder):
    os.makedirs(folder)
with open(os.path.join(folder, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v)
                       for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))


if __name__ == '__main__':
    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    if args.item_num > 0:
        itemnum = args.item_num

    if args.mode == 'test':
        model = SASRec(usernum, itemnum, args).to(args.device)
        ckpt_path = args.state_dict_path
        if ckpt_path is None:
            # find the single .pth file in the folder
            pth_files = [f for f in os.listdir(folder) if f.endswith('.pth')]
            if not pth_files:
                print(f'[Test] No .pth checkpoint found in {folder}')
                exit(1)
            ckpt_path = os.path.join(folder, pth_files[0])
        print(f'[Test] Loading checkpoint: {ckpt_path}')
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device(args.device)))
        evaluate_ddbc_sasrec(
            model, args.maxlen, args.device,
            predict_nums=args.ddbc_predict_nums,
            multipliers=args.ddbc_multipliers,
            seed=args.ddbc_seed,
            writer=None, epoch=0,
            split='test'
        )
        exit(0)

    num_batch = (len(user_train) - 1) // args.batch_size + 1
    cc = sum(len(user_train[u]) for u in user_train)
    print('average sequence length: %.2f' % (cc / len(user_train)))

    sampler = WarpSampler(user_train, usernum, itemnum,
                          batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    model = SASRec(usernum, itemnum, args).to(args.device)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass
    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0

    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(
                torch.load(args.state_dict_path, map_location=torch.device(args.device))
            )
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except Exception as e:
            print(f'Failed to load state_dict: {e}')

    bce_criterion  = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    best_val_recall  = 0.0
    best_epoch       = 0
    best_ckpt_path   = None
    patience_counter = 0

    log_path = os.path.join(folder, 'log.txt')
    f_log = open(log_path, 'w')
    f_log.write('epoch val_recall@3_x19\n')

    t0 = time.time()

    tb_log_dir = args.tb_log_dir if args.tb_log_dir else os.path.join(folder, 'tensorboard')
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f'TensorBoard log dir: {tb_log_dir}')

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        for step in range(num_batch):
            u, seq, pos, neg = sampler.next_batch()
            import numpy as np
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels = torch.ones(pos_logits.shape,  device=args.device)
            neg_labels = torch.zeros(neg_logits.shape, device=args.device)
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.sum(param ** 2)
            loss.backward()
            adam_optimizer.step()
            print(f'loss in epoch {epoch} iteration {step}: {loss.item():.4f}')

        writer.add_scalar('train/loss', loss.item(), epoch)

        if epoch % args.eval_interval == 0:
            t1 = time.time() - t0
            print(f'\n[Epoch {epoch}] time={t1:.1f}s — running DDBC val eval...')
            _, val_recall = evaluate_ddbc_sasrec(
                model, args.maxlen, args.device,
                predict_nums=args.ddbc_predict_nums,
                multipliers=args.ddbc_multipliers,
                seed=args.ddbc_seed,
                writer=writer, epoch=epoch,
                split='val'
            )
            f_log.write(f'{epoch} {val_recall:.6f}\n')
            f_log.flush()
            writer.add_scalar('val/ddbc_recall', val_recall, epoch)

            if val_recall > best_val_recall:
                best_val_recall  = val_recall
                best_epoch       = epoch
                patience_counter = 0
                # Save best checkpoint
                ckpt_name = (f'SASRec.epoch={epoch}.lr={args.lr}'
                             f'.layer={args.num_blocks}.head={args.num_heads}'
                             f'.hidden={args.hidden_units}.maxlen={args.maxlen}.pth')
                ckpt_path = os.path.join(folder, ckpt_name)
                torch.save(model.state_dict(), ckpt_path)
                # Remove previous best checkpoint to save disk space
                if best_ckpt_path is not None and os.path.exists(best_ckpt_path):
                    os.remove(best_ckpt_path)
                best_ckpt_path = ckpt_path
                print(f'[Checkpoint] Saved best (val_recall={val_recall:.4f}) → {ckpt_path}')
            else:
                patience_counter += 1
                print(f'[EarlyStopping] No improvement for {patience_counter}/{args.patience} rounds '
                      f'(best={best_val_recall:.4f} @ epoch {best_epoch})')
                if patience_counter >= args.patience:
                    print(f'[EarlyStopping] Stopping at epoch {epoch}.')
                    break

            t0 = time.time()

    f_log.close()
    sampler.close()
    writer.close()

    # ── Post-training: test evaluation with best checkpoint ───────────────────
    print(f'\n[Training done] Best epoch={best_epoch}, val_recall@3_x19={best_val_recall:.4f}')
    if best_ckpt_path is not None and os.path.exists(best_ckpt_path):
        print(f'[Test] Loading best checkpoint: {best_ckpt_path}')
        model.load_state_dict(
            torch.load(best_ckpt_path, map_location=torch.device(args.device))
        )
        evaluate_ddbc_sasrec(
            model, args.maxlen, args.device,
            predict_nums=args.ddbc_predict_nums,
            multipliers=args.ddbc_multipliers,
            seed=args.ddbc_seed,
            writer=writer, epoch=best_epoch,
            split='test'
        )
    else:
        print('[Test] No checkpoint found, skipping test evaluation.')

    print('Done')
