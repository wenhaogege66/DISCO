import os
import copy
import json
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from loggers import LoggerService, MetricGraphPrinter, RecentModelLogger, BestModelLogger
from config import STATE_DICT_KEY, OPTIMIZER_STATE_DICT_KEY
from utils import AverageMeterSet
from .utils import recalls_and_ndcgs_for_ks


DDBC_CAND_DIR   = "/home/sjj/wenhao/DISCO/datasets/Yelp"
DREAMREC_DATA_DIR = "/home/sjj/wenhao/DreamRec/data/yelp"


# ──────────────────────────────────────────────────────────────────────────────
#  DDBC-aligned evaluation helpers (identical to DreamRec/DiffuRec versions)
# ──────────────────────────────────────────────────────────────────────────────

def _load_or_build_candidate_pool(labels_list, item_num, multiplier, predict_n, seed, cache_path):
    if os.path.exists(cache_path):
        print(f'[Candidate] Loading from {cache_path}')
        with open(cache_path, 'rb') as f:
            return pickle.load(f)['candidates']

    print(f'[Candidate] Building valid pool: predict_n={predict_n}, multiplier={multiplier}')
    rng     = np.random.RandomState(seed)
    all_ids = np.arange(item_num)
    candidate_pool = []
    for label_list in labels_list:
        unique_labels = list(set(label_list))
        mask = np.ones(item_num, dtype=bool)
        for lid in unique_labels:
            mask[lid] = False
        n_random     = predict_n * multiplier
        random_items = rng.choice(all_ids[mask], size=n_random, replace=False).tolist()
        candidate_pool.append(unique_labels + random_items)

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump({'metadata': {'seed': seed, 'multiplier': multiplier,
                                  'predict_n': predict_n, 'num_samples': len(labels_list)},
                     'candidates': candidate_pool}, f)
    print(f'[Candidate] Saved to {cache_path}')
    return candidate_pool


def _seq_mode_metrics(pred_items, label_list):
    pred_counter  = Counter(pred_items)
    label_counter = Counter(label_list)
    intersection  = sum(min(pred_counter[k], label_counter[k]) for k in label_counter)
    total_label   = sum(label_counter.values())
    total_pred    = sum(pred_counter.values())
    recall    = intersection / total_label if total_label > 0 else 0.0
    precision = intersection / total_pred  if total_pred  > 0 else 0.0
    hits      = {f'hit_{n}': (1 if intersection >= n else 0) for n in range(1, 6)}
    hit_full  = 1 if pred_counter == label_counter else 0
    pred_set  = set(pred_items)
    T  = len(label_list)
    sh = sum(1 for y_t in label_list if y_t in pred_set)
    sm = sh / T if T > 0 else 0.0
    return {'recall': recall, 'precision': precision, **hits, 'hit_full': hit_full,
            'sm': sm, 'sh': float(sh), 'sn': sm}


def _stepwise_sm_metrics(step_hits, predict_n):
    sh = sum(step_hits)
    sm = sh / predict_n if predict_n > 0 else 0.0
    return {'sm': sm, 'sh': float(sh), 'sn': sm}


def evaluate_ddbc(model, args, predict_nums, multipliers, seed,
                  writer=None, epoch=None, split='test', topk=1,
                  ddbc_data_dir=None):
    """
    DDBC-aligned multi-item evaluation for BERT4Rec.

    BERT4Rec uses 1-based item IDs internally; DDBC candidates use 0-based IDs.
    This function converts 0-based candidate IDs → 1-based before scoring,
    then converts predicted 1-based IDs → 0-based for metric computation.

    Eval data: {split}_data_items{n}.df from DreamRec/data/yelp/ (0-based).
    Test candidates: DISCO/datasets/Yelp/test_candidates_seed1_x{mult}_items{n}.pkl (0-based).
    """
    if ddbc_data_dir is None:
        ddbc_data_dir = DREAMREC_DATA_DIR
    device    = args.device
    item_num  = args.item_num_0based   # 20033, 0-based space for candidates
    batch_size = 100
    mask_token = item_num + 1          # BERT4Rec MASK token (1-based: 20034)
    max_len    = args.bert_max_len
    all_results = {}

    for predict_n in predict_nums:
        data_path = os.path.join(ddbc_data_dir, f'{split}_data_items{predict_n}.df')
        eval_data    = pd.read_pickle(data_path)
        # seq is in DreamRec format: 0-based IDs padded with PAD=item_num(20033)
        seq_list     = list(eval_data['seq'].values)
        len_seq_list = list(eval_data['len_seq'].values)
        labels_list  = list(eval_data['labels'].values)   # 0-based
        num_total    = len(seq_list)

        for multiplier in multipliers:
            if split == 'test':
                cand_path = os.path.join(
                    DDBC_CAND_DIR,
                    f'test_candidates_seed1_x{multiplier}_items{predict_n}.pkl'
                )
            else:
                cand_dir  = os.path.join(ddbc_data_dir)
                cand_path = os.path.join(
                    cand_dir,
                    f'valid_candidates_seed{seed}_x{multiplier}_items{predict_n}.pkl'
                )

            candidate_pool = _load_or_build_candidate_pool(
                labels_list, item_num, multiplier, predict_n, seed, cand_path
            )

            metric_accum = {'recall': 0., 'precision': 0.,
                            'hit_1': 0., 'hit_2': 0., 'hit_3': 0.,
                            'hit_4': 0., 'hit_5': 0., 'hit_full': 0.,
                            'sm': 0., 'sh': 0., 'sn': 0.}
            n_valid = 0

            model.eval()
            with torch.no_grad():
                n_batches = (num_total + batch_size - 1) // batch_size
                for bi in range(n_batches):
                    s      = slice(bi * batch_size, min((bi + 1) * batch_size, num_total))
                    cands_b = candidate_pool[s.start:s.stop]   # 0-based

                    # ── Build input sequence (BERT4Rec 1-based format, mask at last pos) ──
                    raw_seqs = seq_list[s]
                    batch_seqs = []
                    for raw_seq, slen in zip(raw_seqs, len_seq_list[s]):
                        # raw_seq: 0-based items padded with 20033
                        # convert to 1-based, PAD=0, append MASK token
                        converted = []
                        for x in raw_seq:
                            converted.append(0 if x == item_num else x + 1)
                        seq_1based = converted + [mask_token]
                        seq_1based = seq_1based[-max_len:]
                        pad_len = max_len - len(seq_1based)
                        seq_1based = [0] * pad_len + seq_1based
                        batch_seqs.append(seq_1based)

                    states = torch.LongTensor(batch_seqs).to(device)  # [B, max_len]
                    logits = model(states)       # [B, max_len, vocab]
                    last_scores = logits[:, -1, :]  # [B, vocab]  (vocab = item_num+2)

                    # ── Score candidates (0-based → 1-based) ──
                    max_cand_len = max(len(c) for c in cands_b)
                    padded_cands_1based = []
                    for c in cands_b:
                        c1 = [x + 1 for x in c]  # shift to 1-based
                        c1 += [1] * (max_cand_len - len(c1))  # pad with dummy (won't be selected)
                        padded_cands_1based.append(c1)
                    cand_ids = torch.LongTensor(padded_cands_1based).to(device)  # [B, C]
                    scores_np = last_scores.gather(1, cand_ids).detach().cpu().numpy()  # [B, C]

                    # mask padded positions
                    for j in range(len(cands_b)):
                        actual_len = len(cands_b[j])
                        if actual_len < max_cand_len:
                            scores_np[j, actual_len:] = -np.inf

                    for j in range(len(cands_b)):
                        top_indices  = np.argsort(scores_np[j])[::-1][:predict_n]
                        pred_items   = [cands_b[j][idx] for idx in top_indices]  # 0-based
                        label_items  = list(labels_list[s.start + j])            # 0-based

                        m = _seq_mode_metrics(pred_items, label_items)

                        topk_indices = np.argsort(scores_np[j])[::-1][:topk]
                        topk_items   = [cands_b[j][idx] for idx in topk_indices]
                        step_hits    = [1 if y_t in topk_items else 0 for y_t in label_items]
                        sm_metrics   = _stepwise_sm_metrics(step_hits, predict_n)
                        m.update(sm_metrics)

                        for k in metric_accum:
                            metric_accum[k] += m[k]
                        n_valid += 1

            model.train()

            result = {k: v / n_valid for k, v in metric_accum.items()}
            all_results[(predict_n, multiplier)] = result

    print(f'\n[{split.upper()} DDBC metrics]')
    for (predict_n, mult), m in sorted(all_results.items()):
        tag = f'@{predict_n}_x{mult}'
        print(f"  recall{tag}={m['recall']:.4f}  precision{tag}={m['precision']:.4f}  "
              f"hit_1{tag}={m['hit_1']:.4f}  hit_2{tag}={m['hit_2']:.4f}  "
              f"hit_3{tag}={m['hit_3']:.4f}  hit_full{tag}={m['hit_full']:.4f}  "
              f"sm{tag}={m['sm']:.4f}  sh{tag}={m['sh']:.4f}  sn{tag}={m['sn']:.4f}")

    if writer is not None and epoch is not None:
        for (predict_n, mult), m in all_results.items():
            prefix = f'{split}/x{mult}/top{predict_n}'
            for metric_name, val in m.items():
                writer.add_scalar(f'{prefix}/{metric_name}', val, epoch)

    main_key = (predict_nums[0], multipliers[0])
    return all_results.get(main_key, {}).get('recall', 0.0)


# ──────────────────────────────────────────────────────────────────────────────
#  BERTTrainer
# ──────────────────────────────────────────────────────────────────────────────

class BERTTrainer:
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        self.args         = args
        self.device       = args.device
        self.model        = model.to(self.device)
        self.is_parallel  = args.num_gpu > 1
        if self.is_parallel:
            self.model = nn.DataParallel(self.model)

        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader
        self.export_root  = export_root
        self.num_epochs   = args.num_epochs
        self.metric_ks    = args.metric_ks

        self.ce = nn.CrossEntropyLoss(ignore_index=0)

        if args.optimizer.lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr,
                                       weight_decay=args.weight_decay, momentum=args.momentum)

        self.enable_lr_schedule = args.enable_lr_schedule
        if self.enable_lr_schedule:
            self.lr_scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=args.decay_step, gamma=args.gamma)

        # DDBC eval params
        self.predict_nums  = [int(x) for x in args.predict_nums.split(',')]
        self.multipliers   = [int(x) for x in args.candidate_multipliers.split(',')]
        self.eval_seed     = args.random_seed
        self.ddbc_topk     = args.topk
        self.ddbc_data_dir = args.ddbc_data_dir
        self.eval_freq     = args.eval_freq
        self.patience      = args.patience

        # TensorBoard + checkpoint
        log_dir = Path(export_root) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = Path(export_root) / 'models'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.writer   = SummaryWriter(log_dir=str(log_dir))
        self.ckpt_dir = str(ckpt_dir)

        self.log_period_as_iter = args.log_period_as_iter

    @classmethod
    def code(cls):
        return 'bert'

    def calculate_loss(self, batch):
        seqs, labels = batch
        logits = self.model(seqs)
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        return self.ce(logits, labels)

    def calculate_metrics(self, batch):
        seqs, candidates, labels = batch
        scores = self.model(seqs)
        scores = scores[:, -1, :]
        scores = scores.gather(1, candidates)
        return recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)

    def train(self):
        args = self.args
        best_val_recall = 0.0
        best_epoch      = 0
        best_model      = copy.deepcopy(self.model)
        bad_count       = 0
        accum_iter      = 0

        for epoch in range(self.num_epochs):
            accum_iter = self._train_one_epoch(epoch, accum_iter)

            if (epoch + 1) % self.eval_freq == 0:
                val_recall = evaluate_ddbc(
                    self.model, args,
                    self.predict_nums, self.multipliers, self.eval_seed,
                    writer=self.writer, epoch=epoch, split='valid',
                    topk=self.ddbc_topk, ddbc_data_dir=self.ddbc_data_dir
                )
                self.writer.add_scalar('val/ddbc_recall', val_recall, epoch)

                if val_recall > best_val_recall:
                    best_val_recall = val_recall
                    best_epoch      = epoch
                    bad_count       = 0
                    best_model = copy.deepcopy(self.model)
                    ckpt = {STATE_DICT_KEY: (self.model.module.state_dict() if self.is_parallel
                                             else self.model.state_dict()),
                            OPTIMIZER_STATE_DICT_KEY: self.optimizer.state_dict(),
                            'epoch': epoch, 'val_recall': val_recall}
                    torch.save(ckpt, os.path.join(self.ckpt_dir, 'best_model.pth'))
                    print(f'  [Checkpoint] saved epoch={epoch}, '
                          f'val_recall@{self.predict_nums[0]}_x{self.multipliers[0]}={val_recall:.4f}')
                else:
                    bad_count += 1
                    if bad_count >= self.patience:
                        print(f'Early stopping at epoch {epoch} (patience={self.patience})')
                        break

        # ── Final test evaluation ──
        print('\n========== Final Test Evaluation ==========')
        evaluate_ddbc(
            best_model, args,
            self.predict_nums, self.multipliers, self.eval_seed,
            writer=self.writer, epoch=best_epoch, split='test',
            topk=self.ddbc_topk, ddbc_data_dir=self.ddbc_data_dir
        )
        print(f'Best checkpoint: epoch={best_epoch}, val_recall={best_val_recall:.4f}')
        self.writer.close()

    def test(self):
        print('Test best model with test set!')
        args       = self.args
        best_state = torch.load(os.path.join(self.ckpt_dir, 'best_model.pth'))
        model = copy.deepcopy(self.model)
        model.load_state_dict(best_state[STATE_DICT_KEY])
        evaluate_ddbc(
            model, args,
            self.predict_nums, self.multipliers, self.eval_seed,
            split='test', topk=self.ddbc_topk, ddbc_data_dir=self.ddbc_data_dir
        )

    def _train_one_epoch(self, epoch, accum_iter):
        self.model.train()
        avg_meter = AverageMeterSet()
        tqdm_dl   = tqdm(self.train_loader)

        for batch in tqdm_dl:
            batch_size = batch[0].size(0)
            batch = [x.to(self.device) for x in batch]
            self.optimizer.zero_grad()
            loss = self.calculate_loss(batch)
            loss.backward()
            self.optimizer.step()

            avg_meter.update('loss', loss.item())
            tqdm_dl.set_description(f'Epoch {epoch+1}, loss {avg_meter["loss"].avg:.3f}')
            accum_iter += batch_size

            if accum_iter % self.log_period_as_iter < batch_size and accum_iter != 0:
                self.writer.add_scalar('train/loss', avg_meter['loss'].avg, accum_iter)

        if self.enable_lr_schedule:
            self.lr_scheduler.step()

        return accum_iter
