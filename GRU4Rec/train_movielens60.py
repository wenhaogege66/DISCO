#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GRU4Rec baseline – MovieLens-60, DDBC evaluation protocol.

Run from workspace root:
    conda activate DDBC && bash GRU4Rec/scripts/train_ml60.sh

Training data   : GRU4Rec/data/ml60/train_ml60.tsv  (auto-built from DISCO train.txt)
Eval data       : DreamRec/data/ml60/{valid,test}_data_items{n}.df
Test candidates : DISCO/datasets/MovieLens-20M/len60/test_candidates_seed1_x{mult}_items{n}.pkl  (shared)
Val  candidates : GRU4Rec/data/ml60/valid_candidates_seed100_x{mult}_items{n}.pkl (auto-cached)
"""

import argparse
import logging
import os
import pickle
import random
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WSPACE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from gru4rec_pytorch import GRU4Rec as _GRU4RecWrapper
from gru4rec_pytorch import GRU4RecModel, SessionDataIterator, IndexedAdagradM

# ---------------------------------------------------------------------------
# Constants (must match DISCO / DreamRec)
# ---------------------------------------------------------------------------
ITEM_NUM  = 17188
SEQ_SIZE  = 60
PAD_TOKEN = ITEM_NUM

DATASET_NAME = "ml60"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='GRU4Rec MovieLens-60 – DDBC eval')
    # Paths
    p.add_argument('--data_dir',     default=os.path.join(SCRIPT_DIR, f'data/{DATASET_NAME}'))
    p.add_argument('--disco_dir',    default=os.path.join(WSPACE_DIR, 'DISCO/datasets/MovieLens-20M/len60'))
    p.add_argument('--dreamrec_dir', default=os.path.join(WSPACE_DIR, f'DreamRec/data/{DATASET_NAME}'))
    p.add_argument('--output_dir',   default=os.path.join(SCRIPT_DIR, f'outputs/{DATASET_NAME}'))
    p.add_argument('--log_dir',      default=os.path.join(SCRIPT_DIR, 'tensorboard'))
    p.add_argument('--log_file',     default=os.path.join(SCRIPT_DIR, f'logs/train_{DATASET_NAME}.log'))
    # Model
    p.add_argument('--layers',              default='64',
                   help='Hidden layer sizes, "/" separated.')
    p.add_argument('--constrained_embedding', type=int, default=1)
    p.add_argument('--dropout_p_embed',  type=float, default=0.0)
    p.add_argument('--dropout_p_hidden', type=float, default=0.0)
    # Training
    p.add_argument('--loss',       default='cross-entropy',
                   choices=['cross-entropy', 'bpr-max'])
    p.add_argument('--epochs',     type=int,   default=50)
    p.add_argument('--batch_size', type=int,   default=512)
    p.add_argument('--lr',         type=float, default=0.01)
    p.add_argument('--momentum',   type=float, default=0.0)
    p.add_argument('--n_sample',   type=int,   default=2048)
    p.add_argument('--sample_alpha', type=float, default=0.5)
    p.add_argument('--bpreg',      type=float, default=1.0)
    p.add_argument('--elu_param',  type=float, default=0.5)
    # Evaluation
    p.add_argument('--eval_freq',  type=int, default=5)
    p.add_argument('--predict_nums',          default='30',
                   help='Comma-separated predict counts')
    p.add_argument('--candidate_multipliers', default='19',
                   help='Comma-separated multipliers')
    p.add_argument('--predict_mode', default='single', choices=['single', 'ar'])
    p.add_argument('--topk',       type=int, default=1)
    p.add_argument('--val_seed',   type=int, default=100)
    # Misc
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed',   type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fmt = '%(asctime)s [%(levelname)s] %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def prepare_train_data(disco_dir, data_dir, logger):
    """Convert DISCO train.txt → GRU4Rec session TSV. Cached at data_dir/train_ml60.tsv."""
    tsv_path = os.path.join(data_dir, 'train_ml60.tsv')
    if os.path.exists(tsv_path):
        logger.info(f'Loading cached train TSV from {tsv_path}')
        return pd.read_csv(tsv_path, sep='\t',
                           dtype={'SessionId': 'int32', 'ItemId': 'int32', 'Time': 'int32'})

    logger.info('Converting DISCO train.txt → GRU4Rec session TSV …')
    txt_path = os.path.join(disco_dir, 'train.txt')
    records = []
    with open(txt_path) as f:
        for line in f:
            parts = [int(x) for x in line.strip().split(',')]
            bundle_id = parts[0]
            items = parts[1:]
            for t, item_id in enumerate(items):
                records.append((bundle_id, item_id, t))

    df = pd.DataFrame(records, columns=['SessionId', 'ItemId', 'Time'])
    df = df.astype({'SessionId': 'int32', 'ItemId': 'int32', 'Time': 'int32'})
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(tsv_path, sep='\t', index=False)
    logger.info(f'Saved {len(df):,} rows → {tsv_path}')
    return df


def prepare_split_tsv(split, disco_dir, data_dir, logger):
    """Convert DISCO {valid,test}.txt → GRU4Rec session TSV."""
    tsv_path = os.path.join(data_dir, f'{split}_ml60.tsv')
    if os.path.exists(tsv_path):
        logger.info(f'Cached {split} TSV found: {tsv_path}')
        return
    logger.info(f'Converting DISCO {split}.txt → {tsv_path} …')
    txt_path = os.path.join(disco_dir, f'{split}.txt')
    records = []
    with open(txt_path) as f:
        for line in f:
            parts = [int(x) for x in line.strip().split(',')]
            bundle_id = parts[0]
            items = parts[1:]
            for t, item_id in enumerate(items):
                records.append((bundle_id, item_id, t))
    df = pd.DataFrame(records, columns=['SessionId', 'ItemId', 'Time'])
    df = df.astype({'SessionId': 'int32', 'ItemId': 'int32', 'Time': 'int32'})
    df.to_csv(tsv_path, sep='\t', index=False)
    logger.info(f'Saved {len(df):,} rows → {tsv_path}')


# ---------------------------------------------------------------------------
# Candidate pool helpers
# ---------------------------------------------------------------------------
def _load_or_build_candidate_pool(labels_list, predict_n, multiplier,
                                   split, val_seed, data_dir, disco_dir, logger):
    """
    test  → load directly from DISCO shared pkl (never rebuild).
    valid → build from val labels + random negatives, cache as pkl.
    """
    if split == 'test':
        pkl = os.path.join(disco_dir,
                           f'test_candidates_seed1_x{multiplier}_items{predict_n}.pkl')
        with open(pkl, 'rb') as f:
            return pickle.load(f)['candidates']

    pkl = os.path.join(data_dir,
                       f'valid_candidates_seed{val_seed}_x{multiplier}_items{predict_n}.pkl')
    if os.path.exists(pkl):
        with open(pkl, 'rb') as f:
            data = pickle.load(f)
            return data['candidates'] if isinstance(data, dict) else data

    logger.info(f'Building valid candidates (predict_n={predict_n}, x{multiplier}) …')
    rng = random.Random(val_seed)
    all_items = set(range(ITEM_NUM))
    candidates = []
    for labels in labels_list:
        labels = list(labels)
        unique_labels = list(dict.fromkeys(labels))
        neg_pool = list(all_items - set(unique_labels))
        n_neg = predict_n * multiplier
        negs = rng.sample(neg_pool, n_neg)
        candidates.append(unique_labels + negs)

    os.makedirs(data_dir, exist_ok=True)
    with open(pkl, 'wb') as f:
        pickle.dump({'candidates': candidates, 'metadata': {
            'seed': val_seed, 'multiplier': multiplier, 'predict_num_items': predict_n}}, f)
    logger.info(f'Cached → {pkl}')
    return candidates


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def _seq_mode_metrics(pred_items, label_list):
    """Counter-based recall / precision / hit_j / hit_full."""
    pred_c  = Counter(pred_items)
    label_c = Counter(label_list)
    inter   = sum(min(pred_c[k], label_c[k]) for k in label_c)

    n_pred  = len(pred_items)
    n_label = len(label_list)
    recall    = inter / n_label if n_label > 0 else 0.0
    precision = inter / n_pred  if n_pred  > 0 else 0.0

    hits = {f'hit_{j}': int(inter >= j) for j in range(1, 6)}
    hits['hit_full'] = int(pred_c == label_c)

    sh = sum(min(pred_c[k], label_c[k]) for k in label_c)
    sm = sh / n_label if n_label > 0 else 0.0

    return recall, precision, hits, sm


def _stepwise_sm_metrics(step_hits):
    """SM / SH / SN from per-step binary hits."""
    sh = sum(step_hits)
    sm = sh / len(step_hits) if step_hits else 0.0
    return {'sm': sm, 'sh': sh, 'sn': sm}


# ---------------------------------------------------------------------------
# GRU hidden-state builder
# ---------------------------------------------------------------------------
@torch.no_grad()
def _build_hidden(model, seq_batch, history_n, device):
    B = seq_batch.shape[0]
    H = [torch.zeros(B, model.layers[i], device=device)
         for i in range(len(model.layers))]

    for t in range(history_n):
        items_t = seq_batch[:, t]
        E = model.Wy(items_t)
        model.hidden_step(E, H, training=False)

    return H


# ---------------------------------------------------------------------------
# evaluate_ddbc
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_ddbc(model, device,
                  predict_nums, multipliers, val_seed,
                  data_dir, disco_dir, dreamrec_dir,
                  writer=None, epoch=None, split='test',
                  predict_mode='single', topk=1,
                  eval_batch_size=512, logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)

    model.eval()
    all_metrics = {}
    primary = None

    for predict_n in predict_nums:
        history_n = SEQ_SIZE - predict_n

        eval_path = os.path.join(dreamrec_dir, f'{split}_data_items{predict_n}.df')
        with open(eval_path, 'rb') as f:
            eval_dict = pickle.load(f)
        seqs_all     = eval_dict['seq']
        len_seqs_all = eval_dict['len_seq']
        labels_all   = eval_dict['labels']
        n_samples    = len(seqs_all)

        for multiplier in multipliers:
            candidates = _load_or_build_candidate_pool(
                labels_all, predict_n, multiplier,
                split, val_seed, data_dir, disco_dir, logger
            )

            total_recall = total_precision = total_sm = 0.0
            total_hits   = {f'hit_{j}': 0 for j in range(1, 6)}
            total_hits['hit_full'] = 0
            total_sh = total_sn = 0.0

            for start in range(0, n_samples, eval_batch_size):
                end   = min(start + eval_batch_size, n_samples)
                b     = end - start

                batch_seqs   = seqs_all[start:end]
                batch_labels = labels_all[start:end]
                seqs   = torch.tensor(
                    np.stack(batch_seqs), dtype=torch.long, device=device)
                labels_list  = batch_labels
                batch_cands  = candidates[start:end]

                H = _build_hidden(model, seqs, history_n, device)

                if predict_mode == 'single':
                    preds_batch, step_hits_batch = _single_predict(
                        model, H[-1], batch_cands, predict_n, topk,
                        labels_list, device
                    )
                else:
                    preds_batch, step_hits_batch = _ar_predict(
                        model, H, batch_cands, predict_n, topk,
                        labels_list, device
                    )

                for i in range(b):
                    r, pr, hits, sm = _seq_mode_metrics(preds_batch[i], list(labels_list[i]))
                    total_recall    += r
                    total_precision += pr
                    total_sm        += sm
                    for k, v in hits.items():
                        total_hits[k] += v
                    sm_metrics = _stepwise_sm_metrics(step_hits_batch[i])
                    total_sh += sm_metrics['sh']
                    total_sn += sm_metrics['sn']

            tag = f'{split}_recall@{predict_n}_x{multiplier}'
            metrics = {
                f'{split}_recall@{predict_n}_x{multiplier}':    total_recall    / n_samples,
                f'{split}_precision@{predict_n}_x{multiplier}': total_precision / n_samples,
                f'{split}_sm@{predict_n}_x{multiplier}':        total_sm        / n_samples,
                f'{split}_sh@{predict_n}_x{multiplier}':        total_sh        / n_samples,
                f'{split}_sn@{predict_n}_x{multiplier}':        total_sn        / n_samples,
            }
            for j in range(1, 6):
                metrics[f'{split}_hit_{j}@{predict_n}_x{multiplier}'] = (
                    total_hits[f'hit_{j}'] / n_samples)
            metrics[f'{split}_hit_full@{predict_n}_x{multiplier}'] = (
                total_hits['hit_full'] / n_samples)

            all_metrics.update(metrics)

            r_val = metrics[f'{split}_recall@{predict_n}_x{multiplier}']
            p_val = metrics[f'{split}_precision@{predict_n}_x{multiplier}']
            logger.info(
                f'[{split}] predict_n={predict_n} x{multiplier:>2d} | '
                f'recall={r_val:.4f}  precision={p_val:.4f}  '
                f'sm={metrics[f"{split}_sm@{predict_n}_x{multiplier}"]:.4f}'
            )

            if writer is not None and epoch is not None:
                for k, v in metrics.items():
                    writer.add_scalar(k, v, epoch)

            if split == 'valid' and predict_n == predict_nums[0] and multiplier == multipliers[0]:
                primary = r_val

    # Print full output_results for the primary config
    if split == 'test':
        predict_n_key  = predict_nums[0]
        multiplier_key = multipliers[0]
        sfx = f'@{predict_n_key}_x{multiplier_key}'
        from collections import OrderedDict
        output = OrderedDict([
            ('recall@1',    round(all_metrics.get(f'test_recall{sfx}', 0.0), 4)),
            ('precision@1', round(all_metrics.get(f'test_precision{sfx}', 0.0), 4)),
            ('hit_1@1',     round(all_metrics.get(f'test_hit_1{sfx}', 0.0), 4)),
            ('hit_2@1',     round(all_metrics.get(f'test_hit_2{sfx}', 0.0), 4)),
            ('hit_3@1',     round(all_metrics.get(f'test_hit_3{sfx}', 0.0), 4)),
            ('hit_4@1',     round(all_metrics.get(f'test_hit_4{sfx}', 0.0), 4)),
            ('hit_5@1',     round(all_metrics.get(f'test_hit_5{sfx}', 0.0), 4)),
            ('hit_full@1',  round(all_metrics.get(f'test_hit_full{sfx}', 0.0), 4)),
            ('sm@1',        round(all_metrics.get(f'test_sm{sfx}', 0.0), 4)),
            ('sh@1',        round(all_metrics.get(f'test_sh{sfx}', 0.0), 4)),
            ('sn@1',        round(all_metrics.get(f'test_sn{sfx}', 0.0), 4)),
        ])
        print(f'output_results {output}')
        logger.info(f'output_results {output}')

    return all_metrics, primary


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def _score_candidates(model, h_batch, batch_cands, device):
    B = h_batch.shape[0]
    max_n = max(len(c) for c in batch_cands)

    cand_tensor = torch.zeros(B, max_n, dtype=torch.long, device=device)
    for i, c in enumerate(batch_cands):
        cand_tensor[i, :len(c)] = torch.tensor(c, dtype=torch.long, device=device)

    O = model.Wy(cand_tensor)
    B_bias = model.By(cand_tensor)

    scores = torch.bmm(O, h_batch.unsqueeze(-1)).squeeze(-1) + B_bias.squeeze(-1)

    for i, c in enumerate(batch_cands):
        if len(c) < max_n:
            scores[i, len(c):] = float('-inf')

    return scores, cand_tensor


def _single_predict(model, h_batch, batch_cands, predict_n, topk, labels_list, device):
    """Single-shot: score all candidates once, take top-predict_n."""
    scores, cand_tensor = _score_candidates(model, h_batch, batch_cands, device)
    top_indices = scores.topk(predict_n, dim=1).indices

    preds_batch     = []
    step_hits_batch = []

    for i in range(h_batch.shape[0]):
        pred_items = [int(cand_tensor[i, j].item()) for j in top_indices[i]]
        preds_batch.append(pred_items)

        top_topk = set(int(cand_tensor[i, j].item())
                       for j in scores[i].topk(min(topk, scores.shape[1])).indices)
        step_hits = [int(lbl in top_topk) for lbl in labels_list[i]]
        step_hits_batch.append(step_hits)

    return preds_batch, step_hits_batch


def _ar_predict(model, H, batch_cands, predict_n, topk, labels_list, device):
    """Autoregressive: predict one item at a time, feed back into GRU."""
    B = H[-1].shape[0]
    remaining = [list(c) for c in batch_cands]
    preds_batch     = [[] for _ in range(B)]
    step_hits_batch = [[] for _ in range(B)]

    for step in range(predict_n):
        scores, cand_tensor = _score_candidates(model, H[-1], remaining, device)

        for i in range(B):
            best_local = int(scores[i].argmax().item())
            best_item  = int(cand_tensor[i, best_local].item())
            preds_batch[i].append(best_item)

            top_topk_local = scores[i].topk(min(topk, scores[i].shape[0])).indices
            top_topk_items = {int(cand_tensor[i, j].item()) for j in top_topk_local}
            true_label = list(labels_list[i])[step] if step < len(labels_list[i]) else -1
            step_hits_batch[i].append(int(true_label in top_topk_items))

        pred_items_t = torch.tensor(
            [preds_batch[i][-1] for i in range(B)], dtype=torch.long, device=device)
        E = model.Wy(pred_items_t)
        model.hidden_step(E, H, training=False)

    return preds_batch, step_hits_batch


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_one_epoch(model, data_iterator, loss_fn, optimizer,
                    layers, batch_size, n_sample, device, logger):
    model.train()
    H = [torch.zeros(batch_size, layers[i], device=device)
         for i in range(len(layers))]
    losses, counts = [], []
    n_valid = batch_size

    def reset_hook(nv, finished_mask, valid_mask):
        nonlocal n_valid
        n_valid = nv
        with torch.no_grad():
            for i in range(len(layers)):
                H[i][finished_mask] = 0
        if nv < len(valid_mask):
            for i in range(len(H)):
                H[i] = H[i][valid_mask]
        return nv < 2 and n_sample == 0

    for in_idx, out_idx in data_iterator(
            enable_neg_samples=(n_sample > 0),
            reset_hook=reset_hook):
        for h in H:
            h.detach_()
        model.zero_grad()
        R = model.forward(in_idx, H, out_idx, training=True)
        L = loss_fn(R, out_idx, n_valid) / batch_size
        L.backward()
        optimizer.step()
        L_val = L.item()
        if not np.isfinite(L_val):
            logger.error('NaN/Inf loss detected — stopping epoch.')
            return float('nan')
        losses.append(L_val)
        counts.append(n_valid)

    return float(np.sum(np.array(losses) * np.array(counts)) / np.sum(counts))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir,    exist_ok=True)

    logger = setup_logging(args.log_file)
    logger.info(f'Args: {vars(args)}')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    layers       = [int(x) for x in args.layers.split('/')]
    predict_nums = [int(x) for x in args.predict_nums.split(',')]
    multipliers  = [int(x) for x in args.candidate_multipliers.split(',')]
    constrained  = bool(args.constrained_embedding)

    # ---- Data ----
    train_df = prepare_train_data(args.disco_dir, args.data_dir, logger)
    prepare_split_tsv('valid', args.disco_dir, args.data_dir, logger)
    prepare_split_tsv('test',  args.disco_dir, args.data_dir, logger)

    train_item_ids = np.sort(train_df['ItemId'].unique()).astype('int32')
    identity_map = pd.Series(
        data=train_item_ids,
        index=train_item_ids,
        name='ItemIdx'
    )
    logger.info(f'Train item vocab: {len(train_item_ids):,} / {ITEM_NUM} '
                f'({ITEM_NUM - len(train_item_ids)} items unseen in train)')

    data_iterator = SessionDataIterator(
        train_df, args.batch_size,
        n_sample=args.n_sample,
        sample_alpha=args.sample_alpha,
        sample_cache_max_size=10_000_000,
        item_key='ItemId', session_key='SessionId', time_key='Time',
        session_order='time',
        device=device,
        itemidmap=identity_map,
    )

    # ---- Model ----
    model = GRU4RecModel(
        ITEM_NUM, layers,
        dropout_p_embed=args.dropout_p_embed,
        dropout_p_hidden=args.dropout_p_hidden,
        embedding=0,
        constrained_embedding=constrained,
    ).to(device)
    logger.info(f'Model: layers={layers}, constrained_embedding={constrained}')

    _wrapper = _GRU4RecWrapper(
        device=device, loss=args.loss,
        bpreg=args.bpreg, elu_param=args.elu_param,
    )
    loss_fn = _wrapper.loss_function

    optimizer = IndexedAdagradM(
        model.parameters(), args.lr, args.momentum
    )

    # ---- TensorBoard ----
    writer = SummaryWriter(log_dir=args.log_dir)

    # ---- Training loop ----
    best_val_recall = -1.0
    best_epoch      = -1

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        avg_loss = train_one_epoch(
            model, data_iterator, loss_fn, optimizer,
            layers, args.batch_size, args.n_sample, device, logger
        )
        dt = time.time() - t0

        if np.isnan(avg_loss):
            logger.error('Training diverged. Exiting.')
            break

        logger.info(f'Epoch {epoch:3d}/{args.epochs} | loss={avg_loss:.6f} | {dt:.1f}s')
        writer.add_scalar('train/loss', avg_loss, epoch)

        if epoch % args.eval_freq == 0:
            logger.info(f'--- Evaluating epoch {epoch} ---')

            val_metrics, val_recall = evaluate_ddbc(
                model, device,
                predict_nums, multipliers, args.val_seed,
                args.data_dir, args.disco_dir, args.dreamrec_dir,
                writer=writer, epoch=epoch, split='valid',
                predict_mode='single',
                topk=args.topk,
                logger=logger,
            )

            evaluate_ddbc(
                model, device,
                predict_nums, multipliers, args.val_seed,
                args.data_dir, args.disco_dir, args.dreamrec_dir,
                writer=writer, epoch=epoch, split='test',
                predict_mode=args.predict_mode,
                topk=args.topk,
                logger=logger,
            )

            if val_recall is not None and val_recall > best_val_recall:
                best_val_recall = val_recall
                best_epoch      = epoch
                ckpt_path = os.path.join(args.output_dir, 'best_model.pt')
                torch.save({
                    'epoch':      epoch,
                    'model_state': model.state_dict(),
                    'val_recall': val_recall,
                    'args':       vars(args),
                }, ckpt_path)
                logger.info(
                    f'*** New best val_recall@30_x19={val_recall:.4f} '
                    f'at epoch {epoch} → saved to {ckpt_path}'
                )

    logger.info(f'Training done. Best val_recall@30_x19={best_val_recall:.4f} at epoch {best_epoch}.')

    # ---- Final test evaluation with best checkpoint ----
    logger.info('=== Final test evaluation (best checkpoint) ===')
    ckpt_path = os.path.join(args.output_dir, 'best_model.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        logger.info(f'Loaded best model from epoch {ckpt["epoch"]}')

    evaluate_ddbc(
        model, device,
        predict_nums, multipliers, args.val_seed,
        args.data_dir, args.disco_dir, args.dreamrec_dir,
        writer=writer, epoch=args.epochs + 1, split='test',
        predict_mode=args.predict_mode,
        topk=args.topk,
        logger=logger,
    )

    writer.close()
    logger.info('Done.')


if __name__ == '__main__':
    main()
