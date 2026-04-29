"""
DDBC-compatible evaluation for SASRec.

Scoring approach: dot-product between SASRec's final hidden state and candidate item embeddings.
  - model.predict([dummy_uid], [log_seq], cand_ids_1based) → logits [1, n_cands]

ID convention:
  DISCO 0-based item ID X  →  SASRec 1-based item ID (X+1)
  SASRec PAD = 0

Data sources:
  eval data   : /home/sjj/wenhao/DreamRec/data/yelp/{split}_data_items{n}.df
  test cands  : /home/sjj/wenhao/DISCO/datasets/Yelp/test_candidates_seed1_x{mult}_items{n}.pkl
  valid cands : /home/sjj/wenhao/SASRec/data/yelp/valid_candidates_seed{seed}_x{mult}_items{n}.pkl
"""

import os
import pickle
import numpy as np
import torch
from collections import Counter, OrderedDict


# ── Path mapping (dataset → directories) ─────────────────────────────────────
DATASET_PATHS = {
    'Yelp': {
        'dreamrec': '/home/sjj/wenhao/DreamRec/data/yelp',
        'disco':    '/home/sjj/wenhao/DISCO/datasets/Yelp',
        'sasrec':   '/home/sjj/wenhao/SASRec/data/yelp',
        'item_num': 20033,
    },
    'MovieLens60': {
        'dreamrec': '/home/sjj/wenhao/DreamRec/data/ml60',
        'disco':    '/home/sjj/wenhao/DISCO/datasets/MovieLens-20M/len60',
        'sasrec':   '/home/sjj/wenhao/SASRec/data/ml60',
        'item_num': 17188,
    },
}


# ── Candidate pool ────────────────────────────────────────────────────────────
def _load_or_build_candidate_pool(labels_list, item_num, multiplier, predict_n, seed, cache_path):
    """Build or load a candidate pool. All IDs are 0-based DISCO IDs."""
    if os.path.exists(cache_path):
        print(f'[Candidate] Loading from {cache_path}')
        with open(cache_path, 'rb') as f:
            return pickle.load(f)['candidates']

    print(f'[Candidate] Building pool: predict_n={predict_n}, '
          f'multiplier={multiplier}, n_samples={len(labels_list)}')
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


# ── Metrics (identical to DreamRec) ──────────────────────────────────────────
def _seq_mode_metrics(pred_items, label_list):
    pred_counter  = Counter(pred_items)
    label_counter = Counter(label_list)
    intersection  = sum(min(pred_counter[k], label_counter[k]) for k in label_counter)
    total_label   = sum(label_counter.values())
    total_pred    = sum(pred_counter.values())
    recall        = intersection / total_label if total_label > 0 else 0.0
    precision     = intersection / total_pred  if total_pred  > 0 else 0.0
    hits          = {f'hit_{n}': (1 if intersection >= n else 0) for n in range(1, 6)}
    hit_full      = 1 if pred_counter == label_counter else 0
    T             = len(label_list)
    sh            = sum(min(pred_counter[k], label_counter[k]) for k in label_counter)
    sm            = sh / T if T > 0 else 0.0
    return {'recall': recall, 'precision': precision, **hits, 'hit_full': hit_full,
            'sm': sm, 'sh': float(sh), 'sn': sm}


def _stepwise_sm_metrics(step_hits, predict_n):
    sh = sum(step_hits)
    sm = sh / predict_n if predict_n > 0 else 0.0
    return {'sm': sm, 'sh': float(sh), 'sn': sm}


# ── Input sequence construction ───────────────────────────────────────────────
def build_log_seq(seq_0based, len_seq, maxlen):
    """
    Convert a DreamRec-format sequence to SASRec log_seq.

    seq_0based : array/list of length seq_size (10), PAD=20033
    len_seq    : int, number of valid (non-PAD) items
    maxlen     : int, SASRec maxlen

    Returns:
        log_seq : np.ndarray [maxlen], 1-based SASRec item IDs, PAD=0
    """
    valid_items = list(seq_0based[:len_seq])   # 0-based DISCO IDs
    valid_items = valid_items[-maxlen:]         # take last maxlen

    log_seq = np.zeros(maxlen, dtype=np.int32)
    offset  = maxlen - len(valid_items)
    for i, disco_id in enumerate(valid_items):
        log_seq[offset + i] = int(disco_id) + 1   # 0-based → 1-based

    return log_seq


# ── Candidate scoring ─────────────────────────────────────────────────────────
def score_candidates(model, log_seq, candidate_ids_0based):
    """
    Score DISCO 0-based candidate IDs using SASRec dot-product similarity.

    Returns:
        scores : np.ndarray [n_cands], higher = better
    """
    cand_ids_1based = np.array([int(c) + 1 for c in candidate_ids_0based], dtype=np.int32)

    user_ids  = np.array([1])           # dummy (not used in log2feats)
    log_seqs  = log_seq[np.newaxis, :]  # [1, maxlen]

    with torch.no_grad():
        logits = model.predict(user_ids, log_seqs, cand_ids_1based)  # [1, n_cands]

    return logits[0].cpu().numpy()


# ── Main evaluation loop ──────────────────────────────────────────────────────
def evaluate_ddbc_sasrec(model, maxlen, device,
                         predict_nums, multipliers, seed,
                         writer=None, epoch=None, split='val',
                         dataset='Yelp'):
    """
    DDBC-compatible evaluation for SASRec.

    Returns:
        all_results  : dict[(predict_n, multiplier)] -> metrics dict
        main_recall  : float, recall@3_x19 (primary checkpoint metric)
    """
    paths = DATASET_PATHS[dataset]
    dreamrec_dir = paths['dreamrec']
    disco_dir    = paths['disco']
    sasrec_dir   = paths['sasrec']
    item_num     = paths['item_num']

    all_results = {}
    model.eval()

    file_split = 'valid' if split == 'val' else split

    for predict_n in predict_nums:
        data_path = os.path.join(dreamrec_dir, f'{file_split}_data_items{predict_n}.df')
        with open(data_path, 'rb') as f:
            eval_data = pickle.load(f)
        seq_list     = eval_data['seq']
        len_seq_list = eval_data['len_seq']
        labels_list  = eval_data['labels']
        num_total    = len(seq_list)

        for multiplier in multipliers:
            if split == 'test':
                cand_path = os.path.join(
                    disco_dir,
                    f'test_candidates_seed1_x{multiplier}_items{predict_n}.pkl'
                )
            else:
                cand_path = os.path.join(
                    sasrec_dir,
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

            print(f'[DDBC] {split} predict_n={predict_n} x{multiplier} '
                  f'({num_total} samples)...')

            with torch.no_grad():
                for i in range(num_total):
                    seq     = seq_list[i]
                    len_seq = int(len_seq_list[i])
                    labels  = list(labels_list[i])
                    cands   = candidate_pool[i]

                    log_seq = build_log_seq(seq, len_seq, maxlen)

                    # AR mode: predict_n steps, full candidate pool each step (allows duplicates)
                    pred_items = []
                    step_hits  = []
                    cur_log_seq = log_seq.copy()
                    for t in range(predict_n):
                        step_scores = score_candidates(model, cur_log_seq, cands)
                        best_idx  = int(np.argmax(step_scores))
                        best_item = cands[best_idx]
                        pred_items.append(best_item)
                        true_label = labels[t] if t < len(labels) else -1
                        step_hits.append(1 if true_label == best_item else 0)
                        cur_log_seq = np.roll(cur_log_seq, -1)
                        cur_log_seq[-1] = int(best_item) + 1  # 0-based → 1-based

                    m = _seq_mode_metrics(pred_items, labels)
                    sm_metrics = _stepwise_sm_metrics(step_hits, predict_n)
                    m.update(sm_metrics)

                    for k in metric_accum:
                        metric_accum[k] += m[k]
                    n_valid += 1

            result = {k: v / n_valid for k, v in metric_accum.items()}
            all_results[(predict_n, multiplier)] = result

    # ── Print ─────────────────────────────────────────────────────────────────
    print(f'\n[{split.upper()} DDBC metrics]')
    for (predict_n, mult), m in sorted(all_results.items()):
        tag = f'@{predict_n}_x{mult}'
        print(f"  recall{tag}={m['recall']:.4f}  precision{tag}={m['precision']:.4f}  "
              f"hit_1{tag}={m['hit_1']:.4f}  hit_2{tag}={m['hit_2']:.4f}  "
              f"hit_3{tag}={m['hit_3']:.4f}  hit_full{tag}={m['hit_full']:.4f}  "
              f"sm{tag}={m['sm']:.4f}  sh{tag}={m['sh']:.4f}  sn{tag}={m['sn']:.4f}")

    # ── TensorBoard ───────────────────────────────────────────────────────────
    if writer is not None and epoch is not None:
        for (predict_n, mult), m in all_results.items():
            prefix = f'DDBC_{split}/x{mult}/top{predict_n}'
            for metric_name, val in m.items():
                writer.add_scalar(f'{prefix}/{metric_name}', val, epoch)

    model.train()

    main_key    = sorted(all_results.keys())[0] if all_results else (predict_nums[0], multipliers[0])
    main_metrics = all_results.get(main_key, {})
    output_results = OrderedDict([
        ('recall@1',    round(main_metrics.get('recall', 0.0), 4)),
        ('precision@1', round(main_metrics.get('precision', 0.0), 4)),
        ('hit_1@1',     round(main_metrics.get('hit_1', 0.0), 4)),
        ('hit_2@1',     round(main_metrics.get('hit_2', 0.0), 4)),
        ('hit_3@1',     round(main_metrics.get('hit_3', 0.0), 4)),
        ('hit_4@1',     round(main_metrics.get('hit_4', 0.0), 4)),
        ('hit_5@1',     round(main_metrics.get('hit_5', 0.0), 4)),
        ('hit_full@1',  round(main_metrics.get('hit_full', 0.0), 4)),
        ('sm@1',        round(main_metrics.get('sm', 0.0), 4)),
        ('sh@1',        round(main_metrics.get('sh', 0.0), 4)),
        ('sn@1',        round(main_metrics.get('sn', 0.0), 4)),
    ])
    print(f"output_results {output_results}")

    main_recall = all_results.get(main_key, {}).get('recall', 0.0)
    return all_results, main_recall
