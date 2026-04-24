"""
DDBC-compatible evaluation for LETTER-TIGER.

Scoring approach: T5 log-probability over candidate item token strings.
  - Encoder runs once per sample.
  - All candidates for that sample are batched together in the decoder.
  - Score = sum of log P(token_d | context) for d in 0..3.

ID convention:
  DISCO 0-based item ID X  →  LETTER item ID X  (Yelp.inter.json uses 0-based IDs)
  LETTER item X tokens     →  Yelp.index.json[str(X)]  = ['<a_?>','<b_?>','<c_?>','<d_?>']

Data sources:
  eval data   : /home/sjj/wenhao/DreamRec/data/yelp/{split}_data_items{n}.df
  test cands  : /home/sjj/wenhao/DISCO/datasets/Yelp/test_candidates_seed1_x{mult}_items{n}.pkl
  valid cands : /home/sjj/wenhao/LETTER/data/yelp_valid_cands/valid_candidates_seed{seed}_x{mult}_items{n}.pkl
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from collections import Counter, OrderedDict
from transformers.modeling_outputs import BaseModelOutput


# ── Path constants ────────────────────────────────────────────────────────────
DREAMREC_DATA_DIR  = "/home/sjj/wenhao/DreamRec/data/yelp"
DISCO_CAND_DIR     = "/home/sjj/wenhao/DISCO/datasets/Yelp"
LETTER_CAND_DIR    = "/home/sjj/wenhao/LETTER/data/yelp_valid_cands"
LETTER_DATA_DIR    = "/home/sjj/wenhao/LETTER/data/Yelp"
ITEM_NUM           = 20033   # DISCO 0-based item count


# ── Build item → token-ID list mapping ───────────────────────────────────────
def build_item_token_map(tokenizer, dataset='Yelp'):
    """
    Returns:
        item2token_ids : dict[int -> list[int]]
            DISCO 0-based item ID -> list of 4 T5 token IDs
    """
    index_path = os.path.join(LETTER_DATA_DIR, f'{dataset}.index.json')
    with open(index_path, 'r') as f:
        indices = json.load(f)   # str(item_id) -> ['<a_X>','<b_X>','<c_X>','<d_X>']

    item2token_ids = {}
    for item_id_str, tokens in indices.items():
        # Each token string tokenizes to [token_id, EOS]; take only token_id
        token_ids = [tokenizer(t)['input_ids'][0] for t in tokens]
        item2token_ids[int(item_id_str)] = token_ids

    return item2token_ids


# ── Candidate pool ────────────────────────────────────────────────────────────
def _load_or_build_candidate_pool(labels_list, item_num, multiplier, predict_n, seed, cache_path):
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
        random_items = rng.choice(all_ids[mask], size=predict_n * multiplier, replace=False).tolist()
        candidate_pool.append(unique_labels + random_items)

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump({'metadata': {'seed': seed, 'multiplier': multiplier,
                                  'predict_n': predict_n, 'num_samples': len(labels_list)},
                     'candidates': candidate_pool}, f)
    print(f'[Candidate] Saved to {cache_path}')
    return candidate_pool


# ── Metrics ───────────────────────────────────────────────────────────────────
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


# ── Input sequence tokenization ───────────────────────────────────────────────
_INDEX_CACHE = {}   # module-level cache: dataset -> {item_id_str: token_str_list}

def _get_index(dataset='Yelp'):
    if dataset not in _INDEX_CACHE:
        index_path = os.path.join(LETTER_DATA_DIR, f'{dataset}.index.json')
        with open(index_path, 'r') as f:
            _INDEX_CACHE[dataset] = json.load(f)
    return _INDEX_CACHE[dataset]


def tokenize_seq_for_eval(tokenizer, seq_0based, len_seq, max_his_len=20, dataset='Yelp'):
    """
    Convert a DreamRec-format sequence to LETTER-TIGER encoder input.

    seq_0based : array/list of length seq_size (10), PAD=20033
    len_seq    : int, number of valid (non-PAD) items
    max_his_len: int, max history length (default 20, matches LETTER default)

    Returns:
        input_ids      : tensor [1, seq_len]
        attention_mask : tensor [1, seq_len]
    """
    indices = _get_index(dataset)
    valid_items = list(seq_0based[:len_seq])[-max_his_len:]
    history_str = ''.join(''.join(indices[str(disco_id)]) for disco_id in valid_items)

    enc = tokenizer(
        history_str,
        return_tensors='pt',
        padding=False,
        truncation=True,
        max_length=512,
    )
    return enc['input_ids'], enc['attention_mask']


# ── Candidate scoring ─────────────────────────────────────────────────────────
def score_candidates(model, tokenizer, input_ids, attention_mask,
                     candidate_ids_0based, item2token_ids, device):
    """
    Score DISCO 0-based candidate IDs using T5 log-probability.

    Encoder runs once; all candidates are batched in the decoder.
    Score = sum_d log P(token_d | context, token_0..token_{d-1}).

    Returns:
        scores : np.ndarray [n_cands], higher = better; -inf for unknown items
    """
    n_digit = 4
    n_cands = len(candidate_ids_0based)

    inp_t  = input_ids.to(device)
    mask_t = attention_mask.to(device)

    with torch.no_grad():
        enc_out = model.get_encoder()(
            input_ids=inp_t,
            attention_mask=mask_t,
            return_dict=True,
        )

    # Build decoder inputs: [decoder_start=0, t0, t1, t2, t3]
    decoder_ids  = []
    valid_flags  = []
    tokens_list  = []
    for disco_id in candidate_ids_0based:
        token_ids = item2token_ids.get(int(disco_id))
        if token_ids is None:
            decoder_ids.append([0] + [tokenizer.pad_token_id] * n_digit)
            valid_flags.append(False)
            tokens_list.append(None)
        else:
            decoder_ids.append([0] + token_ids)
            valid_flags.append(True)
            tokens_list.append(token_ids)

    dec_t = torch.tensor(decoder_ids, dtype=torch.long, device=device)  # [n_cands, 1+n_digit]

    expanded_hidden = enc_out.last_hidden_state.expand(n_cands, -1, -1)
    expanded_mask   = mask_t.expand(n_cands, -1)
    expanded_enc    = BaseModelOutput(last_hidden_state=expanded_hidden)

    with torch.no_grad():
        outputs = model(
            encoder_outputs=expanded_enc,
            attention_mask=expanded_mask,
            decoder_input_ids=dec_t,
        )

    log_probs = F.log_softmax(outputs.logits, dim=-1)  # [n_cands, 1+n_digit, vocab]

    scores = np.full(n_cands, -np.inf, dtype=np.float32)
    for i, (is_valid, token_ids) in enumerate(zip(valid_flags, tokens_list)):
        if not is_valid:
            continue
        score = 0.0
        for d in range(n_digit):
            score += log_probs[i, d, token_ids[d]].item()
        scores[i] = score

    return scores


# ── Main evaluation loop ──────────────────────────────────────────────────────
def evaluate_ddbc_letter(model, tokenizer, item2token_ids, device,
                         predict_nums, multipliers, seed,
                         writer=None, epoch=None, split='val'):
    """
    DDBC-compatible evaluation for LETTER-TIGER.

    Returns:
        all_results  : dict[(predict_n, multiplier)] -> metrics dict
        main_recall  : float, recall@3_x19 (primary checkpoint metric)
    """
    all_results = {}
    model.eval()

    file_split = 'valid' if split == 'val' else split

    for predict_n in predict_nums:
        data_path = os.path.join(DREAMREC_DATA_DIR, f'{file_split}_data_items{predict_n}.df')
        eval_data    = pd.read_pickle(data_path)
        seq_list     = list(eval_data['seq'].values)
        len_seq_list = list(eval_data['len_seq'].values)
        labels_list  = list(eval_data['labels'].values)
        num_total    = len(seq_list)

        for multiplier in multipliers:
            if split == 'test':
                cand_path = os.path.join(
                    DISCO_CAND_DIR,
                    f'test_candidates_seed1_x{multiplier}_items{predict_n}.pkl'
                )
            else:
                cand_path = os.path.join(
                    LETTER_CAND_DIR,
                    f'valid_candidates_seed{seed}_x{multiplier}_items{predict_n}.pkl'
                )

            candidate_pool = _load_or_build_candidate_pool(
                labels_list, ITEM_NUM, multiplier, predict_n, seed, cand_path
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

                    input_ids, attention_mask = tokenize_seq_for_eval(
                        tokenizer, seq, len_seq
                    )
                    scores = score_candidates(
                        model, tokenizer, input_ids, attention_mask,
                        cands, item2token_ids, device
                    )

                    # AR mode: predict_n steps, full candidate pool each step (allows duplicates)
                    pred_items = []
                    step_hits  = []
                    cur_valid  = list(seq[:len_seq])  # 0-based valid items
                    for t in range(predict_n):
                        cur_inp, cur_mask = tokenize_seq_for_eval(tokenizer, cur_valid, len(cur_valid))
                        step_scores = score_candidates(
                            model, tokenizer, cur_inp, cur_mask,
                            cands, item2token_ids, device
                        )
                        best_idx  = int(np.argmax(step_scores))
                        best_item = cands[best_idx]
                        pred_items.append(best_item)
                        true_label = labels[t] if t < len(labels) else -1
                        step_hits.append(1 if true_label == best_item else 0)
                        cur_valid.append(best_item)

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
                try:
                    writer.add_scalar(f'{prefix}/{metric_name}', val, epoch)
                except Exception:
                    pass

    model.train()

    main_key    = (3, 19) if (3, 19) in all_results else sorted(all_results.keys())[0]
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
