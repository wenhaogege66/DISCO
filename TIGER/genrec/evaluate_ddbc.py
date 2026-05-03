"""
DDBC-compatible evaluation for TIGER.

Scoring approach: T5 log-probability over candidate semantic tokens.
  - Encoder runs once per sample.
  - All candidates for that sample are batched together in the decoder.
  - Score = sum of log P(token_d | context) for d in 0..n_digit-1.

ID convention:
  DISCO 0-based item ID X  →  TIGER internal ID (X+1)  →  id2item[X+1] (business_id)

Data sources:
  eval data   : /home/sjj/wenhao/DreamRec/data/yelp/{split}_data_items{n}.df
  test cands  : /home/sjj/wenhao/DISCO/datasets/Yelp/test_candidates_seed1_x{mult}_items{n}.pkl
  valid cands : /home/sjj/wenhao/TIGER/data/yelp/valid_candidates_seed{seed}_x{mult}_items{n}.pkl
"""

import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter, OrderedDict
from transformers.modeling_outputs import BaseModelOutput


# ── Path constants ────────────────────────────────────────────────────────────
DISCO_DATA_ROOT = "/home/sjj/wenhao/DISCO/datasets"
TIGER_DATA_ROOT = "/home/sjj/wenhao/TIGER/data"
DEFAULT_DDBC_DATASET = "Yelp"
DEFAULT_ITEM_NUM = 20033


# ── Candidate pool ────────────────────────────────────────────────────────────
def _load_or_build_candidate_pool(labels_list, item_num, multiplier, predict_n, seed, cache_path):
    """
    Build or load a candidate pool identical to DreamRec / DISCO logic.
    All IDs are 0-based DISCO IDs.
    """
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


# ── Tokenisation ──────────────────────────────────────────────────────────────
def tokenize_seq_for_eval(tokenizer, seq_0based, len_seq):
    """
    Convert a DreamRec-format sequence to TIGER input_ids + attention_mask.

    seq_0based : array/list of length seq_size (10), PAD=20033
    len_seq    : int, number of valid (non-PAD) items

    Returns:
        input_ids      : list[int], length = max_token_seq_len
        attention_mask : list[int], length = max_token_seq_len
    """
    max_item_seq_len  = tokenizer.config['max_item_seq_len']   # 20
    n_digit           = tokenizer.n_digit                       # 4
    max_token_seq_len = tokenizer.max_token_seq_len             # 82

    # Valid items (0-based DISCO IDs), take last max_item_seq_len
    valid_items = list(seq_0based[:len_seq])[-max_item_seq_len:]

    # Dummy user token (no user IDs in eval data)
    user_token = tokenizer.base_user_token   # = sum(codebook_sizes) + 1 = 1025

    input_ids = [user_token]
    for disco_id in valid_items:
        tiger_id = int(disco_id) + 1                     # 0-based → 1-based
        biz_id   = tokenizer.id2item[tiger_id]
        tokens   = tokenizer.item2tokens.get(biz_id)
        if tokens is None:
            tokens = (tokenizer.padding_token,) * n_digit
        input_ids.extend(tokens)
    input_ids.append(tokenizer.eos_token)

    actual_len = len(input_ids)
    input_ids.extend([tokenizer.padding_token] * (max_token_seq_len - actual_len))

    item_seq_len   = len(valid_items)
    attended_len   = n_digit * item_seq_len + 2          # user + items + eos
    attention_mask = [1] * attended_len + [0] * (max_token_seq_len - attended_len)

    return input_ids, attention_mask


# ── Candidate scoring ─────────────────────────────────────────────────────────
def score_candidates(model, tokenizer, input_ids, attention_mask, candidate_ids_0based, device):
    """
    Score DISCO 0-based candidate IDs using T5 log-probability.

    Encoder runs once; all candidates are batched in the decoder.
    Score = sum_d log P(token_d | context, token_0..token_{d-1}).

    Returns:
        scores : np.ndarray [n_cands], higher = better; -inf for unknown items
    """
    n_digit = tokenizer.n_digit
    n_cands = len(candidate_ids_0based)

    inp_t  = torch.tensor([input_ids],      dtype=torch.long, device=device)
    mask_t = torch.tensor([attention_mask], dtype=torch.long, device=device)

    with torch.no_grad():
        enc_out = model.t5.get_encoder()(
            input_ids=inp_t,
            attention_mask=mask_t,
            return_dict=True
        )

    # Build decoder inputs: [decoder_start=0, t1, t2, t3, t4] per candidate
    decoder_ids = []
    valid_flags = []
    cand_tokens_list = []
    for disco_id in candidate_ids_0based:
        tiger_id = int(disco_id) + 1
        biz_id   = tokenizer.id2item[tiger_id]
        tokens   = tokenizer.item2tokens.get(biz_id)
        if tokens is None:
            decoder_ids.append([0] + [tokenizer.padding_token] * n_digit)
            valid_flags.append(False)
            cand_tokens_list.append(None)
        else:
            decoder_ids.append([0] + list(tokens))
            valid_flags.append(True)
            cand_tokens_list.append(tokens)

    dec_t = torch.tensor(decoder_ids, dtype=torch.long, device=device)  # [n_cands, 1+n_digit]

    # Expand encoder output (zero-copy view)
    expanded_hidden = enc_out.last_hidden_state.expand(n_cands, -1, -1)
    expanded_mask   = mask_t.expand(n_cands, -1)
    expanded_enc    = BaseModelOutput(last_hidden_state=expanded_hidden)

    with torch.no_grad():
        outputs = model.t5(
            encoder_outputs=expanded_enc,
            attention_mask=expanded_mask,
            decoder_input_ids=dec_t
        )
    # logits: [n_cands, 1+n_digit, vocab_size]
    # logits[:, d, :] predicts the (d+1)-th decoder token, i.e., tokens[d]
    # Gather logits for candidate tokens only (log_softmax normalisation
    # cancels out across candidates → raw logits give identical ranking)
    logits = outputs.logits  # [n_cands, 1+n_digit, vocab_size]
    needed = logits.new_zeros(n_cands, n_digit)
    for i, (is_valid, tokens) in enumerate(zip(valid_flags, cand_tokens_list)):
        if not is_valid:
            continue
        for d in range(n_digit):
            needed[i, d] = logits[i, d, tokens[d]]
    needed_cpu = needed.cpu()  # [n_cands, n_digit] ≈ 9 KB

    scores = np.full(n_cands, -np.inf, dtype=np.float32)
    for i, (is_valid, _) in enumerate(zip(valid_flags, cand_tokens_list)):
        if not is_valid:
            continue
        scores[i] = needed_cpu[i].sum().item()

    return scores


def _load_eval_data_from_txt(disco_dir, split, predict_n):
    file_split = 'valid' if split == 'val' else split
    path = os.path.join(disco_dir, f'{file_split}.txt')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Evaluation txt not found: {path}")

    seq_list = []
    len_seq_list = []
    labels_list = []

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            items = [int(x) for x in parts[1:]]
            if len(items) <= predict_n:
                continue
            seq = items[:-predict_n]
            labels = items[-predict_n:]
            seq_list.append(seq)
            len_seq_list.append(len(seq))
            labels_list.append(labels)

    return seq_list, len_seq_list, labels_list


# ── Main evaluation loop ──────────────────────────────────────────────────────
def evaluate_ddbc_tiger(model, tokenizer, device,
                        predict_nums, multipliers, seed,
                        writer=None, epoch=None, split='val', config=None,
                        predict_mode='ar'):
    """
    DDBC-compatible evaluation for TIGER.

    Returns:
        all_results  : dict[(predict_n, multiplier)] -> metrics dict
        main_recall  : float, recall@3_x19 (primary checkpoint metric)
    """
    all_results = {}
    model.eval()

    cfg = config or {}
    ddbc_dataset = cfg.get('ddbc_dataset', DEFAULT_DDBC_DATASET)
    ddbc_item_num = int(cfg.get('ddbc_item_num', DEFAULT_ITEM_NUM))

    disco_dir = os.path.join(DISCO_DATA_ROOT, ddbc_dataset)
    tiger_cand_dir = os.path.join(TIGER_DATA_ROOT, ddbc_dataset.lower().replace('-', '_').replace('/', '_'))
    os.makedirs(tiger_cand_dir, exist_ok=True)

    for predict_n in predict_nums:
        seq_list, len_seq_list, labels_list = _load_eval_data_from_txt(disco_dir, split, predict_n)
        num_total = len(seq_list)

        for multiplier in multipliers:
            if split == 'test':
                cand_path = os.path.join(
                    disco_dir,
                    f'test_candidates_seed1_x{multiplier}_items{predict_n}.pkl'
                )
            else:
                cand_path = os.path.join(
                    tiger_cand_dir,
                    f'valid_candidates_seed{seed}_x{multiplier}_items{predict_n}.pkl'
                )

            candidate_pool = _load_or_build_candidate_pool(
                labels_list, ddbc_item_num, multiplier, predict_n, seed, cand_path
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
                        model, tokenizer, input_ids, attention_mask, cands, device
                    )

                    # Single-pass: score once, take top-predict_n (fast, no duplicates)
                    # AR mode: predict_n steps, full candidate pool each step (allows duplicates)
                    if predict_mode == 'single':
                        top_indices = np.argsort(scores)[-predict_n:][::-1]
                        pred_items  = [cands[int(i)] for i in top_indices]
                        step_hits   = [1 if t < len(labels) and labels[t] == pred_items[t] else 0
                                       for t in range(predict_n)]
                    else:
                        pred_items = []
                        step_hits  = []
                        cur_valid  = list(seq[:len_seq])  # 0-based valid items
                        for t in range(predict_n):
                            cur_inp, cur_mask = tokenize_seq_for_eval(tokenizer, cur_valid, len(cur_valid))
                            step_scores = score_candidates(
                                model, tokenizer, cur_inp, cur_mask, cands, device
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
        tb_metrics = {}
        for (predict_n, mult), m in all_results.items():
            prefix = f'DDBC_{split}/x{mult}/top{predict_n}'
            for metric_name, val in m.items():
                tb_metrics[f'{prefix}/{metric_name}'] = val
        try:
            writer.log(tb_metrics, step=epoch)
        except Exception:
            pass   # non-fatal if accelerator logging fails

    model.train()

    # Primary metric: recall@3_x19
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
