import torch.nn as nn
import torch.optim as optim
import datetime
import torch
import numpy as np
import copy
import time
import pickle
import os
import pandas as pd
from collections import Counter, OrderedDict


DDBC_CAND_DIR = "/home/sjj/wenhao/DISCO/datasets/Yelp"
DREAMREC_DATA_DIR = "/home/sjj/wenhao/DreamRec/data/yelp"


def optimizers(model, args):
    if args.optimizer.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        raise ValueError


def cal_hr(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = label == topk_predict
    hr = [hit[:, :ks[i]].sum().item()/label.size()[0] for i in range(len(ks))]
    return hr


def cal_ndcg(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = (label == topk_predict).int()
    ndcg = []
    for k in ks:
        max_dcg = dcg(torch.tensor([1] + [0] * (k-1)))
        predict_dcg = dcg(hit[:, :k])
        ndcg.append((predict_dcg/max_dcg).mean().item())
    return ndcg


def dcg(hit):
    log2 = torch.log2(torch.arange(1, hit.size()[-1] + 1) + 1).unsqueeze(0)
    rel = (hit/log2).sum(dim=-1)
    return rel


def hrs_and_ndcgs_k(scores, labels, ks):
    metrics = {}
    ndcg = cal_ndcg(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    hr = cal_hr(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    for k, ndcg_temp, hr_temp in zip(ks, ndcg, hr):
        metrics['HR@%d' % k] = hr_temp
        metrics['NDCG@%d' % k] = ndcg_temp
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
#  DDBC-aligned evaluation helpers (mirrored from DreamRec/DreamRec.py)
# ──────────────────────────────────────────────────────────────────────────────

def _load_or_build_candidate_pool(labels_list, item_num, multiplier, predict_n, seed, cache_path):
    if os.path.exists(cache_path):
        print(f'[Candidate] Loading from {cache_path}')
        with open(cache_path, 'rb') as f:
            return pickle.load(f)['candidates']

    print(f'[Candidate] Building valid pool: predict_n={predict_n}, multiplier={multiplier}, '
          f'n_samples={len(labels_list)}')
    rng = np.random.RandomState(seed)
    all_ids = np.arange(item_num)
    candidate_pool = []
    for label_list in labels_list:
        unique_labels = list(set(label_list))
        mask = np.ones(item_num, dtype=bool)
        for lid in unique_labels:
            mask[lid] = False
        n_random = predict_n * multiplier
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
    intersection = sum(min(pred_counter[k], label_counter[k]) for k in label_counter)
    total_label = sum(label_counter.values())
    total_pred  = sum(pred_counter.values())
    recall    = intersection / total_label if total_label > 0 else 0.0
    precision = intersection / total_pred  if total_pred  > 0 else 0.0
    hits = {f'hit_{n}': (1 if intersection >= n else 0) for n in range(1, 6)}
    hit_full = 1 if pred_counter == label_counter else 0
    T = len(label_list)
    sh = sum(min(pred_counter[k], label_counter[k]) for k in label_counter)
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
    DDBC-aligned multi-item evaluation for DiffuRec.

    Eval data is reused from DreamRec/data/yelp/{split}_data_items{n}.df.
    Candidates:
      - test : DISCO/datasets/Yelp/test_candidates_seed1_x{mult}_items{n}.pkl
      - valid : DiffuRec/data/yelp/valid_candidates_seed{seed}_x{mult}_items{n}.pkl (auto-built)
    """
    if ddbc_data_dir is None:
        ddbc_data_dir = DREAMREC_DATA_DIR
    device = args.device
    item_num = args.item_num
    batch_size = 100
    all_results = {}

    for predict_n in predict_nums:
        data_path = os.path.join(ddbc_data_dir, f'{split}_data_items{predict_n}.df')
        eval_data = pd.read_pickle(data_path)
        seq_list    = list(eval_data['seq'].values)
        len_seq_list = list(eval_data['len_seq'].values)
        labels_list  = list(eval_data['labels'].values)
        num_total = len(seq_list)

        for multiplier in multipliers:
            if split == 'test':
                cand_path = os.path.join(
                    DDBC_CAND_DIR,
                    f'test_candidates_seed1_x{multiplier}_items{predict_n}.pkl'
                )
            else:
                cand_dir = os.path.join(os.path.dirname(ddbc_data_dir), 'yelp')
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
                    s = slice(bi * batch_size, min((bi + 1) * batch_size, num_total))
                    cands_b = candidate_pool[s.start:s.stop]
                    # Build mutable sequence lists (0-based, PAD=item_num)
                    cur_seqs = [list(seq_list[s.start + j]) for j in range(len(cands_b))]
                    B = len(cands_b)
                    preds_batch = [[] for _ in range(B)]
                    step_hits_batch = [[] for _ in range(B)]

                    # AR mode: predict_n steps, full candidate pool each step (allows duplicates)
                    for t in range(predict_n):
                        states = torch.LongTensor(np.array(cur_seqs)).to(device)

                        max_cand_len = max(len(c) for c in cands_b)
                        padded_cands = [c + [0] * (max_cand_len - len(c)) for c in cands_b]
                        cand_ids = torch.LongTensor(np.array(padded_cands)).to(device)

                        scores_np = model.predict_ddbc(states, candidate_ids=cand_ids).detach().cpu().numpy()

                        for j in range(B):
                            actual_len = len(cands_b[j])
                            if actual_len < max_cand_len:
                                scores_np[j, actual_len:] = -np.inf
                            best_idx  = int(np.argmax(scores_np[j]))
                            best_item = cands_b[j][best_idx]
                            preds_batch[j].append(best_item)
                            true_label = list(labels_list[s.start + j])[t] if t < len(labels_list[s.start + j]) else -1
                            step_hits_batch[j].append(1 if true_label == best_item else 0)
                            # Shift sequence left, append predicted item
                            cur_seqs[j] = cur_seqs[j][1:] + [best_item]

                    for j in range(B):
                        pred_items  = preds_batch[j]
                        label_items = list(labels_list[s.start + j])

                        m = _seq_mode_metrics(pred_items, label_items)
                        sm_metrics = _stepwise_sm_metrics(step_hits_batch[j], predict_n)
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
    return all_results.get(main_key, {}).get('recall', 0.0)


# ──────────────────────────────────────────────────────────────────────────────

def LSHT_inference(model_joint, args, data_loader):
    device = args.device
    model_joint = model_joint.to(device)
    with torch.no_grad():
        test_metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
        test_metrics_dict_mean = {}
        for test_batch in data_loader:
            test_batch = [x.to(device) for x in test_batch]
            scores_rec, rep_diffu, _, _, _, _ = model_joint(test_batch[0], test_batch[1], train_flag=False)
            scores_rec_diffu = model_joint.diffu_rep_pre(rep_diffu)
            metrics = hrs_and_ndcgs_k(scores_rec_diffu, test_batch[1], [5, 10, 20])
            for k, v in metrics.items():
                test_metrics_dict[k].append(v)
    for key_temp, values_temp in test_metrics_dict.items():
        values_mean = round(np.mean(values_temp) * 100, 4)
        test_metrics_dict_mean[key_temp] = values_mean
    print(test_metrics_dict_mean)


def model_train(tra_data_loader, val_data_loader, test_data_loader, model_joint, args, logger):
    epochs = args.epochs
    device = args.device
    metric_ks = args.metric_ks
    model_joint = model_joint.to(device)
    is_parallel = args.num_gpu > 1
    if is_parallel:
        model_joint = nn.DataParallel(model_joint)
    optimizer = optimizers(model_joint, args)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.gamma)

    # DDBC eval params
    predict_nums  = [int(x) for x in args.predict_nums.split(',')]
    multipliers   = [int(x) for x in args.candidate_multipliers.split(',')]
    eval_seed     = args.random_seed
    ddbc_data_dir = args.ddbc_data_dir

    # TensorBoard
    writer = None
    if hasattr(args, 'tb_log_dir') and args.tb_log_dir:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=args.tb_log_dir)
        print(f'TensorBoard log dir: {args.tb_log_dir}')

    # Checkpoint dir
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    best_val_recall = 0.0
    best_epoch = 0
    best_model = copy.deepcopy(model_joint)
    bad_count = 0

    for epoch_temp in range(epochs):
        print('Epoch: {}'.format(epoch_temp))
        logger.info('Epoch: {}'.format(epoch_temp))
        model_joint.train()

        for index_temp, train_batch in enumerate(tra_data_loader):
            train_batch = [x.to(device) for x in train_batch]
            optimizer.zero_grad()
            scores, diffu_rep, weights, t, item_rep_dis, seq_rep_dis = model_joint(train_batch[0], train_batch[1], train_flag=True)
            loss_all = model_joint.loss_diffu_ce(diffu_rep, train_batch[1])
            loss_all.backward()
            optimizer.step()
            if index_temp % int(len(tra_data_loader) / 5 + 1) == 0:
                print('[%d/%d] Loss: %.4f' % (index_temp, len(tra_data_loader), loss_all.item()))
                logger.info('[%d/%d] Loss: %.4f' % (index_temp, len(tra_data_loader), loss_all.item()))
        print("loss in epoch {}: {}".format(epoch_temp, loss_all.item()))
        if writer:
            writer.add_scalar('train/loss', loss_all.item(), epoch_temp)
        lr_scheduler.step()

        if epoch_temp != 0 and epoch_temp % args.eval_interval == 0:
            print('start predicting: ', datetime.datetime.now())
            logger.info('start predicting: {}'.format(datetime.datetime.now()))

            val_recall = evaluate_ddbc(
                model_joint, args,
                predict_nums, multipliers, eval_seed,
                writer=writer, epoch=epoch_temp, split='valid',
                topk=args.topk, ddbc_data_dir=ddbc_data_dir
            )

            if val_recall > best_val_recall:
                best_val_recall = val_recall
                best_epoch = epoch_temp
                bad_count = 0
                best_model = copy.deepcopy(model_joint)
                ckpt_path = os.path.join(save_dir, 'best_model.pt')
                torch.save({'epoch': epoch_temp, 'model_state_dict': model_joint.state_dict(),
                            'val_recall': val_recall}, ckpt_path)
                print(f'  [Checkpoint] saved (epoch={epoch_temp}, val_recall@{predict_nums[0]}_x{multipliers[0]}={val_recall:.4f})')
                logger.info(f'Checkpoint saved: epoch={epoch_temp}, val_recall={val_recall:.4f}')
            else:
                bad_count += 1
                if bad_count >= args.patience:
                    print(f'Early stopping at epoch {epoch_temp} (patience={args.patience})')
                    logger.info(f'Early stopping at epoch {epoch_temp}')
                    break

    # ── Final test evaluation with best model ──
    print('\n========== Final Test Evaluation ==========')
    logger.info('Final Test Evaluation')
    test_recall = evaluate_ddbc(
        best_model, args,
        predict_nums, multipliers, eval_seed,
        writer=writer, epoch=best_epoch, split='test',
        topk=args.topk, ddbc_data_dir=ddbc_data_dir
    )
    print(f'Best checkpoint: epoch={best_epoch}, val_recall={best_val_recall:.4f}')
    logger.info(f'Best epoch={best_epoch}, val_recall={best_val_recall:.4f}')

    if writer:
        writer.close()

    return best_model, {'val_recall': best_val_recall, 'test_recall': test_recall}
