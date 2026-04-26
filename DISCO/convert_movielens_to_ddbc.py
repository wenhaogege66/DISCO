"""
MovieLens-20M → DISCO/LETTER conversion script (semantic IDs unified with TIGER).

Reads from:  TIGER/cache/MovieLens-20M/len{N}/processed/
Writes to:   DISCO/datasets/MovieLens-20M/len{N}/
          +  LETTER/data/MovieLens-20M/len{N}/

Usage:
    python convert_movielens_to_ddbc.py --seq_len 60
    python convert_movielens_to_ddbc.py --seq_len 90
    python convert_movielens_to_ddbc.py --seq_len 120
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch

TIGER_CACHE_ROOT = "/home/sjj/wenhao/TIGER/cache"
DISCO_DATASETS_ROOT = "/home/sjj/wenhao/DISCO/datasets"
LETTER_DATA_ROOT = "/home/sjj/wenhao/LETTER/data/MovieLens-20M"
LETTER_PRIMARY_SEQ_LEN = int(os.environ.get("LETTER_PRIMARY_SEQ_LEN", "60"))
EMBEDDING_FILE = "sentence-t5-base.sent_emb"
SEM_ID_FILE = "sentence-t5-base_256,256,256,256.sem_ids"
EMBEDDING_DIM = 64
TRAIN_RATIO = 0.7
VALID_RATIO = 0.1
RQ_N_CODEBOOKS = 3
RQ_CODEBOOK_SIZE = 256


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seq_len", type=int, default=60, choices=[60, 90, 120])
    return p.parse_args()


def build_disco_sid_from_tiger(
    tiger_item2sid: Dict[str, List[int]],
    id_mapping: Dict,
    n_items: int,
) -> Dict[int, Tuple[int, int, int, int]]:
    sid = {}
    for iid_1based in range(1, n_items + 1):
        orig_item = id_mapping["id2item"][iid_1based]
        if orig_item not in tiger_item2sid:
            raise KeyError(f"Missing TIGER semantic id for item: {orig_item}")
        vals = tiger_item2sid[orig_item]
        if len(vals) < 4:
            raise ValueError(f"Invalid semantic id for {orig_item}: {vals}")
        sid[iid_1based - 1] = (int(vals[0]), int(vals[1]), int(vals[2]), int(vals[3]))
    return sid


def build_disco_token_from_sid(sid: Dict[int, Tuple[int, int, int, int]]) -> Dict[str, List[int]]:
    token = {}
    for item_id, (a, b, c, d) in sid.items():
        token[str(item_id)] = [
            a + 1,
            b + RQ_CODEBOOK_SIZE + 1,
            c + RQ_CODEBOOK_SIZE * 2 + 1,
            d + RQ_CODEBOOK_SIZE * 3,
        ]
    return token


def build_disco_weight_from_sid(
    embs: np.ndarray,
    sid: Dict[int, Tuple[int, int, int, int]],
) -> np.ndarray:
    global_mean = embs.mean(axis=0)
    weight = np.zeros((RQ_N_CODEBOOKS * RQ_CODEBOOK_SIZE, embs.shape[1]), dtype=np.float32)

    for codebook_idx in range(RQ_N_CODEBOOKS):
        assignments = np.array([sid[i][codebook_idx] for i in range(len(sid))], dtype=np.int32)
        for code in range(RQ_CODEBOOK_SIZE):
            mask = assignments == code
            row = codebook_idx * RQ_CODEBOOK_SIZE + code
            if np.any(mask):
                weight[row] = embs[mask].mean(axis=0)
            else:
                weight[row] = global_mean
    return weight


def build_letter_index_from_sid(sid: Dict[int, Tuple[int, int, int, int]]) -> Dict[str, List[str]]:
    index = {}
    for item_id, (a, b, c, d) in sid.items():
        index[str(item_id)] = [f"<a_{a}>", f"<b_{b}>", f"<c_{c}>", f"<d_{d}>"]
    return index


def build_letter_inter(user_seqs: Dict[str, List[str]], id_mapping: Dict, seq_len: int) -> Dict[str, List[int]]:
    inter = {}
    uid = 0
    for _, item_seq_orig in user_seqs.items():
        item_seq = []
        for orig_id in item_seq_orig:
            iid = id_mapping["item2id"].get(orig_id, 0)
            if iid > 0:
                item_seq.append(iid - 1)
        if len(item_seq) == seq_len:
            inter[str(uid)] = item_seq
            uid += 1
    return inter


def split_title_description(sentence: str) -> Tuple[str, str]:
    s = sentence.strip()
    if s.startswith("Title: "):
        rest = s[len("Title: "):]
        if ". Genres:" in rest:
            title, _ = rest.split(". Genres:", 1)
            return title.strip(), s
    return s, s


def build_letter_item(id_mapping: Dict, metadata_orig: Dict[str, str], n_items: int) -> Dict[str, Dict[str, str]]:
    item = {}
    for iid_1based in range(1, n_items + 1):
        orig_item = id_mapping["id2item"][iid_1based]
        sent = metadata_orig.get(orig_item, "")
        title, desc = split_title_description(sent)
        item[str(iid_1based - 1)] = {"title": title, "description": desc}
    return item


def save_json(path: str, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    args = parse_args()
    seq_len = args.seq_len

    processed_dir = os.path.join(TIGER_CACHE_ROOT, "MovieLens-20M", f"len{seq_len}", "processed")
    output_dir = os.path.join(DISCO_DATASETS_ROOT, "MovieLens-20M", f"len{seq_len}")
    letter_len_dir = os.path.join(LETTER_DATA_ROOT, f"len{seq_len}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(letter_len_dir, exist_ok=True)
    os.makedirs(LETTER_DATA_ROOT, exist_ok=True)

    print(f"[MovieLens→DISCO/LETTER] seq_len={seq_len}")
    print(f"  src: {processed_dir}")
    print(f"  DISCO dst: {output_dir}")
    print(f"  LETTER dst: {letter_len_dir}")

    print("[1/8] Loading id_mapping ...")
    with open(os.path.join(processed_dir, "id_mapping.json")) as f:
        id_mapping = json.load(f)
    n_items = len(id_mapping["item2id"]) - 1
    print(f"  items={n_items}")

    print("[2/8] Loading all_item_seqs ...")
    with open(os.path.join(processed_dir, "all_item_seqs.json")) as f:
        user_seqs = json.load(f)
    print(f"  users={len(user_seqs)}")

    print("[3/8] Loading embeddings ...")
    emb_path = os.path.join(processed_dir, EMBEDDING_FILE)
    embs = np.fromfile(emb_path, dtype=np.float32).reshape(n_items, EMBEDDING_DIM)
    torch.save(torch.FloatTensor(embs), os.path.join(output_dir, "clhe.pt"))
    print(f"  clhe.pt shape: {embs.shape}")

    print("[4/8] Loading TIGER semantic IDs and exporting DISCO semantic files ...")
    with open(os.path.join(processed_dir, SEM_ID_FILE)) as f:
        tiger_item2sid = json.load(f)

    disco_sid = build_disco_sid_from_tiger(tiger_item2sid, id_mapping, n_items)
    disco_token = build_disco_token_from_sid(disco_sid)
    disco_weight = build_disco_weight_from_sid(embs, disco_sid)

    np.save(os.path.join(output_dir, "clhe_sid.npy"), np.array(disco_sid, dtype=object))
    save_json(os.path.join(output_dir, "clhe_token.json"), disco_token)
    np.save(os.path.join(output_dir, "clhe_weight.npy"), disco_weight)
    print("  exported: clhe_sid.npy, clhe_token.json, clhe_weight.npy (from TIGER sem_ids)")

    print("[5/8] Converting metadata ...")
    with open(os.path.join(processed_dir, "metadata.sentence.json")) as f:
        meta_orig = json.load(f)
    metadata = {}
    for item_id in range(1, len(id_mapping["item2id"])):
        orig_id = id_mapping["id2item"][item_id]
        if orig_id in meta_orig and orig_id != "[PAD]":
            metadata[str(item_id - 1)] = meta_orig[orig_id]
    save_json(os.path.join(output_dir, "metadata.json"), metadata)
    print(f"  metadata.json: {len(metadata)} items")

    print("[6/8] Building bundles ...")
    bundles = []
    for _, item_seq_orig in user_seqs.items():
        item_seq = []
        for orig_id in item_seq_orig:
            iid = id_mapping["item2id"].get(orig_id, 0)
            if iid > 0:
                item_seq.append(iid - 1)
        if len(item_seq) == seq_len:
            bundles.append(item_seq)
    print(f"  {len(bundles)} bundles, all length {seq_len}")

    print("[7/8] Splitting and saving DISCO txt files ...")
    np.random.seed(42)
    idx = np.random.permutation(len(bundles))
    n_train = int(len(bundles) * TRAIN_RATIO)
    n_valid = int(len(bundles) * VALID_RATIO)
    splits = {
        "train": idx[:n_train],
        "valid": idx[n_train:n_train + n_valid],
        "test": idx[n_train + n_valid:],
    }

    for split_name, split_idx in splits.items():
        path = os.path.join(output_dir, f"{split_name}.txt")
        with open(path, "w") as f:
            for bid, i in enumerate(split_idx):
                line = f"{bid}, " + ", ".join(map(str, bundles[i])) + "\n"
                f.write(line)
        print(f"  {split_name}.txt: {len(split_idx)} bundles")

    with open(os.path.join(output_dir, "bi_full.txt"), "w") as f:
        for bid, bundle in enumerate(bundles):
            f.write(f"{bid}, " + ", ".join(map(str, bundle)) + "\n")

    count = {
        "#U": len(user_seqs),
        "#I": n_items,
        "#B": len(bundles),
        "#B-I": len(bundles) * seq_len,
        "#U-I": len(bundles) * seq_len,
        "#Max. I/B": seq_len,
        "Avg.I/B": float(seq_len),
    }
    save_json(os.path.join(output_dir, "count.json"), count)

    print("[8/8] Exporting LETTER files (aligned with TIGER semantic IDs) ...")
    letter_index = build_letter_index_from_sid(disco_sid)
    letter_inter = build_letter_inter(user_seqs, id_mapping, seq_len)
    letter_item = build_letter_item(id_mapping, meta_orig, n_items)

    save_json(os.path.join(letter_len_dir, "MovieLens-20M.index.json"), letter_index)
    save_json(os.path.join(letter_len_dir, "MovieLens-20M.inter.json"), letter_inter)
    save_json(os.path.join(letter_len_dir, "MovieLens-20M.item.json"), letter_item)
    print(f"  len{seq_len}: index={len(letter_index)} inter={len(letter_inter)} item={len(letter_item)}")

    if seq_len == LETTER_PRIMARY_SEQ_LEN:
        save_json(os.path.join(LETTER_DATA_ROOT, "MovieLens-20M.index.json"), letter_index)
        save_json(os.path.join(LETTER_DATA_ROOT, "MovieLens-20M.inter.json"), letter_inter)
        save_json(os.path.join(LETTER_DATA_ROOT, "MovieLens-20M.item.json"), letter_item)
        print(f"  Updated LETTER primary dataset at {LETTER_DATA_ROOT} using len{seq_len}")

    print(f"\nDone. DISCO output: {output_dir}")
    for k, v in count.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
