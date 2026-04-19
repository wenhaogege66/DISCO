"""
Convert DISCO txt files to BERT4Rec dataset.pkl format for Yelp.

BERT4Rec uses 1-indexed item IDs (1..item_num) with PAD=0 and MASK=item_num+1.
DISCO uses 0-based IDs (0..20032). This script shifts IDs by +1.

Output: BERT4Rec/Data/preprocessed/yelp_min_rating0-min_uc0-min_sc0-splitleave_one_out/dataset.pkl
  train : {bundle_id: [item1..item10]}   — 1-based IDs, full 10-item sequence
  val   : {bundle_id: [item10]}           — last item of each valid bundle
  test  : {bundle_id: [item10]}           — last item of each test bundle
  umap  : {bundle_id: bundle_id}          — identity
  smap  : {orig_0based: 1based}           — shift map (len=20033)

The BERT4Rec BertTrainDataset.split_onebyone equivalent is the masked-LM training
on the full sequence — no further splitting needed here.

DDBC multi-label eval data is reused from DreamRec/data/yelp/ (same source).
"""

import pickle
import os
from pathlib import Path

DISCO_DIR  = "/home/sjj/wenhao/DISCO/datasets/Yelp"
OUTPUT_DIR = "/home/sjj/wenhao/BERT4Rec/Data/preprocessed/yelp_min_rating0-min_uc0-min_sc0-splitleave_one_out"
ITEM_NUM   = 20033   # 0-based IDs: 0..20032  →  1-based: 1..20033
SEQ_SIZE   = 10


def read_txt(path):
    bundles = []
    with open(path) as f:
        for line in f:
            parts = [int(x.strip()) for x in line.strip().split(',')]
            bundle_id = parts[0]
            items = parts[1:]
            assert len(items) == SEQ_SIZE
            bundles.append((bundle_id, items))
    return bundles


def shift(items):
    """0-based → 1-based item IDs."""
    return [x + 1 for x in items]


def main():
    print("=" * 60)
    print("Convert DISCO txt → BERT4Rec dataset.pkl")
    print(f"  ITEM_NUM={ITEM_NUM}, SEQ_SIZE={SEQ_SIZE}")
    print(f"  Item IDs: 0-based → 1-based (shift +1), PAD=0, MASK={ITEM_NUM+1}")
    print("=" * 60)

    train_bundles = read_txt(os.path.join(DISCO_DIR, "train.txt"))
    valid_bundles = read_txt(os.path.join(DISCO_DIR, "valid.txt"))
    test_bundles  = read_txt(os.path.join(DISCO_DIR, "test.txt"))
    print(f"train={len(train_bundles)}, valid={len(valid_bundles)}, test={len(test_bundles)}")

    # Training: full 10-item sequences (1-based); BertTrainDataset applies random masking
    train_dict = {i: shift(items) for i, (_, items) in enumerate(train_bundles)}

    # Val/test: last item of each bundle (1-based); used for original BERT4Rec eval
    val_dict  = {i: shift([items[-1]]) for i, (_, items) in enumerate(valid_bundles)}
    test_dict = {i: shift([items[-1]]) for i, (_, items) in enumerate(test_bundles)}

    # umap: identity (bundle_id → bundle_id)
    n_train = len(train_bundles)
    umap = {i: i for i in range(n_train)}

    # smap: 0-based → 1-based (len=20033, values 1..20033)
    smap = {i: i + 1 for i in range(ITEM_NUM)}

    dataset = {'train': train_dict, 'val': val_dict, 'test': test_dict,
               'umap': umap, 'smap': smap}

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "dataset.pkl")
    with open(out_path, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"\nSaved: {out_path}")
    print(f"  train bundles : {len(train_dict)}")
    print(f"  valid bundles : {len(val_dict)}")
    print(f"  test  bundles : {len(test_dict)}")
    print(f"  item_num (1-based) : {ITEM_NUM}  (MASK token = {ITEM_NUM+1})")
    print(f"\nDDBC multi-label eval data reused from DreamRec/data/yelp/")
    print("=" * 60)


if __name__ == "__main__":
    main()
