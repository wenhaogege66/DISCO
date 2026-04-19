"""
Convert DISCO txt files to DiffuRec pickle format for Yelp dataset.

Output: DiffuRec/data/yelp/dataset.pkl
  train : {bundle_id: [item1..item10]}  — 0-based IDs, from train.txt
  val   : {bundle_id: [item10]}          — last item of each valid bundle (for original DiffuRec eval)
  test  : {bundle_id: [item10]}          — last item of each test bundle
  smap  : {i: i for i in range(20033)}   — identity mapping, item_num=20033

Item IDs: 0-based (0 to 20032), PAD=20033 (=item_num)

DDBC multi-label eval data is reused from DreamRec/data/yelp/:
  valid_data_items{3,5}.df, test_data_items{3,5}.df
"""

import pickle
import os

DISCO_DIR  = "/home/sjj/wenhao/DISCO/datasets/Yelp"
OUTPUT_DIR = "/home/sjj/wenhao/DiffuRec/data/yelp"
ITEM_NUM   = 20033
SEQ_SIZE   = 10


def read_txt(path):
    bundles = []
    with open(path) as f:
        for line in f:
            parts = [int(x.strip()) for x in line.strip().split(',')]
            bundle_id = parts[0]
            items = parts[1:]
            assert len(items) == SEQ_SIZE, f"Expected {SEQ_SIZE} items, got {len(items)}"
            bundles.append((bundle_id, items))
    return bundles


def main():
    print("=" * 60)
    print("Convert DISCO txt → DiffuRec pickle")
    print(f"  ITEM_NUM={ITEM_NUM}, SEQ_SIZE={SEQ_SIZE}, PAD={ITEM_NUM}")
    print("=" * 60)

    train_bundles = read_txt(os.path.join(DISCO_DIR, "train.txt"))
    valid_bundles = read_txt(os.path.join(DISCO_DIR, "valid.txt"))
    test_bundles  = read_txt(os.path.join(DISCO_DIR, "test.txt"))
    print(f"train={len(train_bundles)}, valid={len(valid_bundles)}, test={len(test_bundles)}")

    # Training: full 10-item sequences; Data_Train.split_onebyone() creates growing sequences
    train_dict = {i: items for i, (_, items) in enumerate(train_bundles)}

    # Val/test: single last item per bundle (used only for original DiffuRec HR@k eval)
    val_dict  = {i: [items[-1]] for i, (_, items) in enumerate(valid_bundles)}
    test_dict = {i: [items[-1]] for i, (_, items) in enumerate(test_bundles)}

    # smap: identity mapping (item_num = len(smap) = 20033)
    smap = {i: i for i in range(ITEM_NUM)}

    data = {'train': train_dict, 'val': val_dict, 'test': test_dict, 'smap': smap}

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "dataset.pkl")
    with open(out_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"\nSaved: {out_path}")
    print(f"  train bundles : {len(train_dict)}  ({len(train_dict)*9} growing-sequence samples)")
    print(f"  valid bundles : {len(val_dict)}")
    print(f"  test  bundles : {len(test_dict)}")
    print(f"\nDDBC multi-label eval data reused from DreamRec/data/yelp/")
    print("=" * 60)


if __name__ == "__main__":
    main()
