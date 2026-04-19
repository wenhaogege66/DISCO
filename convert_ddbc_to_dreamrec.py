"""
从 DDBC 的 txt 文件生成 DreamRec 所需的数据文件，实现两个模型的完全数据对齐。

DDBC txt 格式: bundle_id, item1, ..., item10  (item ID 均为 0-based，与 DreamRec 一致)

输出文件:
  train_data.df         — 从 train.txt(69774 bundles) 生成增长序列样本，用于 DreamRec 训练
  valid_data_items3.df  — 从 valid.txt(9967 bundles)  生成，input=前7item, labels=后3item
  valid_data_items5.df  — 从 valid.txt(9967 bundles)  生成，input=前5item, labels=后5item
  test_data_items3.df   — 从 test.txt (19937 bundles) 生成，input=前7item, labels=后3item
  test_data_items5.df   — 从 test.txt (19937 bundles) 生成，input=前5item, labels=后5item

注意：data_statis.df 无需改动 (item_num=20033, seq_size=10 不变)
"""

import numpy as np
import pandas as pd
import os

DDBC_DIR     = "/home/sjj/wenhao/DDBC_f-main/datasets/Yelp"
DREAMREC_DIR = "/home/sjj/wenhao/DreamRec/data/yelp"
PREDICT_NUMS = [3, 5]

# 从 data_statis.df 读取参数，保持与现有 DreamRec 配置一致
statis    = pd.read_pickle(os.path.join(DREAMREC_DIR, "data_statis.df"))
ITEM_NUM  = int(statis['item_num'][0])   # 20033
SEQ_SIZE  = int(statis['seq_size'][0])   # 10
PAD_TOKEN = ITEM_NUM                     # DreamRec 用 item_num 作为 PAD


def read_txt(path):
    """读取 DDBC txt 文件，返回 list of (bundle_id, [item1..item10])。"""
    bundles = []
    with open(path) as f:
        for line in f:
            parts = [int(x.strip()) for x in line.strip().split(',')]
            bundle_id = parts[0]
            items     = parts[1:]   # 10 个 item，0-based
            assert len(items) == SEQ_SIZE, f"期望10个item，实际得到 {len(items)}: {items}"
            bundles.append((bundle_id, items))
    return bundles


def generate_train(bundles):
    """
    从 train.txt 的 bundles 生成增长序列训练样本，与现有 DreamRec 训练格式完全一致。

    对每个 10-item bundle [a,b,c,d,e,f,g,h,i,j] 生成 9 个样本：
      seq=[a, PAD*9], len_seq=1, next=b
      seq=[a,b, PAD*8], len_seq=2, next=c
      ...
      seq=[a,b,c,d,e,f,g,h,i, PAD], len_seq=9, next=j
    """
    examples = []
    for _, items in bundles:
        for i in range(1, len(items)):               # i = 目标 item 的位置
            history_len = min(i, SEQ_SIZE)
            if i <= SEQ_SIZE:
                seq = items[:i] + [PAD_TOKEN] * (SEQ_SIZE - i)
            else:
                seq = items[i - SEQ_SIZE:i]
            examples.append({
                'seq':     seq,
                'len_seq': history_len,
                'next':    items[i],
            })
    return pd.DataFrame(examples)


def generate_eval(bundles, predict_n):
    """
    从 valid/test bundles 生成多 label 评估格式，与 DDBC 评估逻辑对齐。

    predict_n=3: input=前7item(+3×PAD), len_seq=7, labels=[item8,item9,item10]
    predict_n=5: input=前5item(+5×PAD), len_seq=5, labels=[item6,item7,item8,item9,item10]

    labels 保留原始顺序（包含可能的重复 item），指标计算时使用 Counter 模式。
    """
    history_n = SEQ_SIZE - predict_n   # 7 or 5
    examples  = []
    for _, items in bundles:
        history = items[:history_n]
        labels  = items[history_n:]     # predict_n 个 label item，顺序与 DDBC 一致
        seq     = history + [PAD_TOKEN] * predict_n
        examples.append({
            'seq':     seq,
            'len_seq': history_n,
            'labels':  labels,
        })
    return pd.DataFrame(examples)


def main():
    print("=" * 60)
    print("Convert DDBC txt → DreamRec df")
    print(f"  ITEM_NUM={ITEM_NUM}, SEQ_SIZE={SEQ_SIZE}, PAD_TOKEN={PAD_TOKEN}")
    print("=" * 60)

    # ---------- 读取 txt ----------
    print("\n[1/3] Reading DDBC txt files...")
    train_bundles = read_txt(os.path.join(DDBC_DIR, "train.txt"))
    valid_bundles = read_txt(os.path.join(DDBC_DIR, "valid.txt"))
    test_bundles  = read_txt(os.path.join(DDBC_DIR, "test.txt"))
    print(f"  train={len(train_bundles)}, valid={len(valid_bundles)}, test={len(test_bundles)}")

    # ---------- 生成训练集（增长序列格式）----------
    print("\n[2/3] Generating train_data.df (growing-sequence format)...")
    train_df = generate_train(train_bundles)
    train_df.to_pickle(os.path.join(DREAMREC_DIR, "train_data.df"))
    print(f"  Saved train_data.df: {len(train_df)} samples "
          f"({len(train_bundles)} bundles × 9 samples each)")
    print(f"  len_seq distribution: {dict(train_df['len_seq'].value_counts().sort_index())}")

    # ---------- 生成验证集和测试集（多 label 格式）----------
    print("\n[3/3] Generating valid/test multi-label eval files...")
    for predict_n in PREDICT_NUMS:
        history_n = SEQ_SIZE - predict_n

        valid_df = generate_eval(valid_bundles, predict_n)
        test_df  = generate_eval(test_bundles,  predict_n)

        valid_path = os.path.join(DREAMREC_DIR, f"valid_data_items{predict_n}.df")
        test_path  = os.path.join(DREAMREC_DIR, f"test_data_items{predict_n}.df")

        valid_df.to_pickle(valid_path)
        test_df.to_pickle(test_path)

        print(f"\n  predict_n={predict_n}  (input={history_n} items → labels={predict_n} items)")
        print(f"    valid_data_items{predict_n}.df: {len(valid_df)} samples  → {valid_path}")
        print(f"    test_data_items{predict_n}.df:  {len(test_df)} samples   → {test_path}")

        # 抽查第一个样本
        row = test_df.iloc[0]
        print(f"    test[0]: seq={row['seq']}, len_seq={row['len_seq']}, labels={row['labels']}")

    print("\n" + "=" * 60)
    print("Done. data_statis.df unchanged (item_num=20033, seq_size=10).")
    print("=" * 60)


if __name__ == "__main__":
    main()
