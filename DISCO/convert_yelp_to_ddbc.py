"""
Yelp数据转换脚本
将processed格式的Yelp数据转换为DISCO框架所需的格式

输入文件（位于 datasets/Yelp/processed/）：
  - id_mapping.json: ID映射
  - all_item_seqs.json: 用户交互序列
  - metadata.sentence.json: 商家元数据
  - sentence-t5-base.sent_emb: 句子嵌入
  - yelp_academic_dataset_business.json: 原始商家数据

输出文件（位于 datasets/Yelp/）：
  - clhe.pt: item embeddings (torch tensor)
  - train.txt: 训练集bundle数据
  - valid.txt: 验证集bundle数据
  - test.txt: 测试集bundle数据
  - count.json: 统计信息
  - metadata.json: item元数据
"""

import json
import numpy as np
import torch
import os
from pathlib import Path
from collections import Counter

# ================= 配置参数 =================
PROCESSED_DIR = "/home/sjj/wenhao/DISCO/datasets/Yelp/processed"
OUTPUT_DIR = "/home/sjj/wenhao/DISCO/datasets/Yelp"
EMBEDDING_FILE = "sentence-t5-base.sent_emb"
EMBEDDING_DIM = 64

# Bundle定义策略：
# "full_sequence": 每个用户的完整序列作为一个bundle
# "sliding_window": 使用滑动窗口切分序列（窗口大小见下方参数）
# "adaptive": 长度<WINDOW_SIZE的保留，长度>=WINDOW_SIZE的使用滑动窗口
BUNDLE_STRATEGY = "adaptive"
WINDOW_SIZE = 10  # 滑动窗口大小（仅当BUNDLE_STRATEGY="sliding_window"或"adaptive"时使用）
WINDOW_STRIDE = 1  # 滑动步长

# 数据划分比例（与Spotify保持一致：70/10/20）
TRAIN_RATIO = 0.7
VALID_RATIO = 0.1
TEST_RATIO = 0.2

# 最小bundle大小（过滤掉太短的序列）
MIN_BUNDLE_SIZE = 10


def load_id_mapping():
    """加载ID映射"""
    print("[1/7] Loading ID mapping...")
    with open(f"{PROCESSED_DIR}/id_mapping.json") as f:
        id_mapping = json.load(f)

    print(f"  Total users: {len(id_mapping['user2id'])}")
    print(f"  Total items: {len(id_mapping['item2id'])}")
    return id_mapping


def load_user_sequences():
    """加载用户交互序列"""
    print("[2/7] Loading user sequences...")
    with open(f"{PROCESSED_DIR}/all_item_seqs.json") as f:
        seqs = json.load(f)

    print(f"  Total users: {len(seqs)}")
    return seqs


def load_embeddings(id_mapping):
    """加载并转换embeddings到torch格式"""
    print("[3/7] Loading embeddings...")

    # 读取原始嵌入（float32二进制文件）
    emb_path = f"{PROCESSED_DIR}/{EMBEDDING_FILE}"
    embeddings = np.fromfile(emb_path, dtype=np.float32)

    n_items = len(id_mapping['item2id']) - 1  # 减去PAD
    embeddings = embeddings.reshape(n_items, EMBEDDING_DIM)

    print(f"  Loaded embeddings shape: {embeddings.shape}")

    # 转换为torch tensor并保存
    emb_tensor = torch.from_numpy(embeddings)
    output_path = f"{OUTPUT_DIR}/clhe.pt"
    torch.save(emb_tensor, output_path)
    print(f"  Saved to: {output_path}")

    return embeddings


def load_metadata():
    """加载并转换metadata"""
    print("[4/7] Loading metadata...")
    with open(f"{PROCESSED_DIR}/metadata.sentence.json") as f:
        metadata_orig = json.load(f)

    # 加载ID映射
    with open(f"{PROCESSED_DIR}/id_mapping.json") as f:
        id_mapping = json.load(f)

    # 转换为item_id → 描述的映射（0-based索引）
    # 关键：必须保证metadata["0"]对应embedding[0]，即原始item_id=1
    metadata = {}

    # 按照item_id顺序构建（从1开始，跳过PAD）
    for item_id in range(1, len(id_mapping['item2id'])):
        # 获取原始business_id（id2item是list，用整数索引）
        orig_id = id_mapping['id2item'][item_id]

        if orig_id in metadata_orig and orig_id != '[PAD]':
            # 使用0-based索引：item_id=1 → metadata["0"]
            metadata[str(item_id - 1)] = metadata_orig[orig_id]

    # 按key排序保存（虽然JSON的key是无序的，但这样更易读）
    metadata_sorted = {k: metadata[k] for k in sorted(metadata.keys(), key=int)}

    output_path = f"{OUTPUT_DIR}/metadata.json"
    with open(output_path, 'w') as f:
        json.dump(metadata_sorted, f, indent=2)
    print(f"  Saved {len(metadata_sorted)} items to: {output_path}")
    print(f"  Metadata ID range: 0 to {len(metadata_sorted) - 1}")

    return metadata_sorted


def create_bundles(user_sequences, id_mapping, strategy="full_sequence"):
    """
    从用户序列创建bundles

    Args:
        user_sequences: 用户交互序列字典
        id_mapping: ID映射
        strategy: bundle创建策略

    Returns:
        bundles: list of (bundle_id, item_list)
    """
    print(f"[5/7] Creating bundles (strategy: {strategy})...")

    bundles = []
    bundle_id = 0

    for user_orig_id, item_seq_orig in user_sequences.items():
        # 转换item ID（从原始ID到内部ID，并转为0-based）
        item_seq = []
        for item_orig_id in item_seq_orig:
            if item_orig_id in id_mapping['item2id']:
                item_id = id_mapping['item2id'][item_orig_id]
                if item_id > 0:  # 跳过PAD
                    item_seq.append(item_id - 1)  # 转换为0-based索引

        # 过滤太短的序列
        if len(item_seq) < MIN_BUNDLE_SIZE:
            continue

        if strategy == "full_sequence":
            # 整个序列作为一个bundle
            bundles.append((bundle_id, item_seq))
            bundle_id += 1

        elif strategy == "sliding_window":
            # 使用滑动窗口切分所有序列
            for start in range(0, len(item_seq) - WINDOW_SIZE + 1, WINDOW_STRIDE):
                window = item_seq[start:start + WINDOW_SIZE]
                if len(window) >= MIN_BUNDLE_SIZE:
                    bundles.append((bundle_id, window))
                    bundle_id += 1

        elif strategy == "adaptive":
            # 自适应策略：长度<WINDOW_SIZE的保留，长度>=WINDOW_SIZE的使用滑动窗口
            if len(item_seq) < WINDOW_SIZE:
                # 短序列保留原样
                bundles.append((bundle_id, item_seq))
                bundle_id += 1
            else:
                # 长序列使用滑动窗口
                for start in range(0, len(item_seq) - WINDOW_SIZE + 1, WINDOW_STRIDE):
                    window = item_seq[start:start + WINDOW_SIZE]
                    bundles.append((bundle_id, window))
                    bundle_id += 1

    print(f"  Created {len(bundles)} bundles")

    # 统计bundle大小
    bundle_sizes = [len(items) for _, items in bundles]
    print(f"  Bundle size statistics:")
    print(f"    Min: {min(bundle_sizes)}")
    print(f"    Max: {max(bundle_sizes)}")
    print(f"    Mean: {np.mean(bundle_sizes):.2f}")
    print(f"    Median: {np.median(bundle_sizes):.0f}")

    return bundles


def split_and_save_bundles(bundles, id_mapping):
    """
    划分train/valid/test并保存
    """
    print("[6/7] Splitting and saving bundles...")

    # 随机打乱
    np.random.seed(42)
    indices = np.random.permutation(len(bundles))

    # 计算划分点
    n_train = int(len(bundles) * TRAIN_RATIO)
    n_valid = int(len(bundles) * VALID_RATIO)

    train_indices = indices[:n_train]
    valid_indices = indices[n_train:n_train + n_valid]
    test_indices = indices[n_train + n_valid:]

    # 保存各个split
    splits = {
        'train': train_indices,
        'valid': valid_indices,
        'test': test_indices
    }

    stats = {}
    for split_name, split_indices in splits.items():
        output_path = f"{OUTPUT_DIR}/{split_name}.txt"

        # 收集该split的bundles
        split_bundles = [(bundles[idx][0], bundles[idx][1]) for idx in split_indices]

        # 按bundle_id排序（保持与Spotify一致的格式）
        split_bundles.sort(key=lambda x: x[0])

        # 写入文件
        with open(output_path, 'w') as f:
            for bundle_id, items in split_bundles:
                # 格式：bundle_id, item1, item2, ...
                line = f"{bundle_id}, " + ", ".join(map(str, items)) + "\n"
                f.write(line)

        stats[split_name] = len(split_indices)
        print(f"  Saved {split_name}: {len(split_indices)} bundles (sorted by ID) → {output_path}")

    # 生成 bi_full.txt（所有bundles的合集，按bundle_id排序）
    bi_full_path = f"{OUTPUT_DIR}/bi_full.txt"
    all_bundles = sorted(bundles, key=lambda x: x[0])
    with open(bi_full_path, 'w') as f:
        for bundle_id, items in all_bundles:
            line = f"{bundle_id}, " + ", ".join(map(str, items)) + "\n"
            f.write(line)
    print(f"  Saved bi_full: {len(bundles)} bundles → {bi_full_path}")

    return stats, bundles


def compute_and_save_statistics(bundles, id_mapping, split_stats):
    """
    计算并保存count.json
    """
    print("[7/7] Computing statistics...")

    # 统计
    n_users = len(id_mapping['user2id']) - 1  # 减去PAD
    n_items = len(id_mapping['item2id']) - 1  # 减去PAD
    n_bundles = len(bundles)

    # Bundle-Item交互数（所有bundle中item出现的总次数）
    n_bi_interactions = sum(len(items) for _, items in bundles)

    # 最大bundle大小
    max_bundle_size = max(len(items) for _, items in bundles)

    # 平均bundle大小
    avg_bundle_size = n_bi_interactions / n_bundles

    # User-Item交互数（假设每个bundle对应一个user）
    # 这里简化处理，实际应该从原始数据统计
    n_ui_interactions = n_bi_interactions  # 简化：假设bundle就是user的交互

    count_json = {
        "#U": n_users,
        "#I": n_items,
        "#B": n_bundles,
        "#B-I": n_bi_interactions,
        "#U-I": n_ui_interactions,
        "#Max. I/B": max_bundle_size,
        "Avg.I/B": round(avg_bundle_size, 4)
    }

    output_path = f"{OUTPUT_DIR}/count.json"
    with open(output_path, 'w') as f:
        json.dump(count_json, f)

    print(f"  Saved statistics to: {output_path}")
    print(f"\n  Statistics:")
    for key, value in count_json.items():
        print(f"    {key}: {value}")

    return count_json


def main():
    """主函数"""
    print("=" * 60)
    print("Yelp Data Conversion Script")
    print("=" * 60)
    print(f"Input directory: {PROCESSED_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Bundle strategy: {BUNDLE_STRATEGY}")
    print(f"Min bundle size: {MIN_BUNDLE_SIZE}")
    print("=" * 60)
    print()

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 步骤1-2: 加载基础数据
    id_mapping = load_id_mapping()
    user_sequences = load_user_sequences()

    # 步骤3: 转换embeddings
    embeddings = load_embeddings(id_mapping)

    # 步骤4: 转换metadata
    metadata = load_metadata()

    # 步骤5: 创建bundles
    bundles = create_bundles(user_sequences, id_mapping, strategy=BUNDLE_STRATEGY)

    # 步骤6: 划分并保存
    split_stats, bundles = split_and_save_bundles(bundles, id_mapping)

    # 步骤7: 计算统计信息
    count_json = compute_and_save_statistics(bundles, id_mapping, split_stats)

    print()
    print("=" * 60)
    print("✅ Conversion completed successfully!")
    print("=" * 60)
    print("\nGenerated files:")
    print(f"  - {OUTPUT_DIR}/clhe.pt")
    print(f"  - {OUTPUT_DIR}/train.txt ({split_stats['train']} bundles)")
    print(f"  - {OUTPUT_DIR}/valid.txt ({split_stats['valid']} bundles)")
    print(f"  - {OUTPUT_DIR}/test.txt ({split_stats['test']} bundles)")
    print(f"  - {OUTPUT_DIR}/count.json")
    print(f"  - {OUTPUT_DIR}/metadata.json")
    print()
    print("You can now train DDBC with:")
    print("  python main.py data=Yelp")


if __name__ == "__main__":
    main()
