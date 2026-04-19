# Yelp 数据结构与训练增广对照（DISCO + 7 baselines）

## 0. 统一前提

- 原始来源：`TIGER/cache/Yelp/Yelp_2020/raw`
- 处理后序列：`TIGER/cache/Yelp/Yelp_2020/processed/all_item_seqs.json`
- DISCO 数据生成：`DISCO/convert_yelp_to_ddbc.py`
- DISCO 最终切分文件：`DISCO/datasets/Yelp/{train,valid,test}.txt`

示例（DISCO 一条 10-item 序列）：

```text
0, 0, 1, 2, 3, 2, 4, 5, 2, 6, 7
```

其中第一列是 bundle/session id，后面 10 个是 0-based item id。

---

## 1. 每个模型训练时是否有“增广”

> 这里把“把一条序列展开成多个前缀样本”“随机 mask”等都记为训练时增广/样本扩展。

| 模型 | 训练输入数据结构 | 是否有训练时增广/扩展 | 备注 |
|---|---|---|---|
| **DreamRec** | `DreamRec/data/yelp/train_data.df`，列：`seq,len_seq,next` | **有**（每条10-item展开为9个增长前缀） | 来自 `convert_ddbc_to_dreamrec.py` |
| **TIGER** | `processed/all_item_seqs.json` -> tokenizer train split | **有**（train split 生成前缀样本） | 当前不是直接用 DISCO 的 train/valid/test txt |
| **DiffuRec** | `DiffuRec/data/yelp/dataset.pkl` 的 `train` | **有**（`Data_Train.split_onebyone()` 前缀展开） | 转换脚本注明此行为 |
| **LETTER (LETTER-TIGER)** | `LETTER/data/Yelp/Yelp.inter.json` | **有**（`for i in range(1,len(items))` 前缀展开） | 当前是用户全序列风格 |
| **GRU4Rec** | `GRU4Rec/data/yelp/train_yelp.tsv`（SessionId,ItemId,Time） | **有**（next-item 逐步监督，等价每session生成多步样本） | 属于模型原生训练机制 |
| **SASRec** | `SASRec/python/data/Yelp.txt`（user item） | **有**（每个位置做 next-item + 随机负采样） | `utils.py` 的 sampler |
| **BERT4Rec** | `dataset.pkl` + `BertTrainDataset` | **有**（随机 mask，`bert_mask_prob=0.15`） | 这是 BERT4Rec 核心训练方式 |

结论：你这 7 个 baseline 训练阶段都存在某种“样本扩展/增广”机制（大多是模型原生）。

---

## 2. 不同项目的数据结构需求（以同一条序列为例）

以 DISCO 序列：

```text
0, 0, 1, 2, 3, 2, 4, 5, 2, 6, 7
```

### DISCO
- 文件：`train.txt`
- 结构：`bundle_id, item1..item10`
- 直接使用固定长度10序列。

### DreamRec√
- 文件：`train_data.df`
- 结构：`seq(长度10, PAD=20033), len_seq, next`
- 同一条序列会变成 9 条样本：
  - `[0,PAD,...] -> 1`
  - `[0,1,PAD,...] -> 2`
  - ...

### GRU4Rec√
- 文件：`train_yelp.tsv`
- 结构：`SessionId, ItemId, Time`
- 训练时按时间顺序做 next-item：`item_t -> item_{t+1}`。

### DiffuRec√
- 文件：`dataset.pkl['train']`
- 结构：`{session_id: [item1..item10]}`
- 内部再用 `split_onebyone()` 展开前缀序列。

### SASRec（用户时间序列）
- 文件：`python/data/Yelp.txt`
- 结构：每行 `user_id item_id`（按时间顺序）
- 内部 `data_partition` 做 leave-one-out，再在 sampler 中构造 next-item 训练对。

### BERT4Rec√
- 文件：`dataset.pkl['train']`（1-based item id）
- 结构：`{session_id: [item1..item10]}`
- 训练时对序列随机 mask（MLM 目标）。

### TIGER / LETTER-TIGER
- 输入本质：用户历史序列（`all_item_seqs` / `Yelp.inter.json`）
- 训练样本：前缀 history -> 目标 item（自回归生成语义ID）。

---

## 3. 关于“是否都是 7 输入预测 3 输出”

在你们当前 DDBC 对齐脚本里，已对齐模型（或正在对齐的脚本）都支持该目标：

- 默认主对比设置：`predict_n=3`、`multiplier=19`
- 即：**输入前 7 个 item，预测后 3 个 item**

但注意：
- 这通常是**评估协议**（evaluate_ddbc）层面的统一；
- 各模型**训练目标**仍保留其原生形式（next-item、MLM、自回归等）。

---

## 4. 公平性风险（当前最需要优先确认）

虽然 item 空间和候选集可以统一，但目前有些 baseline 训练数据来源仍是“用户全序列+内部切分”（如 TIGER/LETTER/SASRec），而不是直接使用 `DISCO/datasets/Yelp/{train,valid,test}.txt` 的固定切分。这会影响严格公平对比。

建议主实验前统一规则：

1. 训练样本必须由 `DISCO train.txt` 派生；
2. 验证/测试标签必须由 `DISCO valid/test.txt` 派生；
3. 测试候选必须直接用 `DISCO test_candidates_seed1_x19_items3.pkl`；
4. 报告统一为 Counter-based 指标格式。
