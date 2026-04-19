# CLAUDE_zh.md

本文件为 Claude Code (claude.ai/code) 提供在本仓库中工作时的指引。

---

## 1. 项目概述

本工作区是一个时序推荐模型对比研究项目。**DISCO** 是我们自研的核心模型，其余目录均为 baseline。

**最终目标**：在相同数据、相同候选集、相同指标下，对多个 baseline 进行公平对比。

- 所有 baseline 均 clone 到 `wenhao/<ModelName>/`，并按统一评估协议对齐（见第 8–9 节）。
- **DreamRec** 是多 item 预测适配的**参考实现**——接入新 baseline 时请先研究它。

---

## 2. 工作区目录结构（权威）

```
wenhao/
├── DISCO/          ← 核心模型（原 DDBC_f-main）
├── DreamRec/       ← Baseline ✅ 已完全对齐
├── TIGER/          ← Baseline 🔄 已训练，评估未对齐（原 gr-20251202）
├── DiffuRec/       ← [待 clone] 连续扩散 baseline
├── LETTER/         ← [待 clone] Semantic ID baseline
├── GRU4Rec/        ← [待 clone] 传统 baseline
├── SASRec/         ← [待 clone] 传统 baseline
├── BERT4Rec/       ← [待 clone] 传统 baseline
├── convert_ddbc_to_dreamrec.py   ⚠ 第 20 行仍硬编码 DDBC_f-main/，使用前需改为 DISCO/
└── CLAUDE.md / CLAUDE_zh.md / README.md
```

---

## 3. 运行环境

| 模型 | conda 环境 | 说明 |
|------|-----------|------|
| DISCO | `DDBC` | Python 3.9.21，PyTorch 2.2.0+cu121 |
| DreamRec | `DDBC` | 复用 DDBC |
| TIGER | `DDBC` | 复用 DDBC |
| DiffuRec | `DDBC` | 复用 DDBC；论文写的 PyTorch 1.8.0，但在 2.2.0 上运行正常 |
| LETTER | `DDBC` | 复用 DDBC（仅 LETTER-TIGER 变体；transformers 4.57.3 兼容） |
| GRU4Rec | `DDBC` | 复用 DDBC；已通过 pip 向 DDBC 追加安装 `optuna` 和 `pexpect` |
| SASRec | `DDBC` | 复用 DDBC；依赖极简（torch + numpy） |
| BERT4Rec | `BERT4Rec` | **独立环境** — TF 1.15.0，Python 3.7，protobuf 3.20.3；通过 `conda activate BERT4Rec` 启动 |

```bash
conda activate DDBC      # 用于 DISCO、DreamRec、TIGER、DiffuRec、LETTER、GRU4Rec、SASRec
conda activate BERT4Rec  # 仅用于 BERT4Rec
```

---

## 4. 核心模型：DISCO

完整文档见 `DISCO/CLAUDE.md`，以下为摘要。

```bash
# 训练
bash DISCO/scripts/train_yelp.sh

# 评估（rec_eval 模式）
bash DISCO/scripts/test.sh

# 清除数据集缓存（修改 rq_n_codebooks 或 rq_codebook_size 后必须执行）
bash DISCO/scripts/clear_dataset_cache.sh
```

**关键配置**（均通过 Hydra override 传入）：
- `mode`：`train` / `rec_eval` / `ppl_eval` / `sample_eval`
- `rq_n_codebooks` / `rq_codebook_size`：RVQ 结构参数，修改后需清除缓存
- `model.length`：必须满足 `1 + (物品数 × (n_codebooks + 2)) + 1`
- `eval.predict_num_items`：每次预测的物品数量

**数据集目录**：`DISCO/datasets/Yelp/`
- `train.txt`、`valid.txt`、`test.txt` — 纯文本，格式：`bundle_id, item_1, ..., item_10`（0-based ID）
- `test_candidates_seed1_x{9,19,49,99}_items{3,5}.pkl` — **所有模型共享的测试候选集**

**Item embedding**：DISCO 使用 CLHE embedding（`clhe.pt`），经 PCA + RQ-VAE 处理生成 Semantic ID（RVQ token）。这是 DISCO 特有的；其他模型使用各自的 embedding 方案。

---

## 5. Baseline：DreamRec ✅

**论文**：[NeurIPS 2023 — "Generate What You Prefer"](https://arxiv.org/abs/2310.20453)

这是多 item 预测适配的**参考实现**，接入新 baseline 时请研究 `DreamRec/DreamRec.py`。

### 数据（Yelp — 已生成）

由 `convert_ddbc_to_dreamrec.py` 从 DISCO 的 txt 文件生成。

- **物品空间**：0-based ID，`item_num=20033`，`seq_size=10`，`PAD=20033`
- **`DreamRec/data/yelp/` 中的文件**：
  - `data_statis.df`：seq_size=10，item_num=20033
  - `train_data.df`：627,966 条——增长序列格式（`seq`、`len_seq`、`next`）
  - `valid_data_items3.df` / `valid_data_items5.df`：9,967 条——多标签格式（`seq`、`len_seq`、`labels`）
  - `test_data_items3.df` / `test_data_items5.df`：19,937 条——多标签格式
  - `valid_candidates_seed100_x{mult}_items{n}.pkl`：首次评估时自动生成并缓存
- **测试候选集**：直接从 `DISCO/datasets/Yelp/test_candidates_seed1_x{mult}_items{n}.pkl` 加载

重新生成数据文件：
```bash
conda run -n DDBC python convert_ddbc_to_dreamrec.py
# ⚠ 先将脚本第 20 行的 DDBC_f-main/ 改为 DISCO/
```

### 训练

```bash
bash DreamRec/scripts/train_yelp.sh
```

`train_yelp.sh` 中的关键超参数：
- `EPOCH=75`、`BATCH_SIZE=256`、`LR=0.001`、`TIMESTEPS=500`、`BETA_SCHE=exp`、`W=10`
- `DROPOUT_RATE=0.15`、`L2_DECAY=1e-4`
- `PREDICT_NUMS="3"`、`CANDIDATE_MULTIPLIERS="19"`、`EVAL_FREQ=5`
- `PREDICT_MODE="single"` — 验证期间固定用 `single`（快速）；最终测试可改为 `ar`
- `TOPK=1` — SM@K 每步命中检查取前 K 名

### 架构

1. **Transformer 编码器**：读取历史物品序列 → 上下文 embedding `h`（64 维）
2. **逆向扩散**（DDPM）：以 `h` 为条件从 `x_T ~ N(0,I)` 去噪到 `x_0`，使用 CFG：`x_0 = (1+w)*cond - w*uncond`
3. **候选集打分**：`scores = candidate_embs @ x_0` → 取 top-predict_n 个物品

### 评估协议

每隔 `EVAL_FREQ` 个 epoch，对 valid 和 test 各调用一次 `evaluate_ddbc()`，共输出 **16 套指标**（2 predict_nums × 4 multipliers × 2 splits）。最优 checkpoint 以 `val_recall@3_x9` 为准。

**`DreamRec/DreamRec.py` 中的关键函数**：
`_load_or_build_candidate_pool()`、`_seq_mode_metrics()`、`_stepwise_sm_metrics()`、`evaluate_ddbc()`

**DreamRec 固有限制**：top-k 预测结果不含重复物品，当 label 含重复时，recall 上限为 `len(unique_labels) / predict_n`。

**预测模式**：
- `single`：单次扩散推理 → 对所有候选打分 → 取 top-predict_n（快速，约 50s/次评估）
- `ar`：自回归模式 — 每步完整扩散取 top-1，追加到历史，重复 predict_n 步

---

## 6. Baseline：TIGER 🔄

**论文**：[NeurIPS 2023 — "Recommender Systems with Generative Retrieval"](https://arxiv.org/abs/2305.05065)

- **目录**：`TIGER/`（原 `gr-20251202/`）
- **框架**：GenRec — T5 encoder-decoder + 自定义 beam search，自回归 Semantic ID 生成
- **RVQ 配置**：3 codebooks × 256 entries（与 DISCO 结构相同）

```bash
# 在 Yelp 上训练（多 GPU）
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 12347 \
    TIGER/main.py --model=TIGER --dataset=Yelp --category=Yelp
# 或使用脚本：
bash TIGER/train_tiger_yelp.sh
```

**状态**：Yelp 上已完成初步训练。评估使用 NDCG/Recall——**尚未与 DISCO 指标对齐**。

**待完成**：
1. 实现多 item beam search（predict top-N，N ∈ {3, 5}）
2. 接入共享测试候选集：`DISCO/datasets/Yelp/test_candidates_seed1_x*_items*.pkl`
3. 实现 `evaluate_ddbc()` 等价函数（Counter-based 指标）

**注意**：内部脚本可能仍引用 `gr-20251202/`，接入评估时检查。

---

## 7. 待接入的 Baseline

| 模型 | 类别 | 论文 | 目录 | 状态 | 关键备注 |
|------|------|------|------|------|---------|
| DiffuRec | 连续扩散 | ACM TOIS 2023 | `DiffuRec/` | 已 clone，待集成 | 单 item → 多 item 预测适配；环境：DDBC |
| LETTER | Semantic ID | arXiv 2024 | `LETTER/` | 已 clone，待集成 | 使用 LETTER-TIGER 变体；环境：DDBC |
| GRU4Rec | 传统 | ICLR 2016 | `GRU4Rec/` | 🔄 已集成，待训练 | `train_yelp.py` + `scripts/train_yelp.sh` 已就绪；训练 TSV 已生成；环境：DDBC |
| SASRec | 传统 | ICDM 2018 | `SASRec/` | 已 clone，待集成 | `DreamRec/SASRec.py` 内已有实现；环境：DDBC |
| BERT4Rec | 传统 | CIKM 2019 | `BERT4Rec/` | 已 clone，待集成 | TF 1.x 代码库；环境：BERT4Rec（独立） |

---

## 8. 跨模型一致性约束——哪些东西必须保持相同

以 DreamRec vs DISCO 为参照，**所有模型**必须在以下维度与 DISCO 保持完全一致，结果才具有可比性：

| 维度 | 必须一致的值/文件 | 说明 |
|------|-----------------|------|
| **物品 ID 空间** | 0-based，`item_num=20033` | 不得做任何 ID 映射或偏移 |
| **PAD token** | `item_num`（即 20033） | 序列不足时用此值填充 |
| **序列长度** | 固定 10 | 每条记录的 item 序列均为 10 |
| **源序列数据** | `DISCO/datasets/Yelp/{train,valid,test}.txt` | 所有模型的划分必须来自这批文件 |
| **测试候选集** | `DISCO/datasets/Yelp/test_candidates_seed1_x{9,19,49,99}_items{3,5}.pkl` | **直接加载，绝对不能自行重建** |
| **验证候选集** | 由验证 label + 随机负样本构建，seed 和构建逻辑与 DreamRec 一致 | DreamRec 以 seed=100 缓存 |
| **评估 label 划分** | `items3`：前 7 为 input，后 3 为 label；`items5`：前 5 为 input，后 5 为 label | valid 和 test 均如此 |
| **预测数量** | predict_n ∈ {3, 5} | 必须与 label 数对应 |
| **候选集倍数** | multiplier ∈ {9, 19, 49, 99} | 候选池大小 = predict_n × multiplier |

**哪些可以不同**：
- 训练数据格式（txt、pickle 等，各模型自行处理）
- 训练样本构造方式（growing sequence、完整 bundle 等）
- Item embedding 初始化（DISCO 用 CLHE+RVQ；DreamRec 用随机 `nn.Embedding`；TIGER/LETTER 用 item 内容特征）
- 模型内部 padding 处理细节，只要最终 item ID 在 `[0, 20032]` 范围内即可

---

## 9. 新 Baseline 集成 Checklist

接入新 baseline 的标准步骤：

1. **Clone** 到 `wenhao/<ModelName>/`
2. **确认物品 ID 空间**：0-based，item_num=20033，PAD=20033，不得重映射
3. **准备 Yelp 数据**：格式按模型需求自定，但底层序列必须来自 `DISCO/datasets/Yelp/{train,valid,test}.txt`；可新写 converter（参考 `convert_ddbc_to_dreamrec.py`）
4. **实现多 item 预测**：predict top-N，N ∈ {3, 5}
5. **接入共享测试候选集**（直接 load，不复制不重建）：`DISCO/datasets/Yelp/test_candidates_seed1_x{9,19,49,99}_items{3,5}.pkl`
6. **实现 `evaluate_ddbc()` 等价函数**：Counter-based 指标，2 predict_nums × 4 multipliers（参考 `DreamRec/DreamRec.py`）
7. **添加训练脚本**：`<ModelName>/scripts/train_yelp.sh`
8. **在第 7 节对应条目标记 ✅**

---

## 10. 共享数据与评估协议

**数据来源（唯一真相源）**：`DISCO/datasets/Yelp/`

**候选集**：
- `test`：`DISCO/datasets/Yelp/test_candidates_seed1_x{9,19,49,99}_items{3,5}.pkl` — 共享，固定，绝不重建
- `valid`：各模型从验证集 label + 随机负样本构建，缓存为 pkl

**评估网格**：2 predict_nums × 4 multipliers = 每个 split 8 组指标

**默认评估配置**（所有 baseline 对比均使用）：
- `predict_n = 3`，`multiplier = 19`（候选集大小 = 57）
- 最优 checkpoint 选择：`val_recall@3_x19`

**最优 checkpoint 选择**：`val_recall@3_x9`

**DISCO 参考指标**（测试集，predict_n=3，x19，topk=1）：
```
recall@1=0.1624  precision@1=0.1624  hit_1@1=0.4216  hit_2@1=0.0636
hit_3@1=0.0021   hit_4@1=0.0        hit_5@1=0.0     hit_full@1=0.0021
sm@1=0.0629      sh@1=0.1886        sn@1=0.0629
```
所有 baseline 必须以相同格式汇报指标，以便直接比较。

**指标**（Counter-based，`allow_duplicate_items=True`，所有模型完全一致）：
- `recall`、`precision` — 多重集合交集 / predict_n 或 label_n
- `hit_1` ~ `hit_5`、`hit_full` — 命中数阈值
- `SM@K`、`SH@K`、`SN@K` — 逐位置序列匹配（见第 11 节）

**TensorBoard**：各模型日志写入各自的 `<ModelName>/tensorboard/`

---

## 11. 评估指标说明

**当前启用指标**：`recall@1`、`precision@1`、`hit_1@1` ~ `hit_5@1`、`hit_full@1`、`sm@1`、`sh@1`、`sn@1`

**SM / SH / SN — 逐位置序列匹配**：
- `SH@K = sum_{t=1}^{T} 1(y_t ∈ TopK_t)` — 所有位置的命中原始计数
- `SM@K = SH@K / T` — 归一化（命中位置比例）
- `SN@K = SM@K` — 与 SM 定义相同
- **逐位置**：在位置 t，TopK_t 是该位置预测 embedding 检索到的前 K 个物品，每个位置独立检查

**两个 K 参数（不要混淆）**：

| 参数 | 位置 | 含义 |
|------|------|------|
| `topk: [1]` | DISCO `config.yaml` evaluator | Beam/样本维度——每个样本生成几套预测，取并集计算 recall/precision/hit（Oracle@K 风格） |
| `sm_topk: 1` | DISCO `config.yaml` evaluator | SM@K 检索 K——每个位置取前 K 名判断是否命中 |
| `TOPK=1` | DreamRec `train_yelp.sh` | DreamRec 的 SM@K K 值，通过 `--topk` 传入，作用与 `sm_topk` 相同 |

**DISCO 生成过程是随机的**（`_sample_categorical` 中 Gumbel-max trick）：
```python
gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
return (categorical_probs / gumbel_norm).argmax(dim=-1)
```
等价于 multinomial 采样——`topk > 1` 时多套预测序列真正不同。

---

## 12. 已知问题

- `convert_ddbc_to_dreamrec.py` 第 20 行硬编码 `DDBC_f-main/datasets/Yelp`，下次使用前需改为 `DISCO/datasets/Yelp`
- `TIGER/` 内部脚本可能仍引用 `gr-20251202/`，接入评估时检查
- **修改代码后不要创建 markdown 总结文档**，直接在对话中说明改动内容

## 13. GRU4Rec 数据文件

- `GRU4Rec/data/yelp/train_yelp.tsv` — 由 `DISCO/datasets/Yelp/train.txt` 生成（697,740 行，69,774 个 session）
- 评估数据复用 `DreamRec/data/yelp/{valid,test}_data_items{n}.df`，无需单独拷贝
- 验证候选集首次运行时自动构建并缓存至 `GRU4Rec/data/yelp/valid_candidates_seed100_x{mult}_items{n}.pkl`
- 测试候选集直接从 `DISCO/datasets/Yelp/test_candidates_seed1_x{mult}_items{n}.pkl` 加载
