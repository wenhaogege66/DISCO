# DISCO: DIscrete Semantic ID Diffusion with COntextual Guidance for Sequential Recommendation

> 毕业设计 · 时序推荐 · 离散扩散 · Semantic ID · Contextual Guidance

---

## 目录

1. [研究背景与动机](#1-研究背景与动机)
2. [方法演进脉络](#2-方法演进脉络)
3. [DISCO 模型](#3-disco-模型)
4. [评估指标](#4-评估指标)
5. [实验设置](#5-实验设置)
6. [当前进展](#6-当前进展)
7. [后续规划](#7-后续规划)
8. [参考文献](#8-参考文献)

---

## 1. 研究背景与动机

**时序推荐（Sequential Recommendation）** 的核心任务是：给定用户的历史交互序列，预测其下一个（或下几个）感兴趣的 item。这一任务在电商、流媒体、社交平台等场景中具有重要的实用价值。

随着深度学习的发展，时序推荐经历了从传统判别式模型到生成式模型的范式转变。然而，每一代方法都存在固有的局限性，推动了下一代方法的诞生。DISCO 正是在这一演进脉络中，尝试融合**离散扩散**与 **Semantic ID** 两条技术路线，并引入**上下文引导（Contextual Guidance）** 机制，以解决现有方法的核心问题。

---

## 2. 方法演进脉络

### 2.1 传统时序推荐模型

早期时序推荐依赖 RNN 或 Attention 机制对用户行为序列建模，将每个 item 表示为一个固定的 embedding 向量，通过点积或 softmax 进行候选排序。

| 模型 | 核心思路 | 局限性 |
|------|---------|--------|
| **GRU4Rec** | GRU 对点击序列建模，捕捉时序依赖 | 长程依赖弱，item embedding 固定 |
| **SASRec** | 单向 Transformer，自注意力建模序列 | 单一确定性预测，无法建模偏好不确定性 |
| **BERT4Rec** | 双向 Transformer + Cloze 任务 | 推理时无法利用双向上下文 |
| **DuoRec / CL4Rec** | 对比学习增强序列表示 | 仍依赖固定 item embedding，表达能力受限 |
| **HSTU** | 万亿参数级层次化序列转换单元（Meta, ICML 2024） | 工业级规模，学术场景难以复现 |

**核心问题**：这些方法将 item 表示为**固定向量**，无法捕捉 item 的多面性（latent aspects）和用户偏好的不确定性。预测结果是确定性的单点估计，缺乏对偏好分布的建模能力。

---

### 2.2 基于扩散模型的推荐

扩散模型（Diffusion Models）通过学习数据分布的去噪过程，天然具备建模不确定性和多样性的能力，被引入推荐系统以克服固定 embedding 的局限。

#### DiffuRec（2023）

[DiffuRec](https://arxiv.org/abs/2304.00686) 是最早将 DDPM 引入时序推荐的工作之一。其核心思路是：将目标 item 的 embedding 视为"干净信号"，在扩散阶段加噪为高斯分布，再以用户历史序列为条件进行去噪，最终生成目标 item 的 embedding 表示，通过相似度检索得到推荐结果。

**贡献**：将 item 表示从固定向量扩展为分布，能够建模用户偏好的不确定性。

**局限**：
- 扩散空间为**连续 embedding 空间**，与离散 item 空间存在天然的 gap，检索时依赖近似最近邻；
- item embedding 仍是随机初始化的 ID embedding，缺乏语义信息；
- 无显式的偏好引导机制，生成方向不够精准。

#### DreamRec（NeurIPS 2023）

[DreamRec](https://arxiv.org/abs/2310.20453) 在 DiffuRec 基础上引入了**Classifier-Free Guidance（CFG）**：训练时以一定概率 drop 掉历史序列条件，推理时用引导强度 `w` 混合有条件和无条件预测：

```
x̂₀ = (1 + w) · f(xₜ, h) − w · f(xₜ, ∅)
```

**贡献**：CFG 机制显著提升了生成 oracle embedding 的方向性，使推荐结果更贴近用户真实偏好。

**局限**：
- 仍在**连续 embedding 空间**扩散，item 语义信息未被充分利用；
- 模型预测的是单一 oracle embedding，对多样化偏好建模能力有限；
- 无法直接生成 item 标识符，必须依赖 embedding 检索，存在误差累积。

#### PreferDiff（2024）

[PreferDiff](https://arxiv.org/abs/2410.13117) 针对扩散推荐模型的**排序损失**问题提出改进，设计了专门面向扩散推荐器的个性化排序损失，引入多个负样本以更好地捕捉用户偏好，并解决了 hard negative 问题，加速收敛。

**局限**：同样在连续空间操作，未解决语义 gap 问题。

**扩散推荐的共同瓶颈**：
1. **连续空间 vs 离散 item**：扩散在 embedding 空间进行，最终仍需检索映射回离散 item，引入误差；
2. **语义缺失**：item embedding 通常是随机初始化的协同过滤向量，缺乏内容语义；
3. **生成粒度粗**：以整个 item embedding 为生成目标，无法利用 item 内部的层次结构。

---

### 2.3 基于 Semantic ID 的推荐

Semantic ID 方法的核心思想是：用**内容语义驱动的离散 token 序列**来表示每个 item，而非随机分配的整数 ID。这使得推荐模型能够直接在语义空间中进行生成，无需 embedding 检索。

#### TIGER（NeurIPS 2023）

[TIGER](https://arxiv.org/abs/2305.05065) 是 Semantic ID 推荐的奠基工作。其流程为：

1. 用 RQ-VAE（Residual Quantization VAE）将 item 的内容 embedding（如文本、图像特征）量化为层次化离散 token 序列（Semantic ID）；
2. 训练 Transformer seq2seq 模型，以用户历史的 Semantic ID 序列为输入，自回归生成下一个 item 的 Semantic ID；
3. 通过 Semantic ID 直接定位 item，无需 embedding 检索。

**贡献**：首次实现了在语义离散空间中端到端的生成式推荐，语义相近的 item 共享 token 前缀，天然具备泛化能力。

**局限**：
- RQ-VAE 的量化过程与推荐任务解耦，存在**目标不一致**问题；
- 自回归生成对 token 顺序敏感，而 item 的 Semantic ID 各位之间并非严格的自然语言顺序关系；
- 无法建模用户偏好的不确定性（确定性生成）。

#### LETTER（2024）

[LETTER](https://arxiv.org/abs/2405.07314)（LEarnable Tokenizer for generaTivE Recommendation）针对 TIGER 的量化-推荐目标不一致问题，提出**可学习的 tokenizer**：

- 引入 RQ-VAE 语义正则化、对比对齐损失（collaborative regularization）和多样性损失（code assignment diversity）；
- 使 Semantic ID 的学习过程同时考虑内容语义和协同过滤信号；
- 缓解了 codebook 利用率低（code assignment bias）的问题。

#### ActionPiece（2025）

[ActionPiece](https://arxiv.org/abs/2502.13581) 提出**上下文感知的 action 序列 tokenization**：同一个 item 在不同上下文中可以被 tokenize 为不同的 token，使 Semantic ID 具备上下文依赖性，更好地捕捉用户行为的语境信息。

**Semantic ID 推荐的共同局限**：
- 现有方法均采用**自回归生成**，对 token 顺序敏感，而 RVQ 的各层 code 之间并非严格的序列依赖关系；
- 缺乏对用户偏好不确定性的建模，生成结果确定性强；
- 没有显式的引导机制，生成方向难以精确控制。

---

## 3. DISCO 模型

### 3.1 核心思想

DISCO（**DI**screte **S**emantic ID diffusion with **CO**ntextual guidance）融合了扩散模型和 Semantic ID 两条技术路线的优势，同时引入上下文引导机制，旨在解决上述各类方法的核心局限。

**关键设计选择**：

| 设计维度 | 传统扩散推荐 | 传统 Semantic ID 推荐 | DISCO |
|---------|------------|---------------------|-------|
| 扩散空间 | 连续 embedding | — | **离散 token 空间** |
| item 表示 | 随机 ID embedding | RVQ Semantic ID | **RVQ Semantic ID** |
| 生成方式 | 连续去噪 | 自回归 | **掩码离散去噪（非自回归）** |
| 偏好引导 | CFG（DreamRec）| 无 | **Contextual Guidance（规划中）** |
| 顺序敏感性 | 高 | 高（自回归） | **低（并行去噪，集合语义）** |

### 3.2 模型架构

```
用户历史序列 [i₁, i₂, ..., iₙ]
        ↓  RVQ Tokenization
历史 token 序列 [BOS | item₁_tokens | ... | itemₙ_tokens]
        ↓  Transformer Encoder (context embedding h)

目标 bundle/item 的 Semantic ID tokens（部分被 [MASK] 遮盖）
        ↓  离散扩散去噪（DiT Transformer）
        ↓  条件：h（用户历史上下文）
        ↓  引导：Contextual Guidance（规划中）
去噪后的 token 序列
        ↓  RVQ Codebook 检索
推荐 item 列表
```

**RVQ Tokenization**：每个 item 被编码为 `n_codebooks` 个离散 code 的序列（Semantic ID），语义相近的 item 共享高层 code 前缀，形成层次化的语义结构。

**掩码离散扩散**：前向过程将目标 item 的 token 随机替换为 `[MASK]`（absorbing state），反向过程由 DiT Transformer 并行预测所有被掩盖的 token。相比自回归生成，这一方式：
- 不依赖 token 的生成顺序，更适合 RVQ 的层次结构；
- 支持并行解码，推理效率更高；
- 天然建模了预测的不确定性（多次采样可得不同结果）。

### 3.3 Contextual Guidance（规划中）

受 DreamRec 的 CFG 启发，DISCO 将在去噪过程中引入**上下文引导**机制：

```
ε̂ = (1 + w) · Denoiser(xₜ, h_context) − w · Denoiser(xₜ, ∅)
```

其中 `h_context` 可以是：
- **序列 embedding**：用户历史交互序列的 Transformer 编码；
- **语义 embedding**：item 内容特征（文本/图像）的预训练表示；
- **混合 embedding**：两者的融合，提供更丰富的引导信号。

**引导的动机**：纯掩码扩散的去噪方向由训练数据的统计分布决定，缺乏对特定用户偏好的精确引导。Contextual Guidance 通过放大条件信号与无条件信号的差异，使生成结果更贴近用户的个性化偏好，同时保持生成多样性。

---

## 4. 评估指标

所有指标均采用 **Counter-based 序列模式**（`allow_duplicate_items=True`），即用多重集合（multiset）计算交集，与 DDBC 评估器完全对齐。

设预测集合为 $\hat{Y} = [\hat{y}_1, ..., \hat{y}_k]$，标签集合为 $Y = [y_1, ..., y_m]$，$\text{cnt}(x, S)$ 表示 $x$ 在集合 $S$ 中出现的次数。

**多重集合交集**：

$$|{\hat{Y} \cap Y}|_{\text{count}} = \sum_{x} \min(\text{cnt}(x, \hat{Y}),\ \text{cnt}(x, Y))$$

### 4.1 主要指标

| 指标 | 公式 | 含义 |
|------|------|------|
| **Recall@k** | $\frac{|\hat{Y} \cap Y|_{\text{count}}}{|Y|}$ | 预测覆盖了多少标签 |
| **Precision@k** | $\frac{|\hat{Y} \cap Y|_{\text{count}}}{k}$ | 预测中有多少是正确的 |
| **Hit@j** | $\mathbb{1}[|\hat{Y} \cap Y|_{\text{count}} \geq j]$ | 是否至少命中 j 个标签 |

### 4.2 序列匹配指标（SeqMatch 系列）

针对时序推荐场景设计，考虑预测顺序与标签顺序的匹配程度：

| 指标 | 说明 |
|------|------|
| **SeqMatch (SM)** | 序列级精确匹配率，衡量预测序列与标签序列的整体一致性 |
| **Sequential Hit (SH)** | 序列命中率，衡量预测序列中至少有一个位置与标签完全匹配的比例 |
| **Sequential Normalized (SN)** | 归一化序列匹配分数，对部分匹配给予连续奖励 |

### 4.3 细粒度命中指标

| 指标 | 说明 |
|------|------|
| **recall@1** | 预测的第 1 个 item 命中标签的比例 |
| **precision@1** | 预测的第 1 个 item 的精确率 |
| **hit_1@1** | 预测中至少命中 1 个标签（k=1） |
| **hit_2@1** | 预测中至少命中 2 个标签（k=1） |
| **hit_3@1** | 预测中至少命中 3 个标签（k=1） |
| **hit_full@1** | 预测完全覆盖所有标签（k=1） |

### 4.4 候选集设置

为保证公平比较，所有模型使用**相同的候选集**：

- 候选集大小：`predict_n × multiplier`，其中 multiplier ∈ {9, 19, 49, 99}
- 测试候选集：直接使用 DDBC 预构建的 pkl 文件（固定 seed=1）
- 验证候选集：从验证集标签 + 随机负样本构建，首次运行时缓存

---

## 5. 实验设置

### 5.1 数据集

| 数据集 | 领域 | Items | 训练样本 | 验证样本 | 测试样本 | 状态 |
|--------|------|-------|---------|---------|---------|------|
| **Yelp** | 本地商户 | 20,033 | 627,966 | 9,967 | 19,937 | ✅ 已完成 |
| **Amazon Beauty** | 美妆个护 | — | — | — | — | 🔲 规划中 |
| **Steam** | 游戏 | — | — | — | — | 🔲 规划中 |

所有数据集统一格式：序列长度 10，0-based item ID，PAD = item_num。

### 5.2 Baseline 模型

#### 传统时序推荐

| 模型 | 来源 | 核心方法 |
|------|------|---------|
| **GRU4Rec** | RecSys 2015 | GRU 序列建模 |
| **SASRec** | ICDM 2018 | 单向 Transformer |
| **BERT4Rec** | RecSys 2019 | 双向 Transformer + Cloze |
| **DuoRec (CL4Rec)** | WSDM 2022 | 对比学习增强序列表示 |
| **HSTU** | ICML 2024 | 层次化序列转换单元（Meta） |

#### 基于扩散模型

| 模型 | 来源 | 核心方法 |
|------|------|---------|
| **DreamRec** | NeurIPS 2023 | 连续扩散 + CFG，生成 oracle embedding |
| **DiffuRec** | ACM TOIS 2023 | 连续扩散，item embedding 分布建模 |
| **PreferDiff** | arXiv 2024 | 扩散推荐专用排序损失 |

#### 基于 Semantic ID

| 模型 | 来源 | 核心方法 |
|------|------|---------|
| **TIGER** | NeurIPS 2023 | RQ-VAE Semantic ID + seq2seq 自回归生成 |
| **LETTER** | arXiv 2024 | 可学习 tokenizer，协同+语义联合优化 |
| **ActionPiece** | arXiv 2025 | 上下文感知 action 序列 tokenization |

### 5.3 DISCO 超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `rq_n_codebooks` | 3 | RVQ 层数 |
| `rq_codebook_size` | 256 | 每层 codebook 大小 |
| `model.hidden_size` | 64 | 与 DreamRec 对齐 |
| `sampling.steps` | 25 | 去噪步数 |
| `parameterization` | subs | 替换参数化 |
| `eval.predict_num_items` | 3 / 5 | 预测 item 数 |

---

## 6. 当前进展

### ✅ 已完成

- **DISCO 核心模型**：基于 MDLM 框架实现了离散扩散 + RVQ Semantic ID 的时序推荐，支持 bundle 和 sequence 两种任务模式
- **DreamRec 对齐**：修改 DreamRec 评估逻辑，使其与 DISCO 使用完全相同的候选集、指标计算方式（Counter-based）和数据划分
- **Yelp 数据集**：完成 DDBC → DreamRec 格式转换，两个模型在完全相同的数据上训练和评估
- **评估框架**：实现 16 组指标（2 predict_nums × 4 multipliers × 2 splits），支持 TensorBoard 可视化和 best checkpoint 自动保存
- **初步对比实验**：在 Yelp 上完成 DISCO vs DreamRec 的基准对比
- **TIGER 框架**：GenRec 框架（T5 + 自回归 Semantic ID）已在 Yelp 上完成初步训练（`TIGER/`，原 `gr-20251202/`）；评估对齐进行中

### 🔄 进行中

- 分析 DISCO 的 RVQ 检索重复问题（~99.8% 的重复预测来自 codebook 碰撞，而非扩散模型本身）
- 调优 DreamRec 超参数（dropout=0.15, L2=1e-4, w=10）以减少过拟合

---

## 7. 后续规划

### 7.1 模型改进：Contextual Guidance

**目标**：在 DISCO 的离散扩散去噪过程中引入上下文引导，使生成方向更贴近用户个性化偏好。

**计划方案**：
1. 参考 DreamRec 的 CFG 机制，在训练时以概率 `p` drop 掉历史序列条件；
2. 推理时用引导强度 `w` 混合有条件和无条件的 logit 预测；
3. 探索将序列 embedding 和语义 embedding 作为不同粒度的引导信号；
4. 对比 guidance 前后的 recall、precision、hit 指标变化。

### 7.2 数据集扩展

- **Amazon Beauty**：美妆类电商数据，item 具有丰富的文本/图像内容特征，适合验证 Semantic ID 的语义泛化能力
- **Steam**：游戏平台数据，用户行为序列较长，适合验证长序列建模能力

### 7.3 Baseline 完善

按优先级：
1. 完成 TIGER 评估对齐（接入共享候选集 + Counter-based 指标，`TIGER/` 目录）
2. 复现 DiffuRec 和 PreferDiff（扩散 baseline，与 DreamRec 形成扩散方法组对比）
3. 复现 LETTER（Semantic ID baseline，与 DISCO 形成离散生成方法组对比）
4. 复现传统 baseline（SASRec、BERT4Rec、GRU4Rec 等，作为下界参考）
5. ActionPiece 作为最新 Semantic ID 方法的对比

### 7.4 消融实验

| 消融维度 | 对比设置 |
|---------|---------|
| 离散 vs 连续扩散 | DISCO vs DreamRec（相同 guidance） |
| Semantic ID vs 随机 ID | DISCO vs DISCO w/o RVQ |
| Guidance 强度 | w ∈ {0, 2, 5, 10, 20} |
| Codebook 规模 | n_codebooks ∈ {2, 3, 4}，size ∈ {128, 256, 512} |
| 去噪步数 | steps ∈ {10, 25, 50, 100} |

---

## 8. 参考文献

### 传统时序推荐
- **GRU4Rec**: Hidasi et al., *Session-based Recommendations with Recurrent Neural Networks*, ICLR 2016
- **SASRec**: Kang & McAuley, *Self-Attentive Sequential Recommendation*, ICDM 2018
- **BERT4Rec**: Sun et al., *BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer*, CIKM 2019
- **DuoRec**: Qiu et al., *Contrastive Learning for Representation Degeneration Problem in Sequential Recommendation*, WSDM 2022
- **HSTU**: Zhai et al., [*Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations*](https://arxiv.org/abs/2402.17152), ICML 2024

### 基于扩散模型
- **DiffuRec**: Li et al., [*DiffuRec: A Diffusion Model for Sequential Recommendation*](https://arxiv.org/abs/2304.00686), ACM TOIS 2023
- **DreamRec**: Yang et al., [*Generate What You Prefer: Reshaping Sequential Recommendation via Guided Diffusion*](https://arxiv.org/abs/2310.20453), NeurIPS 2023
- **PreferDiff**: Liu et al., [*Preference Diffusion for Recommendation*](https://arxiv.org/abs/2410.13117), arXiv 2024

### 基于 Semantic ID
- **TIGER**: Rajput et al., [*Recommender Systems with Generative Retrieval*](https://arxiv.org/abs/2305.05065), NeurIPS 2023
- **LETTER**: Zhang et al., [*Learnable Item Tokenization for Generative Recommendation*](https://arxiv.org/abs/2405.07314), arXiv 2024
- **ActionPiece**: Hou et al., [*ActionPiece: Contextually Tokenizing Action Sequences for Generative Recommendation*](https://arxiv.org/abs/2502.13581), arXiv 2025

### 扩散模型基础
- **MDLM**: Sahoo et al., *Simple and Effective Masked Diffusion Language Models*, NeurIPS 2024
- **DDPM**: Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020
- **CFG**: Ho & Salimans, *Classifier-Free Diffusion Guidance*, NeurIPS Workshop 2021
