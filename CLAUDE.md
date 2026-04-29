# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 1. Project Overview

This workspace is a research project comparing sequential recommendation models. **DISCO** is our core model; all other directories are baselines.

**Final goal**: Fair multi-baseline comparison on identical data, identical candidate sets, and identical metrics.

- All baselines are cloned to `wenhao/<ModelName>/` and adapted to the shared evaluation protocol (see §8–9).
- The reference adaptation is **DreamRec** — study it first when integrating a new baseline.

---

## 2. Workspace Directory Layout

```
wenhao/
├── DISCO/          ← Core model (was DDBC_f-main)
├── DreamRec/       ← Baseline ✅ fully aligned
├── TIGER/          ← Baseline 🔄 trained, evaluation not yet aligned (was gr-20251202)
├── DiffuRec/       ← [to clone] continuous diffusion baseline
├── LETTER/         ← [to clone] Semantic ID baseline
├── GRU4Rec/        ← [to clone] traditional baseline
├── SASRec/         ← [to clone] traditional baseline
├── BERT4Rec/       ← [to clone] traditional baseline
├── convert_ddbc_to_dreamrec.py   ⚠ line 20 still hardcodes DDBC_f-main/ — update to DISCO/ before use
└── CLAUDE.md / CLAUDE_zh.md / README.md
```

---

## 3. Environment

| Model | conda env | Notes |
|-------|-----------|-------|
| DISCO | `DDBC` | Python 3.9.21, PyTorch 2.2.0+cu121 |
| DreamRec | `DDBC` | reuses DDBC |
| TIGER | `DDBC` | reuses DDBC |
| DiffuRec | `DDBC` | reuses DDBC; paper lists PyTorch 1.8.0 but runs fine on 2.2.0 |
| LETTER | `DDBC` | reuses DDBC (LETTER-TIGER variant only; transformers 4.57.3 compatible) |
| GRU4Rec | `DDBC` | reuses DDBC; `optuna` and `pexpect` added to DDBC via pip |
| SASRec | `DDBC` | reuses DDBC; minimal deps (torch + numpy only) |
| BERT4Rec | `BERT4Rec` | **separate env** — TF 1.15.0, Python 3.7, protobuf 3.20.3; activate with `conda activate BERT4Rec` |

```bash
conda activate DDBC      # for DISCO, DreamRec, TIGER, DiffuRec, LETTER, GRU4Rec, SASRec
conda activate BERT4Rec  # for BERT4Rec only
```

---

## 4. Core Model: DISCO

Full documentation in `DISCO/CLAUDE.md`. Summary:

```bash
# Train
bash DISCO/scripts/train_yelp.sh

# Evaluate (rec_eval mode)
bash DISCO/scripts/test.sh

# Clear dataset cache (required after changing rq_n_codebooks or rq_codebook_size)
bash DISCO/scripts/clear_dataset_cache.sh
```

**Key configs** (Hydra overrides):
- `mode`: `train` / `rec_eval` / `ppl_eval` / `sample_eval`
- `rq_n_codebooks` / `rq_codebook_size`: RVQ structure — changing these requires cache clear
- `model.length`: Must be `1 + (num_items × (n_codebooks + 2)) + 1`
- `eval.predict_num_items`: Number of items to predict

**Datasets**: `DISCO/datasets/Yelp/`
- `train.txt`, `valid.txt`, `test.txt` — plain text, format: `bundle_id, item_1, ..., item_10` (0-based IDs)
- `test_candidates_seed1_x{9,19,49,99}_items{3,5}.pkl` — **shared test candidates for all models**

**Item embeddings**: DISCO uses CLHE embeddings (`clhe.pt`) processed via PCA + RQ-VAE to produce Semantic IDs (RVQ tokens). This is DISCO-specific; other models use their own embedding schemes.

---

## 5. Baseline: DreamRec ✅

**Paper**: [NeurIPS 2023 — "Generate What You Prefer"](https://arxiv.org/abs/2310.20453)

This is the **reference implementation** for multi-item prediction adaptation. Study `DreamRec/DreamRec.py` when adapting a new baseline.

### Data (Yelp — already generated)

Generated from DISCO's txt files via `convert_ddbc_to_dreamrec.py`.

- **Item space**: 0-based IDs, `item_num=20033`, `seq_size=10`, `PAD=20033`
- **Files in `DreamRec/data/yelp/`**:
  - `data_statis.df`: seq_size=10, item_num=20033
  - `train_data.df`: 627,966 samples — growing-sequence format (`seq`, `len_seq`, `next`)
  - `valid_data_items3.df` / `valid_data_items5.df`: 9,967 samples — multi-label format (`seq`, `len_seq`, `labels`)
  - `test_data_items3.df` / `test_data_items5.df`: 19,937 samples — multi-label format
  - `valid_candidates_seed100_x{mult}_items{n}.pkl`: auto-generated/cached on first eval run
- **Test candidates**: loaded directly from `DISCO/datasets/Yelp/test_candidates_seed1_x{mult}_items{n}.pkl`

To regenerate data files:
```bash
conda run -n DDBC python convert_ddbc_to_dreamrec.py
# ⚠ First update DDBC_f-main/ → DISCO/ on line 20 of that script
```

### Training

```bash
bash DreamRec/scripts/train_yelp.sh
```

Key hyperparameters in `train_yelp.sh`:
- `EPOCH=75`, `BATCH_SIZE=256`, `LR=0.001`, `TIMESTEPS=500`, `BETA_SCHE=exp`, `W=10`
- `DROPOUT_RATE=0.15`, `L2_DECAY=1e-4`
- `PREDICT_NUMS="3"`, `CANDIDATE_MULTIPLIERS="19"`, `EVAL_FREQ=5`
- `PREDICT_MODE="single"` — validation always uses `single` (fast); change for final test to `ar`
- `TOPK=1` — K for SM@K per-step hit check

### Architecture

1. **Transformer encoder** reads item history → context embedding `h` (64-dim)
2. **Reverse diffusion** (DDPM): denoises from `x_T ~ N(0,I)` to `x_0` conditioned on `h`, using CFG: `x_0 = (1+w)*cond - w*uncond`
3. **Candidate scoring**: `scores = candidate_embs @ x_0` → top-predict_n items

### Evaluation Protocol

Every `EVAL_FREQ` epochs, `evaluate_ddbc()` is called for both `valid` and `test` splits — **16 metric sets** total (2 predict_nums × 4 multipliers × 2 splits). Best checkpoint by `val_recall@3_x9`.

**Key modified functions** in `DreamRec/DreamRec.py`:
`_load_or_build_candidate_pool()`, `_seq_mode_metrics()`, `_stepwise_sm_metrics()`, `evaluate_ddbc()`

**DreamRec limitation**: predicts distinct items only — for labels with duplicates, recall is capped at `len(unique_labels) / predict_n`.

**Predict modes**:
- `single`: one diffusion pass → score all candidates → top-predict_n (fast, ~50s/eval)
- `ar`: autoregressive — each step runs full diffusion, takes top-1, appends to history, repeats predict_n times

---

## 6. Baseline: TIGER 🔄

**Paper**: [NeurIPS 2023 — "Recommender Systems with Generative Retrieval"](https://arxiv.org/abs/2305.05065)

- **Directory**: `TIGER/` (was `gr-20251202/`)
- **Framework**: GenRec — T5 encoder-decoder with custom beam search, autoregressive Semantic ID generation
- **RVQ config**: 3 codebooks × 256 entries (same structure as DISCO)

```bash
# Train on Yelp (multi-GPU)
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 12347 \
    TIGER/main.py --model=TIGER --dataset=Yelp --category=Yelp
# or use the script:
bash TIGER/train_tiger_yelp.sh
```

**Status**: Initial training on Yelp complete. Evaluation uses NDCG/Recall — **not yet aligned** with DISCO metrics.

**Integration TODO**:
1. Implement multi-item beam search (predict top-N items, N ∈ {3, 5})
2. Load shared test candidates from `DISCO/datasets/Yelp/test_candidates_seed1_x*_items*.pkl`
3. Implement `evaluate_ddbc()` equivalent (Counter-based metrics)

**Note**: Internal scripts may still reference `gr-20251202/` — check when integrating.

---

## 7. Upcoming Baselines

| Model | Category | Paper | Directory | Status | Notes |
|-------|----------|-------|-----------|--------|-------|
| DiffuRec | Continuous diffusion | ACM TOIS 2023 | `DiffuRec/` | cloned, integration pending | Adapt single→multi item prediction; env: DDBC |
| LETTER | Semantic ID | arXiv 2024 | `LETTER/` | cloned, integration pending | Use LETTER-TIGER variant; env: DDBC |
| GRU4Rec | Traditional | ICLR 2016 | `GRU4Rec/` | 🔄 integrated, training pending | `train_yelp.py` + `scripts/train_yelp.sh` ready; train TSV generated; env: DDBC |
| SASRec | Traditional | ICDM 2018 | `SASRec/` | cloned, integration pending | `SASRec.py` already exists inside `DreamRec/`; env: DDBC |
| BERT4Rec | Traditional | CIKM 2019 | `BERT4Rec/` | cloned, integration pending | TF 1.x codebase; env: BERT4Rec (separate) |

---

## 8. Cross-Model Invariants — What Must Stay Identical

These dimensions must be exactly the same across **all** models for results to be comparable:

| Dimension | Required value / file | Notes |
|-----------|----------------------|-------|
| **Item ID space** | 0-based, `item_num=20033` | No remapping or offset allowed |
| **PAD token** | `item_num` = 20033 | Used for sequence padding |
| **Sequence length** | Fixed 10 | Every record has exactly 10 items |
| **Source sequences** | `DISCO/datasets/Yelp/{train,valid,test}.txt` | All models derive their splits from these files |
| **Test candidate sets** | `DISCO/datasets/Yelp/test_candidates_seed1_x{9,19,49,99}_items{3,5}.pkl` | **Load directly — never rebuild** |
| **Validation candidate sets** | Built from val labels + random negatives, same seed logic as DreamRec | Cache as pkl; seed=100 in DreamRec |
| **Eval label split** | `items3`: first 7 as input, last 3 as labels; `items5`: first 5 as input, last 5 as labels | Apply to both valid and test |
| **Predict counts** | predict_n ∈ {3, 5} | Must match label count |
| **Candidate multipliers** | multiplier ∈ {9, 19, 49, 99} | Candidate pool size = predict_n × multiplier |

**What may differ across models**:
- Training data format (txt, pickle, etc.) — each model handles its own loading
- Training sample construction (growing sequence, full bundle, etc.)
- Item embedding initialization (DISCO: CLHE+RVQ; DreamRec: random `nn.Embedding`; TIGER/LETTER: item content features)
- Internal padding mechanics, as long as item IDs stay in `[0, 20032]`

---

## 9. Baseline Integration Checklist

Steps to align a new baseline with the shared evaluation framework:

1. **Clone** to `wenhao/<ModelName>/`
2. **Verify item ID space**: 0-based, item_num=20033, PAD=20033 — no remapping
3. **Prepare Yelp data**: format per model needs, but source sequences must come from `DISCO/datasets/Yelp/{train,valid,test}.txt`. Write a new converter if needed (see `convert_ddbc_to_dreamrec.py` as reference)
4. **Implement multi-item prediction**: predict top-N items, N ∈ {3, 5}
5. **Load shared test candidates** directly: `DISCO/datasets/Yelp/test_candidates_seed1_x{9,19,49,99}_items{3,5}.pkl` — do not copy or rebuild
6. **Implement `evaluate_ddbc()` equivalent**: Counter-based metrics, 2 predict_nums × 4 multipliers (see `DreamRec/DreamRec.py` for reference implementation)
7. **Add training script**: `<ModelName>/scripts/train_yelp.sh`
8. **Mark as ✅** in §7 of this file once complete

---

## 10. Shared Data & Evaluation Protocol

**Data source** (single source of truth): `DISCO/datasets/Yelp/`

**Candidate sets**:
- `test`: `DISCO/datasets/Yelp/test_candidates_seed1_x{9,19,49,99}_items{3,5}.pkl` — shared, fixed, never rebuilt
- `valid`: built per model from val labels + random negatives, cached as pkl

**Evaluation grid**: 2 predict_nums × 4 multipliers = 8 combinations per split

**Default evaluation config** (used for all baseline comparisons):
- `predict_n = 3`, `multiplier = 19` (candidate pool = 57 items)
- Best checkpoint selection: `val_recall@3_x19`

**Best checkpoint selection**: `val_recall@3_x9`

**DISCO reference results** (test set, predict_n=3, x19, topk=1):
```
recall@1=0.1624  precision@1=0.1624  hit_1@1=0.4216  hit_2@1=0.0636
hit_3@1=0.0021   hit_4@1=0.0        hit_5@1=0.0     hit_full@1=0.0021
sm@1=0.0629      sh@1=0.1886        sn@1=0.0629
```
All baselines must report metrics in this same format for direct comparison.

**Metrics** (Counter-based, `allow_duplicate_items=True` — identical across all models):
- `recall`, `precision` — multiset intersection / predict_n or label_n
- `hit_1` ~ `hit_5`, `hit_full` — hit count thresholds
- `SM@K`, `SH@K`, `SN@K` — per-position sequential match (see §11)

**TensorBoard**: each model logs to its own `<ModelName>/tensorboard/`

---

## 11. Evaluation Metrics Reference

**Active metrics**: `recall@1`, `precision@1`, `hit_1@1` ~ `hit_5@1`, `hit_full@1`, `sm@1`, `sh@1`, `sn@1`

**SM / SH / SN — Sequential Match**:
- `SH@K = sum_{t=1}^{T} 1(y_t ∈ TopK_t)` — raw hit count across all positions
- `SM@K = SH@K / T` — normalised (fraction of positions hit)
- `SN@K = SM@K` — same as SM by definition
- **Per-position**: at position t, TopK_t is the top-K retrieved items for that position's predicted embedding

**Two K parameters (do not confuse)**:

| Parameter | Location | Meaning |
|-----------|----------|---------|
| `topk: [1]` | DISCO `config.yaml` evaluator | Beam/sample dimension — how many full prediction sequences to generate and union for recall/precision/hit. Oracle@K style. |
| `sm_topk: 1` | DISCO `config.yaml` evaluator | Per-position retrieval K for SM@K. |
| `TOPK=1` | DreamRec `train_yelp.sh` | Same role as `sm_topk` for DreamRec; passed as `--topk`. |

**DISCO generation is stochastic** (Gumbel-max trick in `_sample_categorical`):
```python
gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
return (categorical_probs / gumbel_norm).argmax(dim=-1)
```
Equivalent to multinomial sampling — `topk > 1` produces genuinely diverse beams.

---

## 12. Known Issues

- `convert_ddbc_to_dreamrec.py` line 20: hardcodes `DDBC_f-main/datasets/Yelp` — update to `DISCO/datasets/Yelp` before next use
- `TIGER/` internal scripts may reference `gr-20251202/` — check when integrating evaluation
- Do **not** create markdown summary documents after code modifications; explain changes directly in conversation

## 13. GRU4Rec Data Files

- `GRU4Rec/data/yelp/train_yelp.tsv` — generated from `DISCO/datasets/Yelp/train.txt` (697,740 rows, 69,774 sessions)
- Eval data reused from `DreamRec/data/yelp/{valid,test}_data_items{n}.df` — no separate copy needed
- Valid candidates auto-built and cached at `GRU4Rec/data/yelp/valid_candidates_seed100_x{mult}_items{n}.pkl` on first eval run
- Test candidates loaded directly from `DISCO/datasets/Yelp/test_candidates_seed1_x{mult}_items{n}.pkl`

---

## 14. Data Generation Policy — Single Source of Truth

**All baseline training data must be derived from `DISCO/datasets/Yelp/{train,valid,test}.txt`.**

- Each line in these files is treated as one "user" (bundle) with a 10-item history
- `train.txt`: 69,774 lines → 69,774 training users/bundles
- `valid.txt`: 9,967 lines → 9,967 validation samples
- `test.txt`: 19,937 lines → 19,937 test samples

**Conversion script** (run once per dataset, or after any change to DISCO's txt files):
```bash
conda run -n DDBC python DISCO/datasets/Yelp/convert_disco_to_sasrec.py
```

This generates:
| File | Model | Notes |
|------|-------|-------|
| `SASRec/python/data/Yelp.txt` | SASRec | `user_id item_id` (1-based), 69,774 users × 10 items |
| `TIGER/cache/Yelp/Yelp_2020/processed/all_item_seqs.json` | TIGER | dict[bundle_N → [biz_id, ...]], 69,774 entries |
| `TIGER/cache/Yelp/Yelp_2020/processed/id_mapping.json` | TIGER | user2id updated; item2id unchanged |
| `LETTER/data/Yelp/Yelp.inter.json` | LETTER | dict[str(N) → [item_0based, ...]], 69,774 entries |

**For new datasets**: follow the same pattern — generate DISCO's txt files first, then run (or extend) `convert_disco_to_sasrec.py` for the new dataset.

**Evaluation data** (valid/test) is NOT regenerated by this script — all baselines load eval data from:
- `DreamRec/data/yelp/{valid,test}_data_items{n}.df` (derived from DISCO's valid/test.txt via `convert_ddbc_to_dreamrec.py`)
- Test candidates: `DISCO/datasets/Yelp/test_candidates_seed1_x{mult}_items{n}.pkl` (shared, never rebuilt)
- Valid candidates: auto-built per model on first eval run (seed=100)

---

## 15. MovieLens-60 (ML-60) Dataset

**Source**: `DISCO/datasets/MovieLens-20M/len60/`

| Property | Value |
|----------|-------|
| Train bundles | 52,857 |
| Valid bundles | 7,551 |
| Test bundles | 15,103 |
| Item count | 17,188 (0-based: 0..17187) |
| Seq size | 60 items/bundle |
| PAD token | 17,188 |
| DISCO predict_num_items | 30 |
| RVQ config | 3 codebooks × 256 entries |

**DISCO, TIGER, LETTER** already have ML-60 data with consistent Semantic IDs.

### ML-60 Baseline Data Files

| Model | Data Path | Generated By |
|-------|-----------|--------------|
| DreamRec | `DreamRec/data/ml60/{data_statis,train_data,valid_data_items30,test_data_items30}.df` | `DISCO/datasets/MovieLens-20M/len60/convert_disco_to_dreamrec.py` |
| DiffuRec | `DiffuRec/data/ml60/dataset.pkl` | `DISCO/datasets/MovieLens-20M/len60/convert_disco_to_diffurec.py` |
| BERT4Rec | `BERT4Rec/Data/preprocessed/ml60_min_rating0-.../dataset.pkl` | `DISCO/datasets/MovieLens-20M/len60/convert_disco_to_bert4rec.py` |
| SASRec | `SASRec/python/data/MovieLens60.txt` | `DISCO/datasets/MovieLens-20M/len60/convert_disco_to_sasrec.py` |
| GRU4Rec | `GRU4Rec/data/ml60/train_ml60.tsv` | `GRU4Rec/train_movielens60.py` (inline, first run) |
| TIGER | `TIGER/cache/MovieLens-20M/len60/processed/` | Already exists |
| LETTER | `LETTER/data/MovieLens-20M/len60/` | Already exists |

### ML-60 Conversion (One-Time)

```bash
conda run -n DDBC python DISCO/datasets/MovieLens-20M/len60/convert_disco_to_dreamrec.py
conda run -n DDBC python DISCO/datasets/MovieLens-20M/len60/convert_disco_to_diffurec.py
conda run -n DDBC python DISCO/datasets/MovieLens-20M/len60/convert_disco_to_bert4rec.py
conda run -n DDBC python DISCO/datasets/MovieLens-20M/len60/convert_disco_to_sasrec.py
```

### ML-60 Evaluation Config

- **Predict count**: predict_n = 30 (matching DISCO eval.predict_num_items=30)
- **Candidate multiplier**: 19
- **Best checkpoint selection**: `val_recall@30_x19`
- **Test candidates**: `DISCO/datasets/MovieLens-20M/len60/test_candidates_seed1_x19_items30.pkl` (shared, generated by DISCO evaluator)
- **Valid candidates**: auto-built per model on first eval run (seed=100, multiplier=19)
- **Training script**: `bash train_all_ml60.sh [model_list]`
