# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a generative recommendation research codebase (GenRec) that implements the TIGER model from "Recommender Systems with Generative Retrieval" (Rajput et al., NeurIPS 2023). The framework supports training and evaluating sequential recommendation models using transformer-based generative retrieval.

## Training Commands

### Basic Training
```bash
python main.py --model=TIGER --dataset=<dataset_name>
```

### Multi-GPU Training with Accelerate
The codebase uses Hugging Face Accelerate for distributed training. All training scripts use `accelerate launch`:

```bash
# Yelp dataset
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 12347 main.py --model=TIGER --dataset=Yelp --category=Yelp

# Amazon Reviews 2023
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 12345 main.py --model=TIGER --dataset=AmazonReviews2023 --category=Industrial_and_Scientific

# Steam dataset
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 12346 main.py --model=TIGER --dataset=Steam --category=Steam
```

### Custom Configuration
Override configuration via command-line arguments using `--key=value` format:
```bash
python main.py --model=TIGER --dataset=Yelp --lr=0.001 --epochs=100 --train_batch_size=128
```

## Architecture

### Core Pipeline Flow
1. **Entry Point**: `main.py` parses model/dataset args and delegates to model-specific pipeline
2. **Pipeline** (`genrec/pipeline.py`): Orchestrates the full training workflow
   - Loads configuration (default → dataset → model → command-line overrides)
   - Initializes dataset, tokenizer, model, and trainer
   - Creates dataloaders and runs training/evaluation loop
3. **Training** (`genrec/trainer.py`): Handles training loop, evaluation, checkpointing, early stopping
4. **Model**: TIGER uses T5-based encoder-decoder with custom beam search for item generation

### Configuration System
Configuration loading follows a hierarchical override pattern:
- `genrec/default.yaml` (base config)
- `genrec/datasets/{dataset}/config.yaml` (dataset-specific)
- `genrec/models/{model}/config.yaml` (model-specific)
- Command-line args via `--key=value` (highest priority)

Implemented in `get_config()` in `genrec/utils2.py` (lines 540-617).

### Dynamic Module Loading
The framework uses reflection-based module loading for extensibility:
- `get_model()`: Loads model classes from `genrec.models` by name
- `get_dataset()`: Loads dataset classes from `genrec.datasets` by name
- `get_tokenizer()`: Loads tokenizer from `genrec.models.{model_name}.tokenizer`
- `get_trainer()`: Loads model-specific trainer or falls back to base `Trainer`
- `get_pipeline()`: Loads upstream pipeline or falls back to base `Pipeline`

All implemented in `genrec/utils2.py`.

### Model Structure (TIGER)
- **Representation**: Items are encoded as semantic embeddings (sentence-t5-base) then quantized using Residual Quantization (RQ) with 3 codebooks of 256 entries each
- **Architecture**: T5-based encoder-decoder where:
  - Encoder: Takes user history sequence (user token + item sequence)
  - Decoder: Generates target item's quantized code sequence
- **Generation**: Custom beam search implementation (`genrec/models/TIGER/model.py:101-268`) that generates fixed-length item codes without EOS-based stopping

### Abstract Base Classes
- `AbstractDataset` (`genrec/dataset.py`): Defines `split()` method for train/val/test
- `AbstractModel` (`genrec/model.py`): Base for all models, requires `forward()` and `generate()`
- `AbstractTokenizer` (`genrec/tokenizer.py`): Handles data tokenization with `tokenize()` and `collate_fn`

### Supported Datasets
Located in `genrec/datasets/`:
- `AmazonReviews2014`
- `AmazonReviews2023`
- `Yelp`
- `Steam`

Each dataset has its own `dataset.py` and `config.yaml`.

## Key Components

### Distributed Training
- Uses Hugging Face Accelerate for DDP (Data Distributed Parallel)
- Device initialization in `init_device()` checks `WORLD_SIZE` environment variable to detect DDP mode
- Trainer handles model wrapping/unwrapping for checkpointing and evaluation gathering

### Logging and Checkpointing
- Logs saved to `logs/{dataset}/{model}/` with auto-generated filenames based on config hash
- TensorBoard logs in `tensorboard/{dataset}/{model}/`
- Model checkpoints in `ckpt/` directory
- Filename generation in `get_file_name()` (genrec/utils2.py:77-99) includes run ID, command-line args, timestamp, and config hash

### Evaluation
- Evaluator (`genrec/evaluator.py`) computes ranking metrics (NDCG, Recall)
- Configurable via `topk` (default: [5, 10]) and `metrics` (default: [ndcg, recall])
- Validation metric for model selection: `val_metric` (default: `ndcg@10`)
- Early stopping with configurable patience (default: 25 epochs)

### Training Hyperparameters (from default.yaml)
- Learning rate: 0.0003 (TIGER overrides to 0.003)
- Weight decay: 0.0 (TIGER overrides to 0.05)
- Warmup steps: 10000
- Max grad norm: 1.0
- Eval interval: 5 epochs
- Optimizer: AdamW with cosine learning rate schedule

## Development Notes

### Adding New Models
1. Create directory `genrec/models/{ModelName}/`
2. Implement model in `model.py` inheriting from `AbstractModel`
3. Implement tokenizer in `tokenizer.py` inheriting from `AbstractTokenizer`
4. Add model config in `config.yaml`
5. Optionally implement custom trainer in `trainer.py` (inherits from `Trainer`)
6. Register in `genrec/models/__init__.py`

### Adding New Datasets
1. Create directory `genrec/datasets/{DatasetName}/`
2. Implement dataset in `dataset.py` inheriting from `AbstractDataset`
3. Add dataset config in `config.yaml`
4. Register in `genrec/datasets/__init__.py`

### Command-Line Argument Parsing
The framework uses a two-stage parsing approach:
1. `main.py` uses argparse for `--model` and `--dataset`
2. Remaining args parsed by `parse_command_line_args()` which expects `--key=value` format and attempts to eval values for type inference
