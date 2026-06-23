# CIFAR-10 Classification

This project implements various model architectures for **CIFAR-10** classification (10-class image recognition on 32×32 color images) using PyTorch, with a modular, SOLID-compliant training framework.

## Project Structure

```
src/cifar10/
├── config.py                  # BaseConfig — shared defaults for all training runs
├── utils/
│   ├── seed.py                # Reproducibility (random seed)
│   ├── device.py              # Device detection (MPS / CUDA / CPU)
│   ├── ema.py                 # Exponential Moving Average (EMA)
│   └── checkpoint.py          # Save / load checkpoint helpers
├── data/
│   ├── cifar10.py             # CIFAR10 transforms + dataloader factory
│   └── augmentations.py       # MixUp / CutMix augmentation
├── training/
│   ├── evaluate.py            # Shared evaluation loop + detailed evaluation
│   ├── scheduler.py           # Warmup + Cosine annealing factory
│   └── trainer.py             # BaseTrainer (ABC) + StandardTrainer + resume support
├── models/
│   ├── blocks.py              # Shared transformer blocks (MLP, Attention, etc.)
│   ├── vit.py                 # ViT model (architecture only)
│   ├── deit.py                # DeiT model + ConvStemPatchEmbedding (architecture only)
│   └── wideresnet.py          # WideResNet model (architecture only)
└── scripts/
    ├── train_vit.py           # ViT training script (CLI with --resume)
    ├── train_deit.py          # DeiT distillation training script (CLI with --resume)
    ├── train_wrn.py           # WideResNet training script (CLI with --resume)
    └── evaluate_model.py      # Standalone evaluation CLI for trained models
```

## Currently Implemented Architectures

| Model | Architecture File | Training Script | Key Features |
|---|---|---|---|
| **ViT** | `models/vit.py` | `scripts/train_vit.py` | Vision Transformer, MixUp + CutMix, AdamW + Warmup + Cosine |
| **DeiT** | `models/deit.py` | `scripts/train_deit.py` | Convolutional stem, distillation token, stochastic depth, hard distillation from a WRN teacher |
| **WideResNet** | `models/wideresnet.py` | `scripts/train_wrn.py` | WRN-28-10 CNN baseline, SGD + Cosine, RandAugment |

## Setup

```bash
# Requires Python ≥ 3.11
pip install -e .
```

Or using `uv` (recommended):

```bash
uv pip install -e .
```

Dependencies: `torch`, `torchvision`, `tqdm`, `jupyter`.

## Usage

Each architecture has a dedicated training script. Run via module:

```bash
# ViT training
python -m cifar10.scripts.train_vit

# WideResNet training (generates teacher for DeiT)
python -m cifar10.scripts.train_wrn

# DeiT distillation training (requires pre-trained WRN teacher)
python -m cifar10.scripts.train_deit
```

### Output Layout

All outputs are written to hidden folders at the project root:

```
.runs/
├── vit/
│   ├── checkpoints/          # best.pt, last.pt
│   └── logs/                 # metrics.csv
├── deit/
│   ├── checkpoints/
│   └── logs/
└── wrn/
    ├── checkpoints/
    └── logs/

.data/                         # CIFAR10 dataset cache
```

### Training Order (for DeiT)

DeiT requires a pre-trained **WideResNet teacher**. Run in this order:

```bash
# 1. Train the teacher
python -m cifar10.scripts.train_wrn

# 2. Train the student with distillation
python -m cifar10.scripts.train_deit
```

### Resuming Training

All training scripts accept a `--resume` flag to continue from a saved checkpoint:

```bash
# Resume ViT from its last checkpoint
python -m cifar10.scripts.train_vit --resume .runs/vit/checkpoints/last.pt

# Resume WideResNet
python -m cifar10.scripts.train_wrn --resume .runs/wrn/checkpoints/last.pt

# Resume DeiT
python -m cifar10.scripts.train_deit --resume .runs/deit/checkpoints/last.pt
```

Resuming restores:
- Model weights (including EMA shadow weights)
- Optimizer state (momentum, adaptive learning rates)
- Learning rate scheduler state (warmup/cosine position)
- AMP gradient scaler

Logs are appended to the existing `metrics.csv` file, avoiding duplicates.

### Evaluating a Trained Model

The `evaluate_model.py` script loads a checkpoint and runs detailed evaluation on the CIFAR-10 test set:

```bash
python -m cifar10.scripts.evaluate_model \
    --model vit \
    --checkpoint .runs/vit/checkpoints/best.pt
```

Output includes:
- Test loss and overall accuracy
- Per-class accuracy breakdown
- Inference throughput (images/second)

New checkpoints (saved after this feature was added) embed the full model configuration, so the evaluation script reconstructs the model with the exact hyperparameters used during training. Legacy checkpoints are handled transparently by falling back to default configs.

Optional arguments:

| Flag | Default | Description |
|---|---|---|
| `--model` | (required) | Architecture: `vit`, `wrn`, or `deit` |
| `--checkpoint` | (required) | Path to the checkpoint file |
| `--batch-size` | 128 | Batch size for evaluation |
| `--data-dir` | `./.data` | CIFAR-10 dataset root |
| `--num-workers` | 0 | DataLoader workers |
| `--no-detailed` | (flag) | Skip per-class accuracy breakdown |

## Extending with a New Model

To add a new architecture:

1. **Create the model** in `models/your_model.py` — just the `nn.Module` forward pass, no training code.
2. **Create a config** and **trainer** in `scripts/train_your_model.py`:
   - Subclass `TrainerConfig` for hyperparameters
   - Subclass `StandardTrainer` (or `BaseTrainer` for custom loss)
   - Override `_compute_loss()` if needed
   - Override `_build_optimizer()` / `_build_scheduler()` for different optimizers

## Device Support

All scripts automatically detect and use:
- **MPS** (Apple Silicon)
- **CUDA** (NVIDIA GPU)
- **CPU** (fallback)

