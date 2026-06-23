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
│   ├── evaluate.py            # Shared evaluation loop
│   ├── scheduler.py           # Warmup + Cosine annealing factory
│   └── trainer.py             # BaseTrainer (ABC) + StandardTrainer
├── models/
│   ├── blocks.py              # Shared transformer blocks (MLP, Attention, etc.)
│   ├── vit.py                 # ViT model (architecture only)
│   ├── deit.py                # DeiT model + ConvStemPatchEmbedding (architecture only)
│   └── wideresnet.py          # WideResNet model (architecture only)
└── scripts/
    ├── train_vit.py           # ViT training script
    ├── train_deit.py          # DeiT distillation training script
    └── train_wrn.py           # WideResNet training script
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
├── vit_cifar10/
│   ├── checkpoints/          # best.pt, last.pt
│   └── logs/                 # metrics.csv
├── deit_cifar10/
│   ├── checkpoints/
│   └── logs/
└── wrn_cifar10/
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

## Backward Compatibility

The original monolithic standalone scripts are preserved and still work:

```bash
python src/cifar10/models/vit_cifar10.py
python src/cifar10/models/wideresnet_cifar10.py
python src/cifar10/models/deit_cifar10.py