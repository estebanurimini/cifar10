# CIFAR-10 Classification

This project implements several model 
architectures for **CIFAR-10** classification 
(10-class image recognition on 32×32 color images) using PyTorch, with a modular framework.

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
│   ├── wideresnet.py          # WideResNet model (architecture only)
│   ├── vgg.py                 # VGG11-BN + VGG16-BN for CIFAR10 (architecture only)
│   ├── resnet_cifar.py        # ResNet20 + ResNet56 for CIFAR10 (architecture only)
│   └── convnext.py            # ConvNeXt Tiny with ImageNet pretrained weights (architecture only)
└── scripts/
    ├── train_vit.py           # ViT training script (CLI with --resume)
    ├── train_deit.py          # DeiT distillation training script (CLI with --resume)
    ├── train_wrn.py           # WideResNet training script (CLI with --resume)
    ├── train_vgg.py           # VGG training script (CLI with --variant and --resume)
    ├── train_resnet.py        # ResNet training script (CLI with --variant and --resume)
    ├── train_convnext.py      # ConvNeXt transfer learning script (CLI with --resume, --image-size, --batch-size)
    └── evaluate_model.py      # Standalone evaluation CLI for trained models
```

## Currently Implemented Architectures

| Model | Architecture File | Training Script | Variants | Key Features |
|---|---|---|---|---|
| **VGG** | `models/vgg.py` | `scripts/train_vgg.py` | VGG11-BN, VGG16-BN | CIFAR10-adapted (3 max-pools instead of 5), SGD + Cosine, RandAugment |
| **ResNet** | `models/resnet_cifar.py` | `scripts/train_resnet.py` | ResNet20, ResNet56 | CIFAR10 variant (3×3 conv, no max-pool), SGD + Warmup + Cosine, MixUp/CutMix, RandAugment |
| **ViT** | `models/vit.py` | `scripts/train_vit.py` | — | Vision Transformer, MixUp + CutMix, AdamW + Warmup + Cosine |
| **DeiT** | `models/deit.py` | `scripts/train_deit.py` | — | Convolutional stem, distillation token, stochastic depth, hard distillation from a WRN teacher |
| **WideResNet** | `models/wideresnet.py` | `scripts/train_wrn.py` | WRN-28-10 | Pre-activation blocks, SGD + Cosine, RandAugment |
| **ConvNeXt** | `models/convnext.py` | `scripts/train_convnext.py` | ConvNeXt Tiny | ImageNet-pretrained transfer learning, bicubic upscale to 128×128, AdamW + Cosine, TrivialAugmentWide + RandomErasing, label smoothing, MixUp + CutMix, EMA, gradient clipping, staged fine-tuning (10 epochs head-only → full fine-tune) |

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
# VGG training (default: VGG16-BN)
python -m cifar10.scripts.train_vgg
python -m cifar10.scripts.train_vgg --variant vgg11_bn

# ResNet training (default: ResNet56)
python -m cifar10.scripts.train_resnet
python -m cifar10.scripts.train_resnet --variant resnet20

# ViT training
python -m cifar10.scripts.train_vit

# WideResNet training (generates teacher for DeiT)
python -m cifar10.scripts.train_wrn

# DeiT distillation training (requires pre-trained WRN teacher)
python -m cifar10.scripts.train_deit

# ConvNeXt transfer learning (uses ImageNet-pretrained Tiny variant)
python -m cifar10.scripts.train_convnext
python -m cifar10.scripts.train_convnext --image-size 128 --batch-size 128
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
├── wrn/
│   ├── checkpoints/
│   └── logs/
├── vgg/
│   ├── checkpoints/
│   └── logs/
├── resnet/
│   ├── checkpoints/
│   └── logs/
└── convnext/
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

# Resume VGG
python -m cifar10.scripts.train_vgg --resume .runs/vgg/checkpoints/last.pt

# Resume ResNet
python -m cifar10.scripts.train_resnet --resume .runs/resnet/checkpoints/last.pt

# Resume ConvNeXt
python -m cifar10.scripts.train_convnext --resume .runs/convnext/checkpoints/last.pt
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
| `--model` | (required) | Architecture: `vit`, `wrn`, `deit`, `vgg`, `resnet`, or `convnext` |
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
3. **Register in evaluation** by adding your model to `scripts/evaluate_model.py` MODEL_REGISTRY.

## Device Support

All scripts automatically detect and use:
- **MPS** (Apple Silicon)
- **CUDA** (NVIDIA GPU)
- **CPU** (fallback)