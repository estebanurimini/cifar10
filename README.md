# CIFAR-10 Classification

This project implements several model architectures for **CIFAR-10** classification (10-class image recognition on 32×32 color images) using PyTorch, with a modular framework.

## Project Structure

```
src/cifar10/
├── config.py                 # BaseConfig — shared defaults for all training runs
├── utils/                    # Seed, device, EMA, checkpoint helpers
├── data/                     # CIFAR10 loaders + augmentations (MixUp/CutMix)
├── training/                 # BaseTrainer, evaluate, scheduler, resolver, registry
├── models/
│   ├── blocks.py             # Shared transformer blocks (MLP, Attention, etc.)
│   ├── own/                  # From-scratch implementations (one sub-package per architecture)
│   │   ├── vgg/
│   │   ├── resnet/
│   │   ├── wrn/
│   │   ├── vit/
│   │   ├── deit/
│   │   └── efficientnet/
│   ├── tv/                   # TorchVision wrappers with pretrained weights
│   │   ├── convnext/
│   │   └── efficientnet_v2/
│   └── timm/                 # TIMM wrappers (optional dependency)
│       └── resnet/
├── results/                  # Run discovery & filtering API
└── scripts/                  # train.py, evaluate_model.py, migrate_runs.py
```

Each model sub-package (e.g., `models/own/vit/`) is self-contained with:
- `model.py` — the `nn.Module` forward pass
- `config.py` — model-specific hyperparameter defaults (subclass of `TrainerConfig`)
- `trainer.py` — training logic (subclass of `StandardTrainer` or `BaseTrainer`)
- `__init__.py` — re-exports the above

## Currently Implemented Architectures

| Model | Source | Params (M) | Input Size | Pretrained | Best Acc | Variants | Key Features |
|---|---|---|---|---|---|---|---|
| **VGG** | `own` | 19.18M | 32×32 (native) | No (scratch) | 96.38% | VGG11-BN, VGG16-BN | CIFAR10-adapted (3 max-pools instead of 5), SGD + Cosine, RandAugment |
| **ResNet** | `own` | 0.86M | 32×32 (native) | No (scratch) | — | ResNet20, ResNet56 | CIFAR10 variant (3×3 conv, no max-pool), SGD + Warmup + Cosine, MixUp/CutMix, RandAugment |
| **ViT** | `own` | 2.69M | 32×32 (native) | No (scratch) | 89.55% | — | Vision Transformer, MixUp + CutMix, AdamW + Warmup + Cosine |
| **DeiT** | `own` | 2.86M | 32×32 (native) | No (scratch) | 93.15% | — | Convolutional stem, distillation token, stochastic depth, hard distillation from a WRN teacher |
| **WideResNet** | `own` | 2.75M | 32×32 (native) | No (scratch) | 96.59% | WRN-28-10 | Pre-activation blocks, SGD + Cosine, RandAugment |
| **EfficientNet-V2** | `own` | 3.83M | 32×32 (native) | No (scratch) | — | — | MBConv + Fused-MBConv blocks, AdamW + Cosine + Warmup, MixUp/CutMix, EMA, stochastic depth, gradient clipping |
| **ConvNeXt** | `tv` | 27.83M | 128×128 (upsampled) | Yes (ImageNet-1K) | **99.03%** | ConvNeXt Tiny | ImageNet-pretrained transfer learning, bicubic upscale to 128×128, AdamW + Cosine, TrivialAugmentWide + RandomErasing, label smoothing, MixUp + CutMix, EMA, gradient clipping, staged fine-tuning (10 epochs head-only → full fine-tune) |
| **EfficientNet-V2** | `tv` | 20.19M | 128×128 (upsampled) | Yes (ImageNet-1K) | 98.68% | S, M, L | ImageNet-pretrained transfer learning, bicubic upscale to 128×128, AdamW + Cosine + Warmup, TrivialAugmentWide + RandomErasing, label smoothing, MixUp + CutMix, EMA, gradient clipping, staged fine-tuning (16 epochs head-only → full fine-tune) |
| **ResNet** | `timm` | 11.18M | 224×224 (upsampled) | Yes (ImageNet-1K) | 97.00% | ResNet18, 34, 50, 101, 152 | ImageNet-pretrained transfer learning, bicubic upscale to 224×224, AdamW + Cosine + Warmup, TrivialAugmentWide + RandomErasing, label smoothing, MixUp + CutMix, EMA, gradient clipping, staged fine-tuning |

**Note:** Parameter counts are computed by instantiating each model architecture with default configuration. Transfer learning models use ImageNet-1K pretrained weights with upsampled inputs to match the pretrained model's native resolution.

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

All models are trained via a single unified script using the `--model` flag with a `{source}-{architecture}` naming convention:

```bash
# List all available models
python -m cifar10.scripts.train --list-models

# VGG training (default: VGG16-BN)
python -m cifar10.scripts.train --model own-vgg
python -m cifar10.scripts.train --model own-vgg --variant vgg11_bn

# ResNet training (default: ResNet56)
python -m cifar10.scripts.train --model own-resnet
python -m cifar10.scripts.train --model own-resnet --variant resnet20

# ViT training
python -m cifar10.scripts.train --model own-vit

# WideResNet training (generates teacher for DeiT)
python -m cifar10.scripts.train --model own-wrn

# DeiT distillation training (requires pre-trained WRN teacher)
python -m cifar10.scripts.train --model own-deit

# ConvNeXt transfer learning (uses ImageNet-pretrained Tiny variant)
python -m cifar10.scripts.train --model tv-convnext
python -m cifar10.scripts.train --model tv-convnext --image-size 128 --batch-size 128

# OwnEfficientNetV2 from scratch
python -m cifar10.scripts.train --model own-efficientnetv2
python -m cifar10.scripts.train --model own-efficientnetv2 --lr 3e-3 --epochs 300

# TVEfficientNetV2 transfer learning (default: S variant)
python -m cifar10.scripts.train --model tv-efficientnetv2
python -m cifar10.scripts.train --model tv-efficientnetv2 --variant m
python -m cifar10.scripts.train --model tv-efficientnetv2 --variant l --image-size 128 --batch-size 64

# TIMM ResNet transfer learning (default: ResNet50)
python -m cifar10.scripts.train --model timm-resnet
python -m cifar10.scripts.train --model timm-resnet --model-name resnet101
```

### Common CLI Flags

| Flag | Description |
|---|---|
| `--model` | Model identifier (e.g. `own-vit`, `tv-convnext`). Required unless `--list-models` is used. |
| `--list-models` | List all available models and exit. |
| `--variant` | Model variant (e.g. `resnet56`, `vgg16_bn`, `s`, `m`, `l`). |
| `--model-name` | TIMM model name (e.g. `resnet50`, `resnet101`). Used only with `--model timm-resnet`. |
| `--image-size` | Input image size (for models that support it, e.g. transfer learning models). |
| `--batch-size` | Batch size (overrides config default). |
| `--lr` | Learning rate (overrides config default). |
| `--epochs` | Number of epochs (overrides config default). |
| `--resume` | Path to a checkpoint to resume training from. |

### Output Layout

All outputs are written to hidden folders at the project root:

```
.runs/                        # Training outputs (one subfolder per model)
├── own_vit/
├── own_deit/
├── own_wrn/
├── own_vgg/
├── own_resnet/
├── own_efficientnetv2/
├── tv_convnext/
├── tv_efficientnetv2/
└── timm_resnet/

.data/                        # CIFAR10 dataset cache
```

Each run folder contains `checkpoints/` (best.pt, last.pt) and `logs/` (metrics.csv).

### Training Order (for DeiT)

DeiT requires a pre-trained **WideResNet teacher**. Run in this order:

```bash
# 1. Train the teacher
python -m cifar10.scripts.train --model own-wrn

# 2. Train the student with distillation
python -m cifar10.scripts.train --model own-deit
```

### Resuming Training

All training scripts accept a `--resume` flag to continue from a saved checkpoint:

```bash
# Resume ViT from its last checkpoint
python -m cifar10.scripts.train --model own-vit --resume .runs/own_vit/checkpoints/last.pt

# Resume WideResNet
python -m cifar10.scripts.train --model own-wrn --resume .runs/own_wrn/checkpoints/last.pt

# Resume DeiT
python -m cifar10.scripts.train --model own-deit --resume .runs/own_deit/checkpoints/last.pt

# Resume VGG
python -m cifar10.scripts.train --model own-vgg --resume .runs/own_vgg/checkpoints/last.pt

# Resume ResNet
python -m cifar10.scripts.train --model own-resnet --resume .runs/own_resnet/checkpoints/last.pt

# Resume OwnEfficientNetV2
python -m cifar10.scripts.train --model own-efficientnetv2 --resume .runs/own_efficientnetv2/checkpoints/last.pt

# Resume ConvNeXt
python -m cifar10.scripts.train --model tv-convnext --resume .runs/tv_convnext/checkpoints/last.pt

# Resume TVEfficientNetV2
python -m cifar10.scripts.train --model tv-efficientnetv2 --resume .runs/tv_efficientnetv2/checkpoints/last.pt

# Resume TIMM ResNet
python -m cifar10.scripts.train --model timm-resnet --resume .runs/timm_resnet/checkpoints/last.pt
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
    --model own_vit \
    --checkpoint .runs/own_vit/checkpoints/best.pt
```

> **Note:** The evaluation script uses underscores in model names (e.g. `own_vit`), while the training script uses hyphens (e.g. `own-vit`).

Output includes:
- Test loss and overall accuracy
- Per-class accuracy breakdown
- Inference throughput (images/second)

New checkpoints (saved after this feature was added) embed the full model configuration, so the evaluation script reconstructs the model with the exact hyperparameters used during training. Legacy checkpoints are handled transparently by falling back to default configs.

Optional arguments:

| Flag | Default | Description |
|---|---|---|
| `--model` | (required) | Architecture: `own_vit`, `own_wrn`, `own_deit`, `own_vgg`, `own_resnet`, `own_efficientnetv2`, `tv_convnext`, `tv_efficientnetv2`, or `timm_resnet` |
| `--checkpoint` | (required) | Path to the checkpoint file |
| `--batch-size` | 128 | Batch size for evaluation |
| `--data-dir` | `./.data` | CIFAR-10 dataset root |
| `--num-workers` | 0 | DataLoader workers |
| `--no-detailed` | (flag) | Skip per-class accuracy breakdown |

## Extending with a New Model

To add a new architecture:

1. **Create a model sub-package** under `models/own/<name>/` (or `models/tv/<name>/` for torchvision-based models) with:
   - `model.py` — the `nn.Module` forward pass
   - `config.py` — subclass `TrainerConfig` with model-specific hyperparameter defaults
   - `trainer.py` — subclass `StandardTrainer` (or `BaseTrainer` for custom loss/optimizer/scheduler logic)
   - `__init__.py` — re-export the above
2. **Register the model** in `training/registry.py` — add an entry to `MODEL_REGISTRY` mapping your model identifier (e.g. `"own-yourmodel"`) to its config, model, trainer, and data loader builder.
3. **Register in evaluation** by adding your model to `scripts/evaluate_model.py` `MODEL_REGISTRY` and `_MODEL_CLASSES`.

## Device Support

All scripts automatically detect and use:
- **MPS** (Apple Silicon)
- **CUDA** (NVIDIA GPU)
- **CPU** (fallback)