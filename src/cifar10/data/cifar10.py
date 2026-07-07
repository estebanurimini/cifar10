"""CIFAR10 dataset transforms and dataloader factory."""

from pathlib import Path

import torch
from torch import device as TorchDevice
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode, AutoAugmentPolicy

from cifar10.data.augmentations import Cutout

# CIFAR10 normalization constants (computed from the training set)
CIFAR10_NORM = {
    "mean": (0.4914, 0.4822, 0.4465),
    "std": (0.2023, 0.1994, 0.2010),
}

# ImageNet normalization constants (required for pretrained models)
IMAGENET_NORM = {
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
}


class _TransformedSubset(Subset):
    """A Subset that applies a different transform than the original dataset.

    The parent ``torchvision`` dataset (e.g. ``CIFAR10``) is created **without**
    transforms. This wrapper slices to the desired indices and applies the
    correct transform on-the-fly.  This ensures train and validation subsets are
    disjoint *and* can use different transforms.
    """

    def __init__(
        self,
        dataset: datasets.CIFAR10,
        indices: list[int],
        transform: transforms.Compose,
    ) -> None:
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img, label = self.dataset[self.indices[idx]]   # type: ignore[index,arg-type]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __getitems__(self, indices: list[int]) -> list[tuple[torch.Tensor, int]]:
        return [self.__getitem__(idx) for idx in indices]


def build_cifar10_loaders(
    data_dir: str | Path,
    batch_size: int = 128,
    num_workers: int = 0,
    device: TorchDevice | None = None,
    with_validation_split: bool = False,
    validation_pct: float = 0.1,
    use_randaugment: bool = False,
    download: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """Build train / validation / test DataLoaders for CIFAR10.

    Args:
        data_dir: Root directory for the CIFAR10 dataset.
        batch_size: Batch size for all loaders.
        num_workers: Number of DataLoader workers.
        device: If provided, determines ``pin_memory`` (True for CUDA).
        with_validation_split: If True, split a validation set from the 50k
            training samples.
        validation_pct: Fraction of training data to use for validation.
        use_randaugment: If True, apply RandAugment to training transforms.
        download: If True, download the dataset.

    Returns:
        A tuple ``(train_loader, val_loader, test_loader)``.
        If ``with_validation_split`` is False, ``val_loader`` will be the test
        loader and ``test_loader`` will be ``None``.

    """
    pin_memory = device is not None and device.type == "cuda"

    # --- Transforms -----------------------------------------------------------
    # When use_randaugment is True, use the full CIFAR-10 recipe:
    #   AutoAugment(CIFAR10) + Cutout(hole_size=16) + MixUp/CutMix at batch level
    train_tfms = [
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    if use_randaugment:
        train_tfms.append(transforms.AutoAugment(policy=AutoAugmentPolicy.CIFAR10))
    train_tfms.extend([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_NORM["mean"], CIFAR10_NORM["std"]),
        Cutout(hole_size=16),
    ])

    eval_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_NORM["mean"], CIFAR10_NORM["std"]),
    ])

    train_transform = transforms.Compose(train_tfms)

    # --- Datasets -------------------------------------------------------------
    test_dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=download,
        transform=eval_tfms,
    )

    train_dataset: datasets.CIFAR10 | Subset
    val_dataset: datasets.CIFAR10 | Subset | None = None

    if with_validation_split:
        # Single base dataset (no transforms) → deterministic split → disjoint
        # subsets, each with its own transform applied by _TransformedSubset.
        base_train = datasets.CIFAR10(
            root=str(data_dir),
            train=True,
            download=download,
            transform=None,
        )
        val_size = int(validation_pct * len(base_train))
        train_size = len(base_train) - val_size

        # Split the actual dataset (no transforms) → disjoint Subset objects
        train_subset, val_subset = random_split(
            base_train,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        train_dataset = _TransformedSubset(base_train, train_subset.indices, train_transform)
        val_dataset = _TransformedSubset(base_train, val_subset.indices, eval_tfms)

    else:
        train_dataset = datasets.CIFAR10(
            root=str(data_dir),
            train=True,
            download=download,
            transform=train_transform,
        )

    # --- DataLoaders ----------------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader_data = val_dataset if val_dataset is not None else test_dataset
    val_loader = DataLoader(
        val_loader_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory if val_dataset is None else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if with_validation_split:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, test_loader, None


def build_cifar10_imagenet_loaders(
    data_dir: str | Path,
    image_size: int = 128,
    batch_size: int = 128,
    num_workers: int = 0,
    device: TorchDevice | None = None,
    validation_ratio: float = 0.2,
    download: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / validation / test DataLoaders for CIFAR10 with ImageNet preprocessing.

    Designed for transfer learning with ImageNet-pretrained models (e.g.
    ConvNeXt). Images are upsampled to ``image_size`` using bicubic
    interpolation and normalized with ImageNet stats.

    A validation split is carved from the training set using
    ``validation_ratio``. The validation split uses non-augmented evaluation
    transforms (no cropping, no TrivialAugmentWide, no RandomErasing).

    Augmentation pipeline (matching TorchVision's ConvNeXt recipe):
        - RandomResizedCrop (bicubic)
        - RandomHorizontalFlip
        - TrivialAugmentWide (``ta_wide``)
        - ToTensor
        - RandomErasing (p=0.1)
        - Normalize (ImageNet stats)

    Args:
        data_dir: Root directory for the CIFAR10 dataset.
        image_size: Target spatial size (default 128).
        batch_size: Batch size for all loaders.
        num_workers: Number of DataLoader workers.
        device: If provided, determines ``pin_memory`` (True for CUDA).
        validation_ratio: Fraction of training data to hold out for validation
            (default 0.2 = 10k out of 50k).
        download: If True, download the dataset.

    Returns:
        A tuple ``(train_loader, val_loader, test_loader)``.
    """
    pin_memory = device is not None and device.type == "cuda"
    bicubic = InterpolationMode.BICUBIC

    # --- Training transforms (ImageNet-style) ---------------------------------
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.8, 1.0),
            interpolation=bicubic,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.1),
        transforms.Normalize(IMAGENET_NORM["mean"], IMAGENET_NORM["std"]),
    ])

    # --- Evaluation transforms (ImageNet-style) -------------------------------
    eval_transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=bicubic),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_NORM["mean"], IMAGENET_NORM["std"]),
    ])

    # --- Datasets -------------------------------------------------------------
    # Base training set without transforms → deterministic split into
    # train/val subsets, each with its own transforms applied by
    # _TransformedSubset. The validation set gets no augmentations.
    base_train = datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=download,
        transform=None,
    )

    val_size = int(validation_ratio * len(base_train))
    train_size = len(base_train) - val_size

    train_subset, val_subset = random_split(
        base_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_dataset = _TransformedSubset(base_train, train_subset.indices, train_transform)
    val_dataset = _TransformedSubset(base_train, val_subset.indices, eval_transform)

    test_dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=download,
        transform=eval_transform,
    )

    # --- DataLoaders ----------------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
