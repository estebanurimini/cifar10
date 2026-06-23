"""CIFAR10 dataset transforms and dataloader factory."""

from pathlib import Path
from torch import device as TorchDevice
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# CIFAR10 normalisation constants (computed from the training set)
CIFAR10_NORM = {
    "mean": (0.4914, 0.4822, 0.4465),
    "std": (0.2023, 0.1994, 0.2010),
}


def build_cifar10_loaders(
    data_dir: str | Path,
    batch_size: int = 128,
    num_workers: int = 0,
    device: TorchDevice | None = None,
    with_validation_split: bool = False,
    validation_pct: float = 0.1,
    use_randaugment: bool = False,
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

    Returns:
        A tuple ``(train_loader, val_loader, test_loader)``.
        If ``with_validation_split`` is False, ``val_loader`` will be the test
        loader and ``test_loader`` will be ``None``.
    """
    pin_memory = device is not None and device.type == "cuda"

    # --- Transforms -----------------------------------------------------------
    train_tfms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if use_randaugment:
        train_tfms.append(transforms.RandAugment(num_ops=2, magnitude=9))
    train_tfms.extend([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_NORM["mean"], CIFAR10_NORM["std"]),
    ])

    eval_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_NORM["mean"], CIFAR10_NORM["std"]),
    ])

    train_transform = transforms.Compose(train_tfms)

    # --- Datasets -------------------------------------------------------------
    train_dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=True,
        transform=eval_tfms,
    )

    # --- Validation split (optional) ------------------------------------------
    val_dataset: datasets.CIFAR10 | None = None
    if with_validation_split:
        # Use eval transforms for validation subset
        eval_train = datasets.CIFAR10(
            root=str(data_dir),
            train=True,
            download=False,
            transform=eval_tfms,
        )
        val_size = int(validation_pct * len(eval_train))
        train_size = len(eval_train) - val_size
        _, val_subset = random_split(
            eval_train,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        val_dataset = val_subset  # type: ignore[assignment]

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