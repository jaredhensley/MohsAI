"""Datasets and transforms for frozen section QC."""
from __future__ import annotations

from pathlib import Path
from torchvision import transforms
from torchvision.datasets import ImageFolder


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int, train: bool = True) -> transforms.Compose:
    augmentations: list[transforms.transforms] = []
    if train:
        augmentations.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
            ]
        )
    augmentations.extend(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(augmentations)


def build_datasets(data_dir: Path, image_size: int) -> tuple[ImageFolder, ImageFolder]:
    train_ds = ImageFolder(root=str(data_dir), transform=build_transforms(image_size, train=True))
    val_ds = ImageFolder(root=str(data_dir), transform=build_transforms(image_size, train=False))
    return train_ds, val_ds
