"""Utilities to generate deterministic dummy QC images for offline demos."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw


RNG_SEED = 2024


def _draw_cross(draw: ImageDraw.ImageDraw, size: int, color: tuple[int, int, int]) -> None:
    thickness = max(size // 16, 1)
    center = size // 2
    draw.rectangle((center - thickness, 0, center + thickness, size), fill=color)
    draw.rectangle((0, center - thickness, size, center + thickness), fill=color)


def _draw_circle(draw: ImageDraw.ImageDraw, size: int, color: tuple[int, int, int]) -> None:
    radius = size // 3
    left = size // 2 - radius
    top = size // 2 - radius
    right = size // 2 + radius
    bottom = size // 2 + radius
    draw.ellipse((left, top, right, bottom), outline=color, width=max(size // 32, 1))


def _gradient_background(size: int, channels: Iterable[int]) -> np.ndarray:
    x = np.linspace(0, 1, size, dtype=np.float32)
    gradient = np.outer(np.ones(size, dtype=np.float32), x)
    image = np.stack([gradient * c for c in channels], axis=-1)
    return (image * 255).astype(np.uint8)


def create_dummy_images(target_dir: Path, image_size: int = 224, per_class: int = 4) -> None:
    """Create deterministic dummy images if none exist in the class directories."""
    rng = np.random.default_rng(RNG_SEED)
    classes = {
        "good": (np.array([0.4, 0.8, 0.4]), (34, 139, 34)),
        "poor": (np.array([0.9, 0.3, 0.3]), (178, 34, 34)),
    }
    for class_name, (channels, accent) in classes.items():
        class_dir = target_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        existing = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        if existing:
            continue
        for idx in range(per_class):
            background = _gradient_background(image_size, channels)
            noise = rng.normal(0, 5, size=background.shape).astype(np.int16)
            image = np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(image, mode="RGB")
            draw = ImageDraw.Draw(pil_img)
            if class_name == "good":
                _draw_circle(draw, image_size, accent)
            else:
                _draw_cross(draw, image_size, accent)
            pil_img.save(class_dir / f"dummy_{class_name}_{idx + 1}.jpg", format="JPEG", quality=90)
