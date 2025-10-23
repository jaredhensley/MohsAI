"""Patch extraction utilities for future margin analysis models."""
from __future__ import annotations

from typing import Iterable, Tuple
from PIL import Image


def sliding_window_patches(image: Image.Image, patch_size: Tuple[int, int], stride: Tuple[int, int]) -> Iterable[Image.Image]:
    """Yield image patches in a sliding window fashion (stub implementation)."""
    width, height = image.size
    pw, ph = patch_size
    sw, sh = stride
    for y in range(0, height - ph + 1, sh):
        for x in range(0, width - pw + 1, sw):
            yield image.crop((x, y, x + pw, y + ph))
    # TODO: integrate with real WSIs and support padding.
