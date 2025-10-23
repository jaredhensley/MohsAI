"""Stub inference module for the Margin Assist prototype."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image


@dataclass
class MarginStubResult:
    prob_involved: float
    overlay: Image.Image
    explanation: str


def _gaussian_blob(field: np.ndarray, center: Tuple[int, int], sigma: float) -> None:
    h, w = field.shape
    y_indices, x_indices = np.ogrid[:h, :w]
    y0, x0 = center
    blob = np.exp(-((x_indices - x0) ** 2 + (y_indices - y0) ** 2) / (2 * sigma ** 2))
    field += blob


def _generate_heatmap(shape: Tuple[int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    field = np.zeros(shape, dtype=np.float32)
    h, w = shape
    for _ in range(3):
        center = (int(rng.uniform(0.2 * h, 0.8 * h)), int(rng.uniform(0.2 * w, 0.8 * w)))
        sigma = rng.uniform(0.1, 0.3) * min(h, w)
        _gaussian_blob(field, (center[0], center[1]), sigma)
    field = field - field.min()
    if field.max() > 0:
        field = field / field.max()
    return field


def infer_margin_stub(pil_img: Image.Image, out_size: Tuple[int, int] | None = None, seed: int = 1337) -> Dict[str, object]:
    """Generate a deterministic stub prediction for margin involvement."""
    image_rgb = pil_img.convert("RGB")
    arr = np.array(image_rgb)
    if out_size:
        arr = cv2.resize(arr, out_size[::-1], interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    intensity = float(gray.mean() / 255.0)
    score = 0.4 * intensity + 0.02 * np.log1p(lap_var)
    prob = float(1 / (1 + np.exp(-(score - 1.0))))

    heatmap = _generate_heatmap(gray.shape, seed)
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = np.clip(0.6 * arr + 0.4 * heatmap_color, 0, 255).astype(np.uint8)

    explanation = (
        "Prototype score derived from Laplacian focus variance and overall intensity. "
        "Gaussian blobs indicate heuristic regions of interest."
    )

    result = {
        "prob_involved": prob,
        "overlay": Image.fromarray(overlay),
        "explanation": explanation,
    }
    return result


__all__ = ["infer_margin_stub", "MarginStubResult"]
