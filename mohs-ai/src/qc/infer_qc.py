"""ONNX inference for frozen section QC."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms

try:  # allow running as script or package module
    from config import QCConfig  # type: ignore
except ImportError:
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
    from src.config import QCConfig  # type: ignore

from .dataset_qc import IMAGENET_MEAN, IMAGENET_STD


class QCOnnxInferencer:
    def __init__(self, model_path: Path, image_size: int) -> None:
        self.model_path = model_path
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def _forward(self, tensor: np.ndarray) -> float:
        logits = self.session.run(["logits"], {"input": tensor})[0]
        prob = float(1.0 / (1.0 + np.exp(-logits))[0][0])
        return prob

    def predict(self, image_path: Path) -> Dict[str, float | str]:
        image = Image.open(image_path).convert("RGB")
        return self.predict_image(image)

    def predict_image(self, image: Image.Image) -> Dict[str, float | str]:
        tensor = self.transform(image).unsqueeze(0).numpy()
        prob = self._forward(tensor)
        label = "poor" if prob >= 0.5 else "usable"
        return {"prob_poor": prob, "label": label}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QC inference on an image")
    parser.add_argument("--image", type=Path, required=True, help="Path to input image")
    parser.add_argument("--model", type=Path, default=None, help="Path to ONNX model")
    args = parser.parse_args()

    cfg = QCConfig.from_env()
    model_path = args.model or cfg.model_path
    inferencer = QCOnnxInferencer(model_path, cfg.image_size)
    result = inferencer.predict(args.image)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
