from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("PIL")

import torch
from PIL import Image

from src.qc.infer_qc import QCOnnxInferencer


def _export_dummy_onnx(path: Path, image_size: int = 16) -> None:
    class Dummy(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple op
            return torch.zeros(x.size(0), 1)

    model = Dummy()
    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(model, dummy, str(path), input_names=["input"], output_names=["logits"], opset_version=12)


def test_qc_infer_outputs_dict(tmp_path: Path) -> None:
    onnx_path = tmp_path / "dummy.onnx"
    image_size = 16
    _export_dummy_onnx(onnx_path, image_size=image_size)

    inferencer = QCOnnxInferencer(onnx_path, image_size=image_size)
    image = Image.new("RGB", (image_size, image_size), color="white")
    result = inferencer.predict_image(image)

    assert set(result.keys()) == {"prob_poor", "label"}
    assert 0.0 <= result["prob_poor"] <= 1.0
    assert result["label"] in {"poor", "usable"}
