import pytest

pytest.importorskip("PIL")

from PIL import Image

from src.margin.infer_margin import infer_margin_stub


def test_margin_stub_returns_overlay() -> None:
    image = Image.new("RGB", (512, 512), color="white")
    result = infer_margin_stub(image)

    assert set(result.keys()) == {"prob_involved", "overlay", "explanation"}
    assert 0.0 <= result["prob_involved"] <= 1.0
    overlay = result["overlay"]
    assert overlay.size == image.size
    assert isinstance(result["explanation"], str)
