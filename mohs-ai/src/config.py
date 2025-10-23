"""Application configuration utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv(override=False)


def _resolve_path(value: str | None, default: str) -> Path:
    """Resolve a path from the environment or default, expanding user/home."""
    raw = value or default
    path = Path(os.path.expanduser(raw)).resolve()
    return path


@dataclass(frozen=True)
class QCConfig:
    data_dir: Path
    model_path: Path
    metrics_path: Path
    image_size: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    val_split: float
    early_stop_patience: int

    @staticmethod
    def from_env() -> "QCConfig":
        data_root = os.getenv("DATA_ROOT", "data/processed")
        return QCConfig(
            data_dir=_resolve_path(os.getenv("QC_DATA_DIR"), f"{data_root}/qc"),
            model_path=_resolve_path(os.getenv("QC_MODEL_PATH"), "models/qc_v1.onnx"),
            metrics_path=_resolve_path(os.getenv("QC_METRICS_PATH"), "metadata/qc_metrics.json"),
            image_size=int(os.getenv("QC_IMAGE_SIZE", "224")),
            batch_size=int(os.getenv("QC_BATCH_SIZE", "8")),
            epochs=int(os.getenv("QC_EPOCHS", "5")),
            lr=float(os.getenv("QC_LR", "5e-4")),
            weight_decay=float(os.getenv("QC_WEIGHT_DECAY", "1e-4")),
            val_split=float(os.getenv("QC_VAL_SPLIT", "0.2")),
            early_stop_patience=int(os.getenv("QC_EARLY_STOP_PATIENCE", "3")),
        )


def ensure_directory(path: Path) -> Path:
    """Ensure that a directory exists."""
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)
    return path
