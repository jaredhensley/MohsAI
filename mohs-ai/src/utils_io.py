"""Utility helpers for local file IO and metadata workflows."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_images(directory: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    return [p for p in directory.iterdir() if p.suffix.lower() in exts]
