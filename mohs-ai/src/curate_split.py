"""Utility script to split processed images into train/val/test partitions."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Sequence

try:
    from utils_io import list_images  # type: ignore
except ImportError:
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
    from src.utils_io import list_images  # type: ignore


RANDOM_SEED = 2024


def split_paths(paths: Sequence[Path], val_fraction: float = 0.2) -> tuple[list[Path], list[Path]]:
    rnd = random.Random(RANDOM_SEED)
    paths = list(paths)
    rnd.shuffle(paths)
    split_idx = max(1, int(len(paths) * (1 - val_fraction))) if paths else 0
    train = paths[:split_idx]
    val = paths[split_idx:]
    return train, val


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate dataset splits for QC training")
    parser.add_argument("data_dir", type=Path, default=Path("data/processed/qc"))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()

    good_imgs = list_images(args.data_dir / "good")
    poor_imgs = list_images(args.data_dir / "poor")

    train_good, val_good = split_paths(good_imgs, args.val_fraction)
    train_poor, val_poor = split_paths(poor_imgs, args.val_fraction)

    print(f"Good train: {len(train_good)}, Good val: {len(val_good)}")
    print(f"Poor train: {len(train_poor)}, Poor val: {len(val_poor)}")


if __name__ == "__main__":
    main()
