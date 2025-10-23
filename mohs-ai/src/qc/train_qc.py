"""Training script for frozen section QC classifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict
import time
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

try:  # allow running as package or as a standalone script
    from config import QCConfig  # type: ignore
    from utils_io import write_json  # type: ignore
except ImportError:  # executed via `python -m src.qc.train_qc` or lacking cwd on sys.path
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
    from src.config import QCConfig  # type: ignore
    from src.utils_io import write_json  # type: ignore

from .dataset_qc import build_datasets
from .dummy_data import create_dummy_images
from .model_qc import create_model, logits_to_prob


RANDOM_SEED = 1337


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_class_weights(targets: list[int]) -> torch.Tensor:
    counts = np.bincount(targets, minlength=2)
    neg, pos = counts[0], counts[1]
    if pos == 0:
        pos_weight = torch.tensor(1.0)
    else:
        pos_weight = torch.tensor(neg / max(pos, 1))
    return pos_weight


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    metrics: Dict[str, float] = {}
    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auc"] = float("nan")
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["recall_poor"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["precision_poor"] = float(precision_score(y_true, y_pred, zero_division=0))
    return metrics


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> tuple[float, Dict[str, float]]:
    model.train()
    epoch_loss = 0.0
    all_targets: list[int] = []
    all_probs: list[float] = []
    for images, targets in loader:
        images = images.to(device)
        targets = targets.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * images.size(0)
        probs = logits_to_prob(logits.detach()).cpu().numpy().flatten()
        all_probs.extend(probs.tolist())
        all_targets.extend(targets.cpu().numpy().flatten().tolist())
    avg_loss = epoch_loss / max(len(loader.dataset), 1)
    metrics = compute_metrics(np.array(all_targets), np.array(all_probs)) if all_targets else {"auc": float("nan"), "f1": 0.0, "recall_poor": 0.0, "precision_poor": 0.0}
    metrics["loss"] = avg_loss
    return avg_loss, metrics


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_targets: list[int] = []
    all_probs: list[float] = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.float().unsqueeze(1).to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            total_loss += loss.item() * images.size(0)
            probs = logits_to_prob(logits).cpu().numpy().flatten()
            all_probs.extend(probs.tolist())
            all_targets.extend(targets.cpu().numpy().flatten().tolist())
    avg_loss = total_loss / max(len(loader.dataset), 1)
    metrics = compute_metrics(np.array(all_targets), np.array(all_probs)) if all_targets else {"auc": float("nan"), "f1": 0.0, "recall_poor": 0.0, "precision_poor": 0.0}
    metrics["loss"] = avg_loss
    return metrics


def export_onnx(model: nn.Module, path: Path, image_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=12,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Frozen Section QC model")
    parser.add_argument("--data", type=Path, help="Path to QC dataset", default=None)
    parser.add_argument("--model-out", type=Path, default=None, help="Output ONNX path")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--val-split", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    args = parser.parse_args()

    cfg = QCConfig.from_env()
    data_dir = Path(args.data) if args.data is not None else cfg.data_dir
    model_out = args.model_out or cfg.model_path
    metrics_out = cfg.metrics_path

    seed_everything(RANDOM_SEED)

    data_dir.mkdir(parents=True, exist_ok=True)
    create_dummy_images(data_dir, cfg.image_size)

    train_dataset, val_dataset = build_datasets(data_dir, cfg.image_size)
    total_len = len(train_dataset)
    if total_len == 0:
        raise RuntimeError(f"No images found in {data_dir}")

    val_fraction = args.val_split if args.val_split is not None else cfg.val_split
    raw_val_len = max(1, int(total_len * val_fraction)) if total_len > 1 else 1
    train_len = max(total_len - raw_val_len, 1)
    val_len = total_len - train_len
    splits = [train_len, val_len]
    train_subset, val_subset = random_split(train_dataset, splits, generator=torch.Generator().manual_seed(RANDOM_SEED))
    val_subset = Subset(val_dataset, val_subset.indices)

    batch_size = args.batch_size or cfg.batch_size
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    targets = [train_dataset.targets[i] for i in train_subset.indices]
    pos_weight = get_class_weights(targets)

    device = torch.device("cpu")
    model = create_model()
    model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr or cfg.lr, weight_decay=args.weight_decay or cfg.weight_decay)

    best_auc = -np.inf
    best_state: Dict[str, torch.Tensor] | None = None
    patience = args.patience if args.patience is not None else cfg.early_stop_patience
    patience_counter = 0

    history: dict[str, list[float]] = {"train_auc": [], "val_auc": []}

    epochs = args.epochs if args.epochs is not None else cfg.epochs
    for epoch in range(1, epochs + 1):
        start = time.time()
        _, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        duration = time.time() - start

        history["train_auc"].append(train_metrics["auc"])
        history["val_auc"].append(val_metrics["auc"])

        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train": train_metrics,
                    "val": val_metrics,
                    "duration_sec": duration,
                }
            )
        )

        current_auc = val_metrics.get("auc", float("nan"))
        if not np.isnan(current_auc) and current_auc > best_auc:
            best_auc = current_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)

    export_onnx(model, model_out, cfg.image_size)

    final_metrics = {
        "best_val_auc": float(best_auc),
        "history": history,
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": args.lr or cfg.lr,
            "weight_decay": args.weight_decay or cfg.weight_decay,
            "val_split": val_fraction,
        },
    }
    write_json(final_metrics, metrics_out)
    print(f"Saved ONNX model to {model_out}")
    print(f"Metrics written to {metrics_out}")


if __name__ == "__main__":
    main()
