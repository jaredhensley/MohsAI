"""Training placeholder for future margin involvement models."""
from __future__ import annotations

import argparse


def main() -> None:  # pragma: no cover - stub
    parser = argparse.ArgumentParser(description="Stub margin training entrypoint")
    parser.add_argument("--data", help="Path to training data", default="data/processed/margin")
    parser.add_argument("--out", help="Directory for model artifacts", default="models/margin")
    args = parser.parse_args()
    print("Margin training placeholder. No training performed.")
    print(f"Would read data from: {args.data}")
    print(f"Would export artifacts to: {args.out}")


if __name__ == "__main__":
    main()
