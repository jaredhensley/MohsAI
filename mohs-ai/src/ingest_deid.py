"""Ingest and de-identify metadata for Mohs AI workflows."""
from __future__ import annotations

import argparse
from pathlib import Path
import csv

from config import QCConfig


def ingest_metadata(labels_csv: Path, deid_map_csv: Path, output_dir: Path) -> Path:
    """Join label metadata with a de-identification map and persist as CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "qc_labels_deid.csv"

    deid_map: dict[str, str] = {}
    with deid_map_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            deid_map[row["case_id"]] = row["deid_case"]

    with labels_csv.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.DictReader(src)
        fieldnames = reader.fieldnames + ["deid_case"] if reader.fieldnames else ["case_id", "image_id", "label", "deid_case"]
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            case = row.get("case_id", "UNKNOWN")
            row["deid_case"] = deid_map.get(case, f"DEID_{case}")
            writer.writerow(row)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Join labels with de-identification mapping")
    parser.add_argument("--labels", type=Path, default=Path("metadata/labels.csv"), help="Path to labels CSV")
    parser.add_argument("--deid", type=Path, default=Path("metadata/deid_map.csv"), help="Path to de-identification CSV")
    parser.add_argument("--out", type=Path, default=Path("metadata/processed"), help="Output directory")
    args = parser.parse_args()

    cfg = QCConfig.from_env()
    output_path = ingest_metadata(args.labels, args.deid, args.out)
    print(f"Merged metadata saved to {output_path.relative_to(Path.cwd())}")
    print(f"QC data directory: {cfg.data_dir}")


if __name__ == "__main__":
    main()
