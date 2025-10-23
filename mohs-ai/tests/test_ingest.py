from pathlib import Path
import csv

from src.ingest_deid import ingest_metadata


def test_ingest_metadata(tmp_path: Path) -> None:
    labels_csv = tmp_path / "labels.csv"
    with labels_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "image_id", "label"])
        writer.writeheader()
        writer.writerow({"case_id": "CASE1", "image_id": "IMG1", "label": "good"})
    deid_csv = tmp_path / "deid.csv"
    with deid_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "deid_case"])
        writer.writeheader()
        writer.writerow({"case_id": "CASE1", "deid_case": "DEID_CASE1"})

    output_dir = tmp_path / "out"
    output_path = ingest_metadata(labels_csv, deid_csv, output_dir)

    assert output_path.exists()
    rows = list(csv.DictReader(output_path.open("r", encoding="utf-8")))
    assert rows[0]["deid_case"] == "DEID_CASE1"
