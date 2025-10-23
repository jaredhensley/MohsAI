# Mohs AI (Local-Only)

This repository provides an **offline-ready** prototype for a Mohs dermatology clinic. It includes:

- A full frozen section quality control (QC) training + ONNX inference workflow.
- A Streamlit application with two tabs:
  - **QC Check** – uses the trained MobileNetV3-Small model exported to ONNX.
  - **Margin Assist (Prototype)** – heuristic probability + heatmap overlay stub for future development.
- Local-only utilities for metadata ingestion, dataset curation, and basic testing.

All paths and configuration values are controlled via environment variables (see `.env.example`). No external network calls are made at runtime.

## Project Layout

```
mohs-ai/
├─ .env.example            # Sample configuration
├─ Makefile                # Convenience commands
├─ requirements.txt        # Offline-installable dependencies
├─ data/processed/qc/      # Generated dummy images for quick experimentation
├─ metadata/               # Example metadata + de-identification mapping
├─ models/                 # ONNX model output directory
├─ src/                    # Application source code
│  ├─ qc/                  # QC training + inference modules
│  ├─ margin/              # Margin Assist stubs
│  └─ app_streamlit.py     # Streamlit app entrypoint
└─ tests/                  # Pytest coverage for core utilities and stubs
```

## Getting Started

1. **Copy environment variables**
   ```bash
   cp .env.example .env
   ```
   Adjust values as needed (e.g., dataset/model paths).

2. **Install dependencies (creates local virtual environment)**
   ```bash
   make install PYTHON=python3.11
   ```
   ONNX Runtime only ships wheels up through Python 3.12 today, so use a Python 3.11 interpreter (override `PYTHON=...` if your default points elsewhere). This bootstraps a `.venv/` directory with `python3 -m venv` and installs requirements inside it.
   To work interactively after installing, activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

3. **Train the QC model**
   The training script will generate deterministic dummy samples under `data/processed/qc/{good,poor}` if none exist.
   ```bash
   make train-qc
   ```
   The command exports `models/qc_v1.onnx` and writes metrics to `metadata/qc_metrics.json`.

4. **Run the Streamlit app**
   ```bash
   make app
   ```
   - **QC Check tab** – upload a frozen section image to obtain probability + label.
   - **Margin Assist tab** – upload an image to view the heuristic overlay and fake probability.

5. **Execute tests**
   ```bash
   make test
   ```

## Command-Line Tools

- `python src/ingest_deid.py` – join `metadata/labels.csv` with `metadata/deid_map.csv`.
- `python src/curate_split.py` – preview simple train/validation splits.
- `python src/qc/infer_qc.py --image <path>` – run ONNX inference and print JSON results.

## Future Work

- Replace the Margin Assist stub (`src/margin/`) with a true MIL pipeline.
- Integrate WSI ingestion and patch extraction.
- Expand testing and add performance benchmarking for larger datasets.

## License

This project is intended for internal research and prototyping only. Not for clinical use.
