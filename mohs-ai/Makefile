PYTHON ?= python3
VENV_DIR ?= .venv
VENV_PYTHON ?= $(VENV_DIR)/bin/python
VENV_PIP ?= $(VENV_DIR)/bin/pip

.PHONY: install venv train-qc app test

venv: $(VENV_PYTHON)

$(VENV_PYTHON):
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_PYTHON) -m pip install --upgrade pip

install: venv
	$(VENV_PIP) install -r requirements.txt

train-qc: venv
	$(VENV_PYTHON) -m src.qc.train_qc --data $${QC_DATA_DIR:-data/processed/qc} --model-out $${QC_MODEL_PATH:-models/qc_v1.onnx}

app: venv
	$(VENV_PYTHON) -m streamlit run src/app_streamlit.py

test: venv
	$(VENV_PYTHON) -m pytest -q
