.PHONY: install train-qc app test

install:
	pip install -r requirements.txt

train-qc:
	python src/qc/train_qc.py --data $${QC_DATA_DIR:-data/processed/qc} --model-out $${QC_MODEL_PATH:-models/qc_v1.onnx}

app:
	streamlit run src/app_streamlit.py

test:
	pytest -q
