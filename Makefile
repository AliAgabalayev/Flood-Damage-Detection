PY := PYTHONPATH=src .venv/bin/python
CONFIG ?= config/default.yaml
MLFLOW_URI ?= sqlite:///mlflow.db

.DEFAULT_GOAL := help

.PHONY: help install config train eval predict finalists download-weak-data weak-splits pretrain-finetune mlflow-ui dvc-push dvc-pull lint

FINALISTS_MATRIX := config/experiments/loss_finalists.yaml
FINALISTS_CKPT := models/loss_finalists

help:
	@echo "install    install pinned deps into .venv"
	@echo "config     validate config/default.yaml"
	@echo "train      train baseline (DeepLabV3+)"
	@echo "eval       evaluate a checkpoint on a split"
	@echo "predict    tiled predict to GeoTIFF (INPUT=.. OUTPUT=..)"
	@echo "finalists  run M1b loss finalists (ONLY=<run_name> for a single run)"
	@echo "download-weak-data  download Sen1Floods11 weak-labeled chips (~6.8 GiB)"
	@echo "weak-splits         build train/val split CSVs for the weak-labeled chips"
	@echo "pretrain-finetune   pretrain on weak labels, fine-tune on hand labels"
	@echo "mlflow-ui  launch MLflow UI on the sqlite backend"
	@echo "dvc-push   push tracked data/models to DVC remote"
	@echo "dvc-pull   pull tracked data/models from DVC remote"
	@echo "lint       byte-compile all source"

install:
	.venv/bin/pip install -r requirements.txt

config:
	$(PY) -m utils.config $(CONFIG)

train:
	$(PY) -m training.train --config $(CONFIG)

eval:
	$(PY) -m inference.evaluate --config $(CONFIG)

predict:
	$(PY) -m inference.predict \
		--input $(INPUT) \
		--output $(OUTPUT) \
		--config $(CONFIG)

finalists:
	$(PY) scripts/run_loss_study.py --matrix $(FINALISTS_MATRIX) --ckpt-root $(FINALISTS_CKPT) $(if $(ONLY),--only $(ONLY),)

download-weak-data:
	bash scripts/download_weak_data.sh

weak-splits:
	$(PY) scripts/make_weak_splits.py

pretrain-finetune:
	$(PY) scripts/run_pretrain_finetune.py $(if $(EVAL_TEST),--evaluate-test,)

mlflow-ui:
	.venv/bin/mlflow ui --backend-store-uri $(MLFLOW_URI)

dvc-push:
	.venv/bin/dvc push

dvc-pull:
	.venv/bin/dvc pull

lint:
	$(PY) -m compileall -q src
