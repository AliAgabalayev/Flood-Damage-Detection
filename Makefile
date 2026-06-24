PY := PYTHONPATH=src .venv/bin/python
CONFIG ?= config/default.yaml
MLFLOW_URI ?= sqlite:///mlflow.db

.DEFAULT_GOAL := help

.PHONY: help install config train eval predict smoke mlflow-ui dvc-push dvc-pull lint

help:
	@echo "install    install pinned deps into .venv"
	@echo "config     validate config/default.yaml"
	@echo "train      train baseline (DeepLabV3+)"
	@echo "eval       evaluate a checkpoint on a split"
	@echo "predict    tiled predict to GeoTIFF"
	@echo "smoke      plumbing check on a tiny real-data subset"
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
	$(PY) -m inference.predict --config $(CONFIG)

smoke:
	$(PY) -m training.train --config $(CONFIG) --smoke

mlflow-ui:
	.venv/bin/mlflow ui --backend-store-uri $(MLFLOW_URI)

dvc-push:
	.venv/bin/dvc push

dvc-pull:
	.venv/bin/dvc pull

lint:
	$(PY) -m compileall -q src
