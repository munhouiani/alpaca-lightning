.PHONY: prepare-env
prepare-env:
	@echo "Preparing environment"
	@echo "Use Python 3.10"
	poetry env use 3.10
	@echo "Install dependencies"
	poetry install --without dev

.PHONY: train
train:
	@echo "Training model"
	poetry run python train.py
