.PHONY: prepare-env
prepare-env:
	@echo "Preparing environment"
	@echo "Use Python 3.10"
	poetry env use 3.10
	@echo "Install dependencies"
	poetry install --without dev
	@echo "Add Lightning Logs Folders"
	mkdir -p alpaca_logs/lightning_logs

.PHONY: train
train:
	@echo "Training model"
	poetry run python train.py
