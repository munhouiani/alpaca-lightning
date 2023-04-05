.PHONY: install-poetry
install-poetry:
	@echo "Installing Poetry"
	curl -sSL https://install.python-poetry.org | python3 -

.PHONY: install-pyenv
install-pyenv:
	@echo "Installing Pyenv"
	curl https://pyenv.run | bash

.PHONY: prepare-python310
prepare-python310: install-pyenv
	@echo "Preparing Python 3.10"
	pyenv install 3.10.11

.PHONY: prepare-env
prepare-env: prepare-python310 install-poetry
	@echo "Preparing environment"
	@echo "Use Python 3.10"
	poetry env use 3.10
	@echo "Install dependencies"
	poetry install --without dev
	@echo "Add Lightning Logs Folders"
	mkdir -p lightning_logs

.PHONY: train
train:
	@echo "Training model"
	poetry run python train.py
