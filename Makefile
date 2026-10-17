.PHONY: help test install-mac install-cuda data demo cli train-mlx fuse-mlx train-cuda-dry train-cuda lint format clean

help:
	@echo "Enterprise Text-to-SQL Engine Developer Commands:"
	@echo "  make test            Run full pytest test suite"
	@echo "  make install-mac     Install Apple Silicon (MLX) dependencies"
	@echo "  make install-cuda    Install NVIDIA CUDA dependencies"
	@echo "  make data            Generate deterministic enterprise training/validation/test datasets"
	@echo "  make demo            Launch interactive Gradio Web UI with SQLite sandbox"
	@echo "  make cli             Launch interactive terminal CLI query runner"
	@echo "  make train-mlx       Run/display Apple Silicon MLX LoRA training command"
	@echo "  make fuse-mlx        Fuse trained MLX LoRA adapter weights"
	@echo "  make train-cuda-dry  Run CUDA engine dry-run validation (non-GPU compatible)"
	@echo "  make train-cuda      Run full QLoRA training on NVIDIA GPU"
	@echo "  make lint            Run ruff linter check"
	@echo "  make format          Run ruff code formatter"
	@echo "  make clean           Clean cache files and build artifacts"

test:
	pytest tests/ -v

install-mac:
	pip install -r requirements-mac.txt

install-cuda:
	pip install -r requirements-cuda.txt

data:
	python data/prepare_dataset.py

demo:
	python demo/app.py

cli:
	python demo/cli_query.py

train-mlx:
	python mlx_engine/train_mlx.py

fuse-mlx:
	python mlx_engine/fuse_mlx.py

train-cuda-dry:
	python cuda_engine/train_cuda.py --dry-run

train-cuda:
	python cuda_engine/train_cuda.py --config configs/hf_train_config.yaml

lint:
	ruff check .

format:
	ruff format .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
