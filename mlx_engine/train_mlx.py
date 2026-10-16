"""
Apple Silicon M4 Native Training Runner via mlx-lm.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


def build_mlx_train_args(config: dict[str, Any]) -> list[str]:
    """Build command-line arguments for mlx_lm.lora training.

    Args:
        config: Mapping containing MLX training configuration parameters.

    Returns:
        List of command-line arguments suitable for subprocess execution.
    """
    model_path = str(
        config.get("quantized_model_path")
        or config.get("model")
        or "models/qwen2.5-7b-mlx-4bit"
    )
    return [
        sys.executable,
        "-m",
        "mlx_lm.lora",
        "--model",
        model_path,
        "--train",
        "--data",
        str(config.get("data_path", "data/processed")),
        "--batch-size",
        str(config.get("batch_size", 2)),
        "--lora-layers",
        str(config.get("lora_layers", 16)),
        "--iters",
        str(config.get("iters", 1000)),
        "--learning-rate",
        str(config.get("learning_rate", 1e-4)),
        "--steps-per-report",
        str(config.get("steps_per_report", 10)),
        "--steps-per-eval",
        str(config.get("steps_per_eval", 100)),
        "--save-every",
        str(config.get("save_every", 200)),
        "--adapter-path",
        str(config.get("adapter_path", "adapters/mlx_sql_lora")),
        "--max-seq-length",
        str(config.get("max_seq_length", 2048)),
    ]


def train_mlx_cli_command(config: dict[str, Any]) -> list[str]:
    """Generate MLX training CLI command list.

    Args:
        config: MLX configuration mapping.

    Returns:
        List of CLI arguments for MLX LoRA training.
    """
    return build_mlx_train_args(config)


def load_config(path: str | Path = "configs/mlx_train_config.yaml") -> dict[str, Any]:
    """Load MLX training configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Dictionary containing configuration settings.

    Raises:
        FileNotFoundError: If configuration file does not exist.
        yaml.YAMLError: If YAML parsing fails.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def main() -> None:
    """CLI entrypoint to display MLX training command."""
    parser = argparse.ArgumentParser(
        description="Apple Silicon MLX QLoRA Training Runner"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mlx_train_config.yaml",
        help="Path to MLX train config YAML",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cmd = build_mlx_train_args(cfg)
    print("[*] To execute MLX QLoRA training on Apple Silicon M4, run:")
    print(" ".join(cmd))


if __name__ == "__main__":
    main()
