"""
Fuses MLX LoRA adapters back into the base model weights.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_mlx_fuse_args(
    model_path: str | Path, adapter_path: str | Path, save_path: str | Path
) -> list[str]:
    """Build command-line arguments for mlx_lm.fuse.

    Args:
        model_path: Base model or 4-bit model directory path.
        adapter_path: Trained LoRA adapter directory path.
        save_path: Output directory path for the fused model weights.

    Returns:
        List of command-line arguments for mlx_lm.fuse.
    """
    return [
        sys.executable,
        "-m",
        "mlx_lm.fuse",
        "--model",
        str(model_path),
        "--adapter-path",
        str(adapter_path),
        "--save-path",
        str(save_path),
    ]


def fuse_adapter_command(
    model_path: str | Path, adapter_path: str | Path, output_path: str | Path
) -> list[str]:
    """Generate MLX adapter fusion CLI command list.

    Args:
        model_path: Base model or 4-bit model directory path.
        adapter_path: Trained LoRA adapter directory path.
        output_path: Output directory path for fused model weights.

    Returns:
        List of CLI arguments for fusing adapters into base weights.
    """
    return build_mlx_fuse_args(model_path, adapter_path, output_path)


def main() -> None:
    """CLI entrypoint to display MLX adapter fusion command."""
    parser = argparse.ArgumentParser(
        description="Fuse MLX LoRA adapters into base weights"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/qwen2.5-7b-mlx-4bit",
        help="Base or 4-bit model path",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="adapters/mlx_sql_lora",
        help="LoRA adapter checkpoint directory",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="models/qwen2.5-7b-sql-fused",
        help="Target directory for fused weights",
    )
    args = parser.parse_args()

    cmd = build_mlx_fuse_args(
        model_path=args.model,
        adapter_path=args.adapter_path,
        save_path=args.save_path,
    )
    print(" ".join(cmd))


if __name__ == "__main__":
    main()
