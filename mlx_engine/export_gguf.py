"""
Guidance and helper for exporting fused weights to GGUF format for Ollama / LM Studio.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def get_gguf_conversion_command(
    fused_model_dir: str = "models/qwen2.5-7b-sql-fused",
    outtype: str = "q4_k_m",
    llama_cpp_path: str = "llama.cpp",
) -> list[str]:
    """Build the llama.cpp conversion command for exporting HF weights to GGUF.

    Args:
        fused_model_dir: Directory containing fused Hugging Face weights.
        outtype: Target quantization type (e.g. q4_k_m, q8_0, f16).
        llama_cpp_path: Path to cloned llama.cpp repository.

    Returns:
        List of command-line arguments for running convert_hf_to_gguf.py.
    """
    return [
        sys.executable,
        str(Path(llama_cpp_path) / "convert_hf_to_gguf.py"),
        fused_model_dir,
        "--outtype",
        outtype,
    ]


def print_gguf_instructions(
    fused_model_dir: str = "models/qwen2.5-7b-sql-fused",
) -> None:
    """Print step-by-step export instructions for GGUF conversion and Ollama deployment.

    Args:
        fused_model_dir: Directory containing fused model weights.
    """
    print(
        f"""
============================================================
Export to GGUF for Local Ollama / LM Studio Deployment
============================================================
1. Clone llama.cpp if not present:
   git clone https://github.com/ggerganov/llama.cpp.git

2. Run conversion:
   python llama.cpp/convert_hf_to_gguf.py {fused_model_dir} --outtype q4_k_m

3. Deploy with Ollama:
   ollama create qwen-sql -f demo/Modelfile
   ollama run qwen-sql
============================================================
"""
    )


def main() -> None:
    """CLI entrypoint for displaying GGUF export guidance."""
    parser = argparse.ArgumentParser(
        description="GGUF export helper for Ollama / LM Studio"
    )
    parser.add_argument(
        "--fused-model-dir",
        type=str,
        default="models/qwen2.5-7b-sql-fused",
        help="Path to fused model weights",
    )
    args = parser.parse_args()
    print_gguf_instructions(args.fused_model_dir)


if __name__ == "__main__":
    main()
