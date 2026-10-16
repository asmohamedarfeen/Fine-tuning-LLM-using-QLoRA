"""
Tests for MLX QLoRA training engine, configuration, fusion, and GGUF export.
"""

from pathlib import Path

import pytest

from mlx_engine.export_gguf import (
    get_gguf_conversion_command,
    print_gguf_instructions,
)
from mlx_engine.fuse_mlx import build_mlx_fuse_args, fuse_adapter_command
from mlx_engine.train_mlx import (
    build_mlx_train_args,
    load_config,
    train_mlx_cli_command,
)


def test_build_mlx_train_args():
    cfg = {
        "model": "models/qwen2.5-7b-mlx-4bit",
        "data_path": "data/processed",
        "adapter_path": "adapters/mlx_sql_lora",
        "batch_size": 2,
        "iters": 500,
        "learning_rate": 1e-4,
        "lora_layers": 16,
    }
    cmd = build_mlx_train_args(cfg)
    assert "-m" in cmd
    assert "mlx_lm.lora" in cmd
    assert "--train" in cmd
    assert "--batch-size" in cmd
    assert "2" in cmd
    assert "--iters" in cmd
    assert "500" in cmd
    assert "--lora-layers" in cmd
    assert "16" in cmd
    assert "--model" in cmd
    assert "models/qwen2.5-7b-mlx-4bit" in cmd


def test_train_mlx_cli_command():
    cfg = {
        "model": "models/qwen2.5-7b-mlx-4bit",
        "data_path": "data/processed",
    }
    cmd = train_mlx_cli_command(cfg)
    assert "-m" in cmd
    assert "mlx_lm.lora" in cmd
    assert "--data" in cmd
    assert "data/processed" in cmd


def test_build_mlx_train_args_model_selection():
    # quantized_model_path takes precedence over model
    cfg1 = {
        "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "quantized_model_path": "models/custom-mlx-4bit",
    }
    cmd1 = build_mlx_train_args(cfg1)
    model_idx = cmd1.index("--model")
    assert cmd1[model_idx + 1] == "models/custom-mlx-4bit"

    # model used when quantized_model_path is absent
    cfg2 = {
        "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
    }
    cmd2 = build_mlx_train_args(cfg2)
    model_idx2 = cmd2.index("--model")
    assert cmd2[model_idx2 + 1] == "Qwen/Qwen2.5-Coder-7B-Instruct"


def test_build_mlx_fuse_args():
    cmd = build_mlx_fuse_args("models/base", "adapters/lora", "models/fused")
    assert "-m" in cmd
    assert "mlx_lm.fuse" in cmd
    assert "--save-path" in cmd
    assert "models/fused" in cmd
    assert "--model" in cmd
    assert "models/base" in cmd
    assert "--adapter-path" in cmd
    assert "adapters/lora" in cmd


def test_fuse_adapter_command():
    cmd = fuse_adapter_command("models/base", "adapters/lora", "models/fused")
    assert "-m" in cmd
    assert "mlx_lm.fuse" in cmd
    assert "--save-path" in cmd
    assert "models/fused" in cmd


def test_load_config_valid():
    config_file = Path("configs/mlx_train_config.yaml")
    assert config_file.exists()
    cfg = load_config(config_file)
    assert cfg["model"] == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert cfg["quantized_model_path"] == "models/qwen2.5-7b-mlx-4bit"
    assert cfg["data_path"] == "data/processed"
    assert cfg["adapter_path"] == "adapters/mlx_sql_lora"
    assert cfg["lora_layers"] == 16
    assert cfg["batch_size"] == 2
    assert cfg["iters"] == 1000
    assert float(cfg["learning_rate"]) == pytest.approx(1e-4)
    assert cfg["steps_per_report"] == 10
    assert cfg["steps_per_eval"] == 100
    assert cfg["save_every"] == 200
    assert cfg["max_seq_length"] == 2048


def test_load_config_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("configs/nonexistent_config.yaml")


def test_get_gguf_conversion_command():
    cmd = get_gguf_conversion_command(
        fused_model_dir="models/test-fused",
        outtype="q4_k_m",
        llama_cpp_path="tools/llama.cpp",
    )
    assert any("convert_hf_to_gguf.py" in arg for arg in cmd)
    assert "models/test-fused" in cmd
    assert "--outtype" in cmd
    assert "q4_k_m" in cmd


def test_print_gguf_instructions(capsys):
    print_gguf_instructions("models/custom-fused")
    captured = capsys.readouterr()
    assert "Export to GGUF" in captured.out
    assert "models/custom-fused" in captured.out
    assert "llama.cpp" in captured.out
    assert "ollama create" in captured.out
