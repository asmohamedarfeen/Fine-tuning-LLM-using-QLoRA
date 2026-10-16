"""
Tests for NVIDIA CUDA QLoRA training engine, Hugging Face configurations, and Colab notebook.
"""

import json
from pathlib import Path

import pytest
import yaml

from cuda_engine.train_cuda import (
    get_bnb_config,
    get_peft_config,
    get_training_args,
    load_config,
    load_hf_dataset,
    run_training,
)


def test_hf_train_config():
    """Verify Hugging Face training YAML configuration exists and matches specifications."""
    config_path = Path("configs/hf_train_config.yaml")
    assert config_path.exists(), "configs/hf_train_config.yaml must exist"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg["model_id"] == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert cfg["lora_r"] == 16
    assert cfg["lora_alpha"] == 32
    assert cfg["batch_size"] == 2
    assert cfg["gradient_accumulation_steps"] == 4
    assert cfg["num_train_epochs"] == 3
    assert cfg["output_dir"] == "adapters/hf_sql_lora"
    assert cfg["dataset_path"] == "data/processed"
    assert cfg["max_seq_length"] == 2048
    assert "target_modules" in cfg


def test_load_config():
    """Verify loading valid config and handling missing config file."""
    cfg = load_config("configs/hf_train_config.yaml")
    assert cfg["model_id"] == "Qwen/Qwen2.5-Coder-7B-Instruct"

    with pytest.raises(FileNotFoundError):
        load_config("configs/nonexistent_cuda_config.yaml")


def test_get_bnb_config():
    """Verify get_bnb_config behaves correctly across platforms without unhandled crashes."""
    bnb_cfg = get_bnb_config()
    # On macOS Apple Silicon without CUDA bitsandbytes, returns None gracefully
    # On CUDA systems with bitsandbytes, returns BitsAndBytesConfig
    if bnb_cfg is not None:
        assert getattr(bnb_cfg, "load_in_4bit", False) is True
        assert getattr(bnb_cfg, "bnb_4bit_quant_type", None) == "nf4"


def test_get_peft_config():
    """Verify PEFT LoRA configuration construction."""
    cfg = {
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
    }
    peft_cfg = get_peft_config(cfg)
    assert peft_cfg is not None
    assert peft_cfg.r == 16
    assert peft_cfg.lora_alpha == 32
    assert peft_cfg.lora_dropout == pytest.approx(0.05)
    assert "q_proj" in peft_cfg.target_modules
    assert "v_proj" in peft_cfg.target_modules


def test_get_training_args():
    """Verify SFTConfig construction from configuration mapping."""
    cfg = {
        "output_dir": "adapters/test_lora",
        "learning_rate": 2.0e-4,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 3,
        "warmup_ratio": 0.05,
        "logging_steps": 10,
        "save_steps": 200,
        "max_seq_length": 2048,
    }
    args = get_training_args(cfg)
    assert args is not None
    assert args.output_dir == "adapters/test_lora"
    assert args.learning_rate == pytest.approx(2.0e-4)
    assert args.per_device_train_batch_size == 2
    assert args.gradient_accumulation_steps == 4
    assert args.num_train_epochs == 3
    assert args.max_length == 2048


def test_load_hf_dataset():
    """Verify dataset loading from processed JSONL files."""
    ds = load_hf_dataset("data/processed")
    assert "train" in ds
    assert "valid" in ds
    assert "test" in ds
    assert len(ds["train"]) > 0
    first_record = ds["train"][0]
    assert "messages" in first_record
    assert len(first_record["messages"]) >= 3

    with pytest.raises(FileNotFoundError):
        load_hf_dataset("data/nonexistent_dir")


def test_run_training_dry_run(capsys):
    """Verify run_training runs cleanly in dry-run mode."""
    run_training(config_path="configs/hf_train_config.yaml", dry_run=True)
    captured = capsys.readouterr()
    assert "Starting CUDA QLoRA training" in captured.out
    assert "Qwen/Qwen2.5-Coder-7B-Instruct" in captured.out
    assert "Dry run validation completed successfully" in captured.out


def test_colab_notebook_structure():
    """Verify Google Colab notebook exists, is valid JSON, and contains required cells."""
    nb_path = Path("notebooks/colab_qlora_sql.ipynb")
    assert nb_path.exists(), "notebooks/colab_qlora_sql.ipynb must exist"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert nb.get("nbformat") == 4
    assert "cells" in nb
    cells = nb["cells"]
    assert len(cells) >= 5

    all_sources = "".join("".join(c.get("source", [])) for c in cells)
    assert "!pip install" in all_sources
    assert "BitsAndBytesConfig" in all_sources
    assert "SFTTrainer" in all_sources
    assert "sqlglot" in all_sources
    assert "sqlite3" in all_sources
