"""
Companion NVIDIA CUDA QLoRA Training Script using Hugging Face TRL & bitsandbytes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "configs/hf_train_config.yaml") -> dict[str, Any]:
    """Loads YAML training configuration file.

    Args:
        path: Path to the configuration YAML file.

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


def get_bnb_config() -> Any | None:
    """Generates 4-bit NF4 configuration for bitsandbytes if available.

    Dynamically chooses bfloat16 for Ampere/Hopper/Ada GPUs, or float16 for
    Turing / Volta / Tesla T4 architectures.

    Returns:
        BitsAndBytesConfig if torch, transformers, and bitsandbytes are
        available and supported, otherwise None.
    """
    try:
        import torch
        from transformers import BitsAndBytesConfig

        has_cuda = torch.cuda.is_available()
        use_bf16 = bool(has_cuda and torch.cuda.is_bf16_supported())
        compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    except (ImportError, RuntimeError, ValueError):
        return None


def get_peft_config(cfg: dict[str, Any]) -> Any | None:
    """Creates PEFT LoraConfig for target model.

    Args:
        cfg: Configuration dictionary containing LoRA parameters.

    Returns:
        LoraConfig instance if peft is installed, otherwise None.
    """
    try:
        from peft import LoraConfig, TaskType

        target_modules = cfg.get(
            "target_modules",
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        return LoraConfig(
            r=int(cfg.get("lora_r", 16)),
            lora_alpha=int(cfg.get("lora_alpha", 32)),
            lora_dropout=float(cfg.get("lora_dropout", 0.05)),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
        )
    except ImportError:
        return None


def get_training_args(cfg: dict[str, Any]) -> Any | None:
    """Constructs SFTConfig / TrainingArguments from configuration dictionary.

    Dynamically selects BF16 on supported GPUs or FP16 for T4/Turing GPUs.

    Args:
        cfg: Configuration dictionary with hyperparameters.

    Returns:
        SFTConfig instance if trl is installed, otherwise None.
    """
    try:
        import torch
        from trl import SFTConfig

        has_cuda = torch.cuda.is_available()
        use_bf16 = bool(has_cuda and torch.cuda.is_bf16_supported())
        return SFTConfig(
            output_dir=str(cfg.get("output_dir", "adapters/hf_sql_lora")),
            learning_rate=float(cfg.get("learning_rate", 2.0e-4)),
            per_device_train_batch_size=int(cfg.get("batch_size", 2)),
            per_device_eval_batch_size=int(cfg.get("batch_size", 2)),
            gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 4)),
            num_train_epochs=int(cfg.get("num_train_epochs", 3)),
            warmup_ratio=float(cfg.get("warmup_ratio", 0.05)),
            logging_steps=int(cfg.get("logging_steps", 10)),
            save_steps=int(cfg.get("save_steps", 200)),
            max_length=int(cfg.get("max_seq_length", 2048)),
            fp16=bool(has_cuda and not use_bf16),
            bf16=use_bf16,
            logging_dir=f"{cfg.get('output_dir', 'adapters/hf_sql_lora')}/logs",
            report_to="none",
            save_total_limit=2,
        )
    except ImportError:
        return None


def load_hf_dataset(dataset_path: str | Path = "data/processed") -> Any:
    """Loads processed JSONL splits into Hugging Face DatasetDict.

    Args:
        dataset_path: Directory containing train.jsonl, valid.jsonl, test.jsonl.

    Returns:
        DatasetDict containing train, validation, and test splits.

    Raises:
        FileNotFoundError: If the dataset directory or required files do not exist.
    """
    from datasets import load_dataset

    base_path = Path(dataset_path)
    if not base_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {base_path}")

    data_files: dict[str, str] = {}
    for split_name in ("train", "valid", "test"):
        file_path = base_path / f"{split_name}.jsonl"
        if file_path.exists():
            data_files[split_name] = str(file_path)

    if not data_files:
        raise FileNotFoundError(f"No JSONL dataset files found in: {base_path}")

    return load_dataset("json", data_files=data_files)


def run_training(
    config_path: str | Path = "configs/hf_train_config.yaml",
    dry_run: bool = False,
) -> None:
    """Runs or simulates QLoRA training on NVIDIA CUDA with Hugging Face and bitsandbytes.

    Args:
        config_path: Path to YAML training configuration.
        dry_run: If True, validates configuration and inputs without allocating GPU weights.
    """
    cfg = load_config(config_path)

    print(f"[*] Starting CUDA QLoRA training for {cfg['model_id']}")
    print(f"[*] Target directory: {cfg['output_dir']}")
    print("[*] Note: Requires an NVIDIA GPU environment (Colab T4/A10G or AWS EC2).")

    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    if dry_run or not has_cuda:
        print(
            "[*] Running in dry-run/validation mode (CUDA device not detected or dry_run enabled)."
        )
        bnb_cfg = get_bnb_config()
        print(f"[*] BitsAndBytesConfig available: {bnb_cfg is not None}")
        peft_cfg = get_peft_config(cfg)
        print(f"[*] LoraConfig initialized: {peft_cfg is not None}")

        ds_path = Path(cfg.get("dataset_path", "data/processed"))
        if ds_path.exists():
            print(f"[*] Dataset verified at: {ds_path}")
        else:
            print(f"[!] Warning: Dataset path not found: {ds_path}")
        print("[*] Dry run validation completed successfully.")
        return

    # CUDA hardware execution flow
    from peft import prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer

    bnb_config = get_bnb_config()
    peft_config = get_peft_config(cfg)
    training_args = get_training_args(cfg)

    use_bf16 = bool(has_cuda and torch.cuda.is_bf16_supported())
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    print(f"[*] Loading tokenizer for {cfg['model_id']}...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_id"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(
        f"[*] Loading 4-bit base model for {cfg['model_id']} (compute_dtype={compute_dtype})..."
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )
    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)

    dataset = load_hf_dataset(cfg.get("dataset_path", "data/processed"))

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("valid"),
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
    )

    print("[*] Starting SFTTrainer execution loop...")
    trainer.train()

    print(f"[*] Saving fine-tuned adapter to {cfg['output_dir']}...")
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print("[+] Training completed successfully.")


def main() -> None:
    """CLI entrypoint for CUDA QLoRA training."""
    parser = argparse.ArgumentParser(
        description="NVIDIA CUDA Hugging Face QLoRA Training Runner"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/hf_train_config.yaml",
        help="Path to YAML training configuration file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and inputs without starting GPU training",
    )
    args = parser.parse_args()
    run_training(config_path=args.config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
