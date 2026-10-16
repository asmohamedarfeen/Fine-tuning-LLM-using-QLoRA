# ⚡ Enterprise Text-to-SQL Engine: 4-Bit QLoRA on Apple Silicon & CUDA

An end-to-end LLM fine-tuning and evaluation suite demonstrating domain adaptation of **`Qwen/Qwen2.5-Coder-7B-Instruct`** using 4-bit Quantized Low-Rank Adaptation (QLoRA).

Featuring a **dual-track architecture**:
1. **Apple Silicon M4 Native Engine:** 100% local, zero-CUDA fine-tuning using Apple's official `mlx-lm` framework within 24GB Unified Memory.
2. **Companion Cloud Engine:** Industry-standard Hugging Face `peft` + `bitsandbytes` (NF4) + `trl` pipeline for NVIDIA GPU environments (Google Colab / AWS).

---

## 🚀 Project Implementation Progress

- [x] **Task 1: Environment Configuration & Hardware Setup** (Completed)
  - Modular environment configuration (`requirements.txt`, `requirements-mac.txt`, `requirements-cuda.txt`).
  - Apple Silicon M4 runtime verification test (`scripts/verify_env.py`).
- [x] **Task 2: Enterprise Schema & Dataset Pipeline** (Completed)
  - Multi-turn ChatML prompt structure (`<|im_start|>...<|im_end|>`) for `Qwen2.5-Coder`.
  - Realistic multi-table enterprise relational DDL schemas (E-Commerce, SaaS, Financial, Healthcare).
  - Deterministic train/validation/test dataset partitioning (`data/prepare_dataset.py`).
- [x] **Task 3: Evaluation Suite — Syntax Parser & SQLite In-Memory Sandbox** (Completed)
  - SQL dialect syntax validator via `sqlglot`.
  - Execution Accuracy (EX) benchmark against isolated SQLite sandbox.
- [x] **Task 4: Apple Silicon MLX QLoRA Engine & Hyperparameter Configs** (Completed)
  - Native Metal Performance Shaders 4-bit LoRA training configuration (`configs/mlx_train_config.yaml`).
  - Command generator & runner for `mlx_lm.lora` (`mlx_engine/train_mlx.py`).
  - LoRA adapter fusion tool (`mlx_engine/fuse_mlx.py`).
  - GGUF export instructions for Ollama and LM Studio (`mlx_engine/export_gguf.py`).
- [ ] **Task 5: Companion NVIDIA CUDA Engine & Colab Notebook**
  - Hugging Face `SFTTrainer` with bitsandbytes NF4 quantization.
- [ ] **Task 6: Interactive Demo & Deployment Artifacts**
  - Gradio Web UI, interactive terminal CLI, and Ollama `Modelfile`.
- [ ] **Task 7: Comprehensive Documentation & Benchmark Report**

---

## 🛠️ Quickstart

### 1. Apple Silicon M4 Local Setup
```bash
# Clone and setup
pip install -r requirements-mac.txt

# Verify hardware & dependencies
python scripts/verify_env.py
pytest tests/test_environment.py -v
```

### 2. Apple Silicon MLX QLoRA Training & Export
```bash
# Display / generate MLX training CLI command
python mlx_engine/train_mlx.py

# Fuse trained LoRA adapter weights
python mlx_engine/fuse_mlx.py

# Display GGUF export instructions
python mlx_engine/export_gguf.py
```

---

*Author: Mohamed Arfeen | Portfolio Project for Advanced GenAI / Fine-Tuning*
