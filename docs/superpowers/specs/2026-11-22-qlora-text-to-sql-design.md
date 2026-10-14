# Technical Specification: Enterprise Text-to-SQL QLoRA Engine (Dual Apple MLX & CUDA PEFT)

## 1. Project Overview & Objectives

* **Project Name:** Enterprise Text-to-SQL Engine (`sql-craft-qlora`)
* **Core Goal:** Fine-tune `Qwen/Qwen2.5-Coder-7B-Instruct` using 4-bit QLoRA to translate complex natural language questions into syntactically valid and executable SQL queries across enterprise relational schemas.
* **Dual-Track Architecture:**
  1. **Primary Local Desktop Engine:** 100% native Apple Silicon M4 (24GB Unified Memory) execution using Apple's official `mlx-lm` framework with 4-bit quantized Metal kernels.
  2. **Companion Cloud/CUDA Engine:** A standalone, fully reproducible Hugging Face `peft` + `bitsandbytes` + `trl` (`SFTTrainer`) script designed for NVIDIA environments (Google Colab / RunPod / AWS).
* **Portfolio Impact:**
  * Demonstrates mastery of both hardware-native optimization (Apple Metal/MLX) and standard enterprise cloud ML tooling (HF/CUDA).
  * Objective, quantitative evaluation suite: SQL Syntax Validity Rate and Execution Accuracy (EX) comparing Base vs. Fine-Tuned models.
  * Real-world deployment artifact: LoRA adapter fusion, GGUF export for local Ollama/vLLM inference, and an interactive demo interface.

---

## 2. Hardware Budget & Memory Allocation (M4 24GB Unified Memory)

On Apple Silicon M4 with 24GB Unified Memory:
* **Base Model (Qwen2.5-Coder-7B 4-bit):** ~4.5 GB - 5.2 GB VRAM
* **LoRA Trainable Parameters ($r=16, \alpha=32$ on all linear layers):** ~120 MB
* **Gradients & Optimizer States (AdamW in MLX):** ~3.5 GB - 4.5 GB
* **Activation Memory (Context Length = 2048 tokens, batch size = 2 with gradient accumulation = 4):** ~3.0 GB - 4.0 GB
* **Total Estimated Training Footprint:** ~12 GB - 14 GB VRAM (well within the 24 GB ceiling, leaving 10+ GB headroom for macOS and system operations).

---

## 3. Data Pipeline & Schema Prompt Engineering

### 3.1 Dataset Selection
* **Primary Source:** High-difficulty subset of **BIRD-SQL** (cross-domain database benchmark) combined with synthetic enterprise data schemas (e.g. Sales, SaaS Analytics, Financial Ledger).
* **Format:** Multi-turn ChatML prompt structure optimized for `Qwen2.5-Coder-Instruct`.

### 3.2 Prompt Template
```markdown
<|im_start|>system
You are an expert enterprise SQL database engineer. Given the database schema (DDL), constraints, and natural language question, generate the most efficient, syntactically correct SQL query. Always explain your reasoning briefly before outputting the final SQL block enclosed in ```sql ... ```.
<|im_end|>
<|im_start|>user
### Database Schema:
CREATE TABLE customers (
    customer_id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL
);

CREATE TABLE orders (
    order_id VARCHAR(36) PRIMARY KEY,
    customer_id VARCHAR(36) REFERENCES customers(customer_id),
    total_amount NUMERIC(10, 2) NOT NULL,
    status VARCHAR(50) NOT NULL,
    order_date DATE NOT NULL
);

### Question:
Find the top 3 customers by total spend on completed orders in 2024.
<|im_end|>
<|im_start|>assistant
To find the top 3 customers by spend in 2024:
1. Filter orders where `status = 'completed'` and `order_date` falls in year 2024.
2. Join `orders` with `customers` on `customer_id`.
3. Group by customer and compute `SUM(total_amount)`.
4. Order descending by total spend and limit to 3.

```sql
SELECT 
    c.customer_id,
    c.name,
    SUM(o.total_amount) AS total_spend
FROM customers c
INNER JOIN orders o ON c.customer_id = o.customer_id
WHERE o.status = 'completed'
  AND o.order_date >= '2024-01-01' AND o.order_date <= '2024-12-31'
GROUP BY c.customer_id, c.name
ORDER BY total_spend DESC
LIMIT 3;
```
<|im_end|>
```

### 3.3 Splits & Processing
* **Train:** 80% (~4,000 - 8,000 curated samples)
* **Validation:** 10% (for loss evaluation and checkpoint saving)
* **Test:** 10% (held-out unseen schemas for objective evaluation)

---

## 4. Model Architecture & Hyperparameters

### 4.1 Base Model
* **Model ID:** `Qwen/Qwen2.5-Coder-7B-Instruct`
* **Context Window:** 2048 (balanced for schema length and M4 memory speed)

### 4.2 LoRA Configuration
* **LoRA Rank ($r$):** 16 (or 32 for higher representational capacity)
* **LoRA Alpha ($\alpha$):** 32
* **LoRA Dropout:** 0.05
* **Target Linear Projections:**
  * Self-Attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`
  * MLP / Feed-Forward: `gate_proj`, `up_proj`, `down_proj`

### 4.3 Training Parameters
* **Quantization:** 4-bit (Group size 64 for MLX, NormalFloat4 for bitsandbytes)
* **Optimizer:** AdamW ($\beta_1=0.9, \beta_2=0.999, \epsilon=1e-8$)
* **Learning Rate:** $1e-4$ with Cosine Annealing decay
* **Warmup:** 5% of total steps
* **Batch Size:** 2 per step
* **Gradient Accumulation Steps:** 4 (Effective batch size = 8)
* **Epochs:** 3
* **Checkpoint Frequency:** Every 200 steps (saving best validation loss)

---

## 5. Dual Implementation Workflows

### 5.1 Track A: Apple Silicon M4 Desktop Engine (`mlx_engine/`)
* **Tool:** `mlx-lm` CLI and Python API
* **Key Commands:**
  1. **Quantization & Preparation:**
     `python -m mlx_lm.convert --hf-path Qwen/Qwen2.5-Coder-7B-Instruct -q --q-bits 4 -o ./models/qwen2.5-7b-mlx-4bit`
  2. **LoRA Fine-Tuning:**
     `python -m mlx_lm.lora --model ./models/qwen2.5-7b-mlx-4bit --train --data ./data/mlx_format --batch-size 2 --lora-layers 16 --iters 1000 --learning-rate 1e-4 --adapter-path ./adapters/mlx_sql_lora`
  3. **Adapter Fusion:**
     `python -m mlx_lm.fuse --model ./models/qwen2.5-7b-mlx-4bit --adapter-path ./adapters/mlx_sql_lora --save-path ./models/qwen2.5-7b-sql-fused`
  4. **GGUF Export:** Convert fused weights to `.gguf` for instant Ollama and local deployment.

### 5.2 Track B: Companion Cloud CUDA Engine (`cuda_engine/`)
* **Tool:** Hugging Face `transformers`, `peft`, `bitsandbytes`, `trl` (`SFTTrainer`)
* **Features:**
  * `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)`
  * Google Colab / Kaggle / RunPod single-GPU execution script (`train_cuda.py`).
  * Ready-to-run Jupyter notebook (`notebooks/colab_qlora_sql.ipynb`).

---

## 6. Evaluation & Benchmarking Suite (Portfolio Differentiator)

To make your portfolio stand out to recruiters, the repo includes an automated evaluation harness (`eval/evaluate_sql.py`):
1. **Valid SQL Rate (VSR):**
   * Uses `sqlglot` parser to verify syntax accuracy against target dialect (PostgreSQL / SQLite).
2. **Execution Accuracy (EX):**
   * Spawns an in-memory SQLite sandbox database with populated dummy data.
   * Executes the Generated SQL vs. Gold SQL.
   * Checks whether the output result matrices match exactly ($EX = 1$ if identical, $0$ otherwise).
3. **Automated Comparative Benchmark Card:**
   * Automatically prints a markdown table comparing:
     * Base Model (`Qwen2.5-Coder-7B-Instruct`) vs. Fine-Tuned Model.
     * Metrics: Valid Syntax %, Execution Accuracy %, Token Generation Latency (tokens/sec), Memory Footprint.

---

## 7. Interactive Demo & Deployment Artifacts

1. **CLI Playground (`demo/cli_query.py`):**
   * Interactive terminal interface where you paste a table schema, type a question, and see the streaming SQL query.
2. **Web UI (`demo/app.py`):**
   * Lightweight Gradio or Streamlit UI featuring schema presets (E-commerce, SaaS, HR) and live SQL query execution with data table visualization.
3. **Local Ollama Integration:**
   * Includes a `Modelfile` allowing anyone to run your fine-tuned model via `ollama run qwen-sql-craft`.

---

## 8. Repository Structure

```
sql-craft-qlora/
├── README.md                          # Comprehensive portfolio documentation, benchmark tables, and architecture diagram
├── requirements.txt                   # Universal dependencies
├── requirements-mac.txt               # Apple MLX dependencies
├── requirements-cuda.txt              # PyTorch + CUDA dependencies
│
├── configs/
│   ├── mlx_train_config.yaml          # MLX training hyperparameters
│   └── hf_train_config.yaml           # Hugging Face TRL hyperparameters
│
├── data/
│   ├── prepare_dataset.py             # Downloads, formats, and validates ChatML data
│   ├── sample_schemas.json            # Reference enterprise schemas
│   └── raw/                           # Raw input data
│
├── mlx_engine/                        # Native Apple Silicon track
│   ├── train_mlx.py                   # Custom MLX training script wrapper
│   ├── fuse_mlx.py                    # Adapter merging utility
│   └── export_gguf.py                 # GGUF export helper
│
├── cuda_engine/                       # NVIDIA CUDA companion track
│   ├── train_cuda.py                  # Standard HF SFTTrainer + bitsandbytes script
│   └── export_hf.py                   # Hugging Face hub uploader / merger
│
├── eval/
│   ├── evaluate_sql.py                # Syntax parser & Execution Accuracy harness
│   ├── sandbox_db.py                  # In-memory SQLite runner
│   └── benchmark_results.json         # Persisted metrics
│
├── demo/
│   ├── app.py                         # Web UI (Gradio)
│   ├── cli_query.py                   # Terminal interactive runner
│   └── Modelfile                      # Ollama local deployment config
│
└── docs/
    └── superpowers/specs/
        └── 2026-11-22-qlora-text-to-sql-design.md
```

---

## 9. Verification & Success Criteria

1. **Local Training Verification:**
   * Script successfully runs on Apple M4 using MLX without memory spikes or crashes.
   * Training loss decreases smoothly over 1,000+ iterations.
2. **Quantization & Adapter Quality:**
   * Saved LoRA weights are properly structured and successfully fuse back into base weights.
3. **Eval Gate:**
   * Fine-tuned model achieves **>85% Valid SQL rate** and measurable improvement in Execution Accuracy over the base model on held-out test schemas.
4. **Reproducibility:**
   * Clear step-by-step documentation allowing any reviewer/recruiter to run either local M4 inference or Google Colab training with 1 click.
