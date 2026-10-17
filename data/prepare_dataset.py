"""
Prepares enterprise text-to-SQL datasets in ChatML format.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = (
    "You are an expert enterprise SQL database engineer. Given the database schema (DDL), "
    "constraints, and natural language question, generate the most efficient, syntactically "
    "correct SQL query. Always explain your reasoning briefly before outputting the final SQL "
    "block enclosed in ```sql ... ```."
)


def format_chatml_prompt(
    schema: str,
    question: str,
    reasoning: str,
    sql: str,
) -> dict[str, list[dict[str, str]]]:
    """
    Formats schema, question, reasoning, and SQL into ChatML messages format.

    Args:
        schema: DDL schema definition.
        question: Natural language question.
        reasoning: Brief analytical reasoning or query plan.
        sql: Target SQL query.

    Returns:
        Dictionary containing list of ChatML messages.
    """
    user_content = (
        f"### Database Schema:\n{schema.strip()}\n\n### Question:\n{question.strip()}"
    )
    assistant_content = f"{reasoning.strip()}\n\n```sql\n{sql.strip()}\n```"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def render_chatml(messages: list[dict[str, str]]) -> str:
    """
    Renders a list of ChatML messages into raw text with ChatML tokens.

    Format:
        <|im_start|>role
        content<|im_end|>

    Args:
        messages: List of message dictionaries containing 'role' and 'content'.

    Returns:
        Rendered string with ChatML delimiters.
    """
    rendered = []
    for msg in messages:
        rendered.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
    return "\n".join(rendered) + "\n"


def extract_sql_from_text(text: str) -> str:
    """
    Extracts SQL code block from markdown ```sql ... ``` if present, else returns text.

    Args:
        text: Response text or raw SQL string.

    Returns:
        Extracted SQL query string.
    """
    match = re.search(r"```sql\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def create_splits(
    samples: list[Any],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = False,
    seed: int | None = 42,
) -> tuple[list[Any], list[Any], list[Any]]:
    """
    Splits samples into train, validation, and test partitions.

    Args:
        samples: List of data samples.
        train_ratio: Proportion of samples allocated to training set.
        val_ratio: Proportion of samples allocated to validation set.
        test_ratio: Proportion of samples allocated to testing set.
        shuffle: Whether to randomly shuffle samples before partitioning.
        seed: Random seed for deterministic shuffling.

    Returns:
        Tuple of (train_samples, val_samples, test_samples).
    """
    sample_pool = list(samples)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(sample_pool)

    total = len(sample_pool)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)

    train = sample_pool[:n_train]
    val = sample_pool[n_train : n_train + n_val]
    test = sample_pool[n_train + n_val :]
    return train, val, test


def build_dataset_from_seed(
    seed_path: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True,
    seed: int | None = 42,
) -> None:
    """
    Loads seed schemas, formats ChatML prompts, splits, and writes JSONL files.

    Args:
        seed_path: Path to sample schemas JSON file.
        output_dir: Destination directory for train.jsonl, valid.jsonl, test.jsonl.
        train_ratio: Ratio for train partition.
        val_ratio: Ratio for validation partition.
        test_ratio: Ratio for test partition.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(seed_path, "r", encoding="utf-8") as f:
        domains = json.load(f)

    records: list[dict[str, Any]] = []
    for domain_info in domains:
        ddl = domain_info["ddl"]
        for sample in domain_info.get("samples", []):
            record = format_chatml_prompt(
                schema=ddl,
                question=sample["question"],
                reasoning=sample["reasoning"],
                sql=sample["sql"],
            )
            records.append(record)

    train, val, test = create_splits(
        records,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        shuffle=shuffle,
        seed=seed,
    )

    for split_name, split_data in [("train", train), ("valid", val), ("test", test)]:
        split_file = output_path / f"{split_name}.jsonl"
        with open(split_file, "w", encoding="utf-8") as out:
            out.writelines(json.dumps(item) + "\n" for item in split_data)
        print(f"[+] Wrote {len(split_data)} records to {split_file}")


if __name__ == "__main__":
    build_dataset_from_seed("data/sample_schemas.json", "data/processed")
