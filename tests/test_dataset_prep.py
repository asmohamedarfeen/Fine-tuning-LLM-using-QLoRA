"""
Tests for dataset preparation and ChatML formatting.
"""

from __future__ import annotations

import json
from pathlib import Path

from data.prepare_dataset import (
    SYSTEM_PROMPT,
    build_dataset_from_seed,
    create_splits,
    extract_sql_from_text,
    format_chatml_prompt,
    render_chatml,
)


def test_format_chatml_prompt():
    schema = "CREATE TABLE users (id INT, name TEXT);"
    question = "Show all users"
    reasoning = "Query all records from users table."
    sql = "SELECT id, name FROM users;"

    formatted = format_chatml_prompt(schema, question, reasoning, sql)

    assert "messages" in formatted
    assert len(formatted["messages"]) == 3
    assert formatted["messages"][0]["role"] == "system"
    assert formatted["messages"][0]["content"] == SYSTEM_PROMPT
    assert formatted["messages"][1]["role"] == "user"
    assert "### Database Schema:" in formatted["messages"][1]["content"]
    assert (
        "CREATE TABLE users (id INT, name TEXT);" in formatted["messages"][1]["content"]
    )
    assert "### Question:\nShow all users" in formatted["messages"][1]["content"]
    assert formatted["messages"][2]["role"] == "assistant"
    assert "Query all records from users table." in formatted["messages"][2]["content"]
    assert "SELECT id, name FROM users;" in formatted["messages"][2]["content"]
    assert (
        "```sql\nSELECT id, name FROM users;\n```"
        in formatted["messages"][2]["content"]
    )


def test_render_chatml():
    schema = "CREATE TABLE test (id INT);"
    question = "Get IDs"
    reasoning = "Select id."
    sql = "SELECT id FROM test;"

    formatted = format_chatml_prompt(schema, question, reasoning, sql)
    raw_chatml = render_chatml(formatted["messages"])

    assert raw_chatml.startswith("<|im_start|>system\n")
    assert "<|im_end|>\n<|im_start|>user\n" in raw_chatml
    assert "<|im_end|>\n<|im_start|>assistant\n" in raw_chatml
    assert raw_chatml.strip().endswith("<|im_end|>")


def test_extract_sql_from_text():
    sample_text = "Here is the query:\n```sql\nSELECT * FROM test;\n```\nHope it helps."
    extracted = extract_sql_from_text(sample_text)
    assert extracted.strip() == "SELECT * FROM test;"

    # Uppercase block test
    sample_upper = "Query:\n```SQL\nSELECT col FROM tbl;\n```"
    assert extract_sql_from_text(sample_upper) == "SELECT col FROM tbl;"

    # Plain text without block test
    sample_plain = "SELECT a, b FROM table_x;"
    assert extract_sql_from_text(sample_plain) == "SELECT a, b FROM table_x;"


def test_create_splits():
    items = list(range(100))
    train, val, test = create_splits(
        items, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    )
    assert len(train) == 80
    assert len(val) == 10
    assert len(test) == 10
    assert train == list(range(80))
    assert val == list(range(80, 90))
    assert test == list(range(90, 100))


def test_build_dataset_from_seed(tmp_path: Path):
    seed_data = [
        {
            "domain": "test_domain",
            "ddl": "CREATE TABLE mock (id INT);",
            "samples": [
                {
                    "question": f"Question {i}",
                    "reasoning": f"Reasoning {i}",
                    "sql": f"SELECT {i} FROM mock;",
                }
                for i in range(10)
            ],
        }
    ]
    seed_file = tmp_path / "mock_schemas.json"
    with open(seed_file, "w", encoding="utf-8") as f:
        json.dump(seed_data, f)

    output_dir = tmp_path / "processed"
    build_dataset_from_seed(
        seed_file, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    )

    train_file = output_dir / "train.jsonl"
    valid_file = output_dir / "valid.jsonl"
    test_file = output_dir / "test.jsonl"

    assert train_file.exists()
    assert valid_file.exists()
    assert test_file.exists()

    with open(train_file, "r", encoding="utf-8") as f:
        train_lines = [json.loads(line) for line in f]
    with open(valid_file, "r", encoding="utf-8") as f:
        valid_lines = [json.loads(line) for line in f]
    with open(test_file, "r", encoding="utf-8") as f:
        test_lines = [json.loads(line) for line in f]

    assert len(train_lines) == 8
    assert len(valid_lines) == 1
    assert len(test_lines) == 1
    assert train_lines[0]["messages"][0]["role"] == "system"
    assert "SELECT 0 FROM mock;" in train_lines[0]["messages"][2]["content"]


def test_sample_schemas_file():
    seed_path = Path("data/sample_schemas.json")
    assert seed_path.exists()

    with open(seed_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) >= 2

    domains = {item["domain"] for item in data}
    assert "ecommerce_sales" in domains
    assert "saas_analytics" in domains

    for domain_info in data:
        assert "domain" in domain_info
        assert "ddl" in domain_info
        assert "samples" in domain_info
        assert len(domain_info["samples"]) > 0
        for sample in domain_info["samples"]:
            assert "question" in sample
            assert "reasoning" in sample
            assert "sql" in sample
