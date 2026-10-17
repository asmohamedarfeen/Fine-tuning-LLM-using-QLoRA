"""
Tests for interactive demo tools, CLI, and Gradio application.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from demo.app import SAMPLE_SCHEMA, create_demo, generate_and_run
from demo.cli_query import build_inference_prompt
from demo.cli_query import main as cli_main


def test_build_inference_prompt() -> None:
    """Test standard inference prompt formatting."""
    schema = "CREATE TABLE test (id INT);"
    question = "Count tests"
    prompt = build_inference_prompt(schema, question)
    assert "<|im_start|>system" in prompt
    assert "<|im_start|>user" in prompt
    assert "CREATE TABLE test" in prompt
    assert "Count tests" in prompt
    assert "<|im_start|>assistant" in prompt


def test_cli_query_main_interactive(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test interactive execution of CLI main function with inputs."""
    inputs = iter(["CREATE TABLE items (id INT);", "Show all items"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    cli_main([])
    captured = capsys.readouterr().out
    assert "Enterprise Text-to-SQL Interactive CLI" in captured
    assert "CREATE TABLE items (id INT);" in captured
    assert "Show all items" in captured


def test_cli_query_main_defaults(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test CLI main fallback to default schema when empty string entered."""
    inputs = iter(["", "Count active users"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    cli_main([])
    captured = capsys.readouterr().out
    assert "CREATE TABLE users" in captured
    assert "Count active users" in captured


def test_cli_query_main_args(capsys: pytest.CaptureFixture[str]) -> None:
    """Test CLI main with command line arguments."""
    cli_main(
        [
            "--schema",
            "CREATE TABLE orders (order_id INT);",
            "--question",
            "Find order 101",
        ]
    )
    captured = capsys.readouterr().out
    assert "CREATE TABLE orders" in captured
    assert "Find order 101" in captured


def test_generate_and_run_success() -> None:
    """Test simulated SQL generation and execution in SQLite sandbox."""
    simulated_sql, status, result = generate_and_run(
        SAMPLE_SCHEMA, "Show average salary per department"
    )
    assert "SELECT department, AVG(salary)" in simulated_sql
    assert status == "Valid"
    assert "Engineering" in result
    assert "120000" in result or "120000.0" in result


def test_generate_and_run_invalid_schema() -> None:
    """Test handling of invalid schema DDL."""
    _simulated_sql, status, result = generate_and_run(
        "INVALID SYNTAX TABLE ???", "Show all items"
    )
    assert "Error" in status
    assert "Error" in result


def test_create_demo() -> None:
    """Test Gradio Blocks demo creation."""
    demo = create_demo()
    assert demo is not None
    assert demo.title == "Enterprise Text-to-SQL Engine"


def test_modelfile_contents() -> None:
    """Verify Ollama Modelfile exists and contains required directives."""
    modelfile_path = Path(__file__).resolve().parent.parent / "demo" / "Modelfile"
    assert modelfile_path.exists(), f"Modelfile not found at {modelfile_path}"
    content = modelfile_path.read_text(encoding="utf-8")

    assert "FROM ./models/qwen2.5-7b-sql.gguf" in content
    assert "TEMPLATE" in content
    assert "<|im_start|>system" in content
    assert "<|im_start|>user" in content
    assert "<|im_start|>assistant" in content
    assert "SYSTEM" in content
    assert "PARAMETER temperature 0.2" in content
    assert 'PARAMETER stop "<|im_start|>"' in content
    assert 'PARAMETER stop "<|im_end|>"' in content
