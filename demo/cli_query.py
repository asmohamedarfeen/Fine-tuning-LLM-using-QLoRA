"""
Interactive CLI for local SQL query generation and testing.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from data.prepare_dataset import SYSTEM_PROMPT


def build_inference_prompt(schema: str, question: str) -> str:
    """
    Builds a formatted ChatML inference prompt for Qwen2.5-Coder.

    Args:
        schema: Database DDL schema string.
        question: Natural language question.

    Returns:
        Formatted prompt string with ChatML special tokens.
    """
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n### Database Schema:\n{schema.strip()}\n\n"
        f"### Question:\n{question.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def main(argv: Sequence[str] | None = None) -> None:
    """
    Runs the interactive CLI for Text-to-SQL prompt formatting.

    Args:
        argv: Optional command line arguments. Defaults to sys.argv[1:].
    """
    parser = argparse.ArgumentParser(
        description="Enterprise Text-to-SQL Interactive CLI"
    )
    parser.add_argument("--schema", type=str, default=None, help="Database DDL schema")
    parser.add_argument(
        "--question", type=str, default=None, help="Natural language question"
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    print("=" * 60)
    print(" Enterprise Text-to-SQL Interactive CLI")
    print("=" * 60)

    schema = args.schema
    if schema is None:
        schema = input("Enter Schema (or press enter for default):\n")
    if not schema.strip():
        schema = "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(100), active BOOLEAN);"

    question = args.question
    if question is None:
        question = input("\nEnter your question in plain English:\n")

    prompt = build_inference_prompt(schema, question)
    print("\n[Prompt Formatted for Qwen2.5-Coder]:")
    print(prompt)


if __name__ == "__main__":
    main()
