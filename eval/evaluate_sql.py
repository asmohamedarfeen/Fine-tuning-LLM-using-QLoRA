"""
Evaluation metrics: Valid Syntax Rate (VSR) and Execution Accuracy (EX).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import sqlglot
from sqlglot.errors import SqlglotError

from eval.sandbox_db import SQLiteSandbox


def validate_syntax(sql: str, dialect: str = "sqlite") -> tuple[bool, str | None]:
    """
    Validates SQL query syntax using sqlglot.

    Args:
        sql: SQL query string to validate.
        dialect: SQL dialect name (defaults to 'sqlite').

    Returns:
        Tuple of (is_valid, error_message_or_none).
    """
    if not sql or not sql.strip():
        return False, "Empty SQL query"
    try:
        sqlglot.transpile(sql, read=dialect)
        return True, None
    except (SqlglotError, ValueError, Exception) as e:  # noqa: BLE001
        return False, str(e)


def execute_query(
    ddl: str, sql: str
) -> tuple[bool, list[tuple[Any, ...]] | None, str | None]:
    """
    Execute a query against a temporary SQLite sandbox created from DDL schema.

    Args:
        ddl: SQL Data Definition Language string.
        sql: SQL query statement.

    Returns:
        Tuple of (success, rows_or_none, error_or_none).
    """
    with SQLiteSandbox(ddl) as sandbox:
        return sandbox.execute(sql)


def evaluate_single_sample(
    ddl: str, gen_sql: str, gold_sql: str
) -> dict[str, bool | str | None]:
    """
    Evaluates a single generated SQL query against gold reference SQL and DDL schema.

    Args:
        ddl: Schema DDL statement to initialize the database.
        gen_sql: Generated SQL query string.
        gold_sql: Ground-truth reference SQL query string.

    Returns:
        Dictionary containing 'valid_syntax', 'execution_match', and 'error'.
    """
    is_valid, syntax_err = validate_syntax(gen_sql)
    if not is_valid:
        return {
            "valid_syntax": False,
            "execution_match": False,
            "error": f"Syntax Error: {syntax_err}",
        }

    sandbox = SQLiteSandbox(ddl)
    try:
        # Execute gold query
        gold_ok, gold_rows, gold_err = sandbox.execute(gold_sql)
        if not gold_ok:
            return {
                "valid_syntax": True,
                "execution_match": False,
                "error": f"Gold SQL failed: {gold_err}",
            }

        # Execute generated query
        gen_ok, gen_rows, gen_err = sandbox.execute(gen_sql)
        if not gen_ok:
            return {
                "valid_syntax": True,
                "execution_match": False,
                "error": f"Generated SQL execution failed: {gen_err}",
            }

        # Order-agnostic set comparison of rows
        match = False
        if gold_rows is not None and gen_rows is not None:
            if gold_rows == gen_rows:
                match = True
            else:
                try:
                    match = sorted(gold_rows) == sorted(gen_rows)
                except TypeError:
                    match = Counter(gold_rows) == Counter(gen_rows)

        return {
            "valid_syntax": True,
            "execution_match": match,
            "error": None,
        }
    finally:
        sandbox.close()


def compare_execution(ddl: str, gen_sql: str, gold_sql: str) -> bool:
    """
    Execute both queries against the sandbox and return True if results match.

    Args:
        ddl: Schema DDL.
        gen_sql: Generated SQL query string.
        gold_sql: Ground-truth reference SQL query string.

    Returns:
        True if both queries execute successfully and produce equivalent rows.
    """
    res = evaluate_single_sample(ddl, gen_sql, gold_sql)
    return bool(res.get("execution_match", False))


def compute_corpus_metrics(
    results: list[dict[str, Any]],
) -> dict[str, float | int]:
    """
    Compute Valid Syntax Rate (VSR) and Execution Accuracy (EX) across corpus evaluation results.

    Args:
        results: List of evaluation result dicts from evaluate_single_sample.

    Returns:
        Dictionary with valid_syntax_rate (0-100), execution_accuracy (0-100), and total count.
    """
    total = len(results)
    if total == 0:
        return {"valid_syntax_rate": 0.0, "execution_accuracy": 0.0, "total": 0}

    valid_count = sum(1 for r in results if r.get("valid_syntax"))
    match_count = sum(1 for r in results if r.get("execution_match"))

    return {
        "valid_syntax_rate": round((valid_count / total) * 100, 2),
        "execution_accuracy": round((match_count / total) * 100, 2),
        "total": total,
    }
