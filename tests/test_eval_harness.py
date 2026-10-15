"""
Tests for SQL evaluation harness: syntax validation and SQLite sandbox execution.
"""

from __future__ import annotations

from eval.evaluate_sql import (
    compare_execution,
    compute_corpus_metrics,
    evaluate_single_sample,
    execute_query,
    validate_syntax,
)
from eval.sandbox_db import SQLiteSandbox


def test_syntax_validation_valid():
    valid_sql = "SELECT id, name FROM users WHERE age > 18 ORDER BY name ASC;"
    is_valid, error = validate_syntax(valid_sql)
    assert is_valid is True
    assert error is None


def test_syntax_validation_invalid():
    invalid_sql = "SELEC FROM WHERE id = ;"
    is_valid, error = validate_syntax(invalid_sql)
    assert is_valid is False
    assert error is not None


def test_syntax_validation_empty():
    is_valid, error = validate_syntax("")
    assert is_valid is False
    assert error == "Empty SQL query"

    is_valid, error = validate_syntax("   ")
    assert is_valid is False
    assert error == "Empty SQL query"


def test_sqlite_sandbox_execution():
    ddl = "CREATE TABLE products (id INT PRIMARY KEY, name TEXT, price REAL);"
    sandbox = SQLiteSandbox(ddl)

    # Insert test data
    success, _, err = sandbox.execute(
        "INSERT INTO products VALUES (1, 'Widget', 19.99), (2, 'Gadget', 29.99);"
    )
    assert success is True
    assert err is None

    success, rows, err = sandbox.execute("SELECT name FROM products WHERE price > 20;")
    assert success is True
    assert rows == [("Gadget",)]
    assert err is None
    sandbox.close()


def test_sqlite_sandbox_context_manager():
    ddl = "CREATE TABLE items (id INT, label TEXT);"
    with SQLiteSandbox(ddl) as sandbox:
        sandbox.execute("INSERT INTO items VALUES (1, 'Apple');")
        ok, rows, err = sandbox.execute("SELECT label FROM items;")
        assert ok is True
        assert rows == [("Apple",)]
        assert err is None


def test_sqlite_sandbox_error_handling():
    ddl = "CREATE TABLE test (val INT);"
    sandbox = SQLiteSandbox(ddl)
    ok, rows, err = sandbox.execute("SELECT non_existent_col FROM test;")
    assert ok is False
    assert rows is None
    assert err is not None
    sandbox.close()


def test_execute_query_helper():
    ddl = "CREATE TABLE users (id INT, score REAL); INSERT INTO users VALUES (1, 95.5);"
    ok, rows, err = execute_query(ddl, "SELECT score FROM users WHERE id = 1;")
    assert ok is True
    assert rows == [(95.5,)]
    assert err is None


def test_compare_execution_match():
    ddl = "CREATE TABLE scores (val INT); INSERT INTO scores VALUES (10), (20);"
    gold_sql = "SELECT val FROM scores WHERE val >= 15;"
    gen_sql = "SELECT val FROM scores WHERE val > 10;"
    assert compare_execution(ddl, gen_sql, gold_sql) is True


def test_compare_execution_mismatch():
    ddl = "CREATE TABLE scores (val INT); INSERT INTO scores VALUES (10), (20);"
    gold_sql = "SELECT val FROM scores WHERE val >= 15;"
    gen_sql = "SELECT val FROM scores WHERE val = 10;"
    assert compare_execution(ddl, gen_sql, gold_sql) is False


def test_evaluate_single_sample_exact_match():
    ddl = "CREATE TABLE employees (id INT, salary INT); INSERT INTO employees VALUES (1, 60000), (2, 40000);"
    gold_sql = "SELECT id FROM employees WHERE salary > 50000;"
    gen_sql = "SELECT id FROM employees WHERE salary > 50000;"

    res = evaluate_single_sample(ddl, gen_sql, gold_sql)
    assert res["valid_syntax"] is True
    assert res["execution_match"] is True
    assert res["error"] is None


def test_evaluate_single_sample_order_agnostic():
    ddl = "CREATE TABLE students (name TEXT); INSERT INTO students VALUES ('Alice'), ('Bob');"
    gold_sql = "SELECT name FROM students ORDER BY name ASC;"
    gen_sql = "SELECT name FROM students ORDER BY name DESC;"

    res = evaluate_single_sample(ddl, gen_sql, gold_sql)
    assert res["valid_syntax"] is True
    assert res["execution_match"] is True
    assert res["error"] is None


def test_evaluate_single_sample_with_none_values():
    ddl = "CREATE TABLE records (id INT, extra TEXT); INSERT INTO records VALUES (1, NULL), (2, 'notes');"
    gold_sql = "SELECT extra FROM records;"
    gen_sql = "SELECT extra FROM records ORDER BY id DESC;"

    res = evaluate_single_sample(ddl, gen_sql, gold_sql)
    assert res["valid_syntax"] is True
    assert res["execution_match"] is True
    assert res["error"] is None


def test_evaluate_single_sample_invalid_syntax():
    ddl = "CREATE TABLE dummy (id INT);"
    gold_sql = "SELECT id FROM dummy;"
    gen_sql = "SELEC FROM WHERE ;"

    res = evaluate_single_sample(ddl, gen_sql, gold_sql)
    assert res["valid_syntax"] is False
    assert res["execution_match"] is False
    assert "Syntax Error:" in str(res["error"])


def test_evaluate_single_sample_gold_error():
    ddl = "CREATE TABLE dummy (id INT);"
    gold_sql = "SELECT non_existent FROM dummy;"
    gen_sql = "SELECT id FROM dummy;"

    res = evaluate_single_sample(ddl, gen_sql, gold_sql)
    assert res["valid_syntax"] is True
    assert res["execution_match"] is False
    assert "Gold SQL failed:" in str(res["error"])


def test_evaluate_single_sample_gen_runtime_error():
    ddl = "CREATE TABLE dummy (id INT);"
    gold_sql = "SELECT id FROM dummy;"
    gen_sql = "SELECT missing_col FROM dummy;"

    res = evaluate_single_sample(ddl, gen_sql, gold_sql)
    assert res["valid_syntax"] is True
    assert res["execution_match"] is False
    assert "Generated SQL execution failed:" in str(res["error"])


def test_evaluate_single_sample_result_mismatch():
    ddl = "CREATE TABLE numbers (val INT); INSERT INTO numbers VALUES (1), (2), (3);"
    gold_sql = "SELECT val FROM numbers WHERE val > 2;"
    gen_sql = "SELECT val FROM numbers WHERE val < 2;"

    res = evaluate_single_sample(ddl, gen_sql, gold_sql)
    assert res["valid_syntax"] is True
    assert res["execution_match"] is False
    assert res["error"] is None


def test_compute_corpus_metrics():
    # Empty case
    empty_metrics = compute_corpus_metrics([])
    assert empty_metrics == {
        "valid_syntax_rate": 0.0,
        "execution_accuracy": 0.0,
        "total": 0,
    }

    # Populated case
    sample_results = [
        {"valid_syntax": True, "execution_match": True},
        {"valid_syntax": True, "execution_match": False},
        {"valid_syntax": False, "execution_match": False},
        {"valid_syntax": True, "execution_match": True},
    ]
    metrics = compute_corpus_metrics(sample_results)
    assert metrics["total"] == 4
    assert metrics["valid_syntax_rate"] == 75.0
    assert metrics["execution_accuracy"] == 50.0
