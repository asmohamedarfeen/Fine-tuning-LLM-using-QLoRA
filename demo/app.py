"""
Gradio Web UI for Enterprise Text-to-SQL Demo.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from eval.evaluate_sql import validate_syntax
from eval.sandbox_db import SQLiteSandbox

SAMPLE_SCHEMA = """CREATE TABLE employees (
    emp_id INT PRIMARY KEY,
    name VARCHAR(100),
    department VARCHAR(50),
    salary NUMERIC(10, 2)
);"""


def generate_and_run(schema: str, question: str) -> tuple[str, str, str]:
    """
    Simulates or executes query generation and runs it in the SQLite sandbox.

    Args:
        schema: Database DDL schema.
        question: Natural language question.

    Returns:
        Tuple of (generated_sql, syntax_status, result_string).
    """
    is_valid_schema, schema_err = validate_syntax(schema)
    simulated_sql = "SELECT department, AVG(salary) AS avg_salary FROM employees GROUP BY department;"

    if not is_valid_schema:
        return (
            simulated_sql,
            f"Error: {schema_err}",
            f"Schema Validation Error: {schema_err}",
        )

    try:
        sandbox = SQLiteSandbox(schema)
        if "employees" in schema.lower():
            sandbox.execute(
                "INSERT INTO employees VALUES (1, 'Alice', 'Engineering', 120000), "
                "(2, 'Bob', 'HR', 80000);"
            )
        ok, rows, exec_err = sandbox.execute(simulated_sql)
        sandbox.close()

        result_str = str(rows) if ok else f"Execution Error: {exec_err}"
        status_str = "Valid" if ok else f"Error: {exec_err}"
        return simulated_sql, status_str, result_str
    except (sqlite3.Error, Exception) as e:  # noqa: BLE001
        return simulated_sql, f"Error: {e}", f"Execution Error: {e}"


def create_demo() -> Any:
    """
    Builds the Gradio Blocks demo application.

    Returns:
        Gradio Blocks instance.
    """
    import gradio as gr

    with gr.Blocks(title="Enterprise Text-to-SQL Engine") as demo:
        gr.Markdown("# 🚀 Enterprise Text-to-SQL QLoRA Engine")
        with gr.Row():
            schema_input = gr.Textbox(
                lines=6, label="Database DDL Schema", value=SAMPLE_SCHEMA
            )
            query_input = gr.Textbox(
                lines=2,
                label="Natural Language Question",
                value="Show average salary per department",
            )
        generate_btn = gr.Button("Generate & Run SQL")
        with gr.Row():
            sql_output = gr.Code(label="Generated SQL", language="sql")
            syntax_output = gr.Textbox(label="Syntax Status")
        table_output = gr.Textbox(label="Sandbox DB Results")
        generate_btn.click(
            generate_and_run,
            inputs=[schema_input, query_input],
            outputs=[sql_output, syntax_output, table_output],
        )
    return demo


if __name__ == "__main__":
    try:
        demo = create_demo()
        demo.launch()
    except ImportError:
        print("[!] Gradio not installed. Install with: pip install gradio")
