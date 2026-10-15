"""
In-memory SQLite sandbox for deterministic query execution.
"""

from __future__ import annotations

import sqlite3
from types import TracebackType
from typing import Any, Self


class SQLiteSandbox:
    """
    In-memory SQLite sandbox for executing DDL schema setup and SQL queries.
    """

    def __init__(self, ddl: str) -> None:
        """
        Initialize the SQLite sandbox with schema DDL.

        Args:
            ddl: SQL Data Definition Language string (e.g. CREATE TABLE statements).
        """
        self.conn = sqlite3.connect(":memory:")
        self._init_schema(ddl)

    def _init_schema(self, ddl: str) -> None:
        """
        Execute DDL statements to set up schema.

        Args:
            ddl: DDL script string.
        """
        cursor = self.conn.cursor()
        cursor.executescript(ddl)
        self.conn.commit()

    def execute(
        self, sql: str
    ) -> tuple[bool, list[tuple[Any, ...]] | None, str | None]:
        """
        Execute a SQL query against the in-memory database.

        Args:
            sql: SQL statement to execute.

        Returns:
            Tuple of (success_flag, result_rows_or_none, error_message_or_none).
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            self.conn.commit()
            return True, rows, None
        except (sqlite3.Error, Exception) as e:  # noqa: BLE001
            return False, None, str(e)

    def close(self) -> None:
        """Close the SQLite database connection."""
        self.conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
