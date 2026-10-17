"""Tests for environment verification and dependency availability."""

import importlib.util

from scripts.verify_env import check_env


def test_core_dependencies() -> None:
    """Verify core required dependencies are installed in the environment."""
    core_packages = [
        "sqlglot",
        "pydantic",
        "pytest",
        "yaml",
        "tqdm",
        "rich",
    ]
    for package in core_packages:
        module = importlib.util.find_spec(package)
        assert module is not None, (
            f"Required core package '{package}' is not installed."
        )


def test_check_env() -> None:
    """Verify check_env runs without throwing exceptions and returns True."""
    result = check_env()
    assert result is True
