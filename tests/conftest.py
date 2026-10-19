"""Pytest fixtures shared across the Canvas Copilot test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the project importable regardless of pytest invocation directory:
# the src/ tree (package source) and the tests/ directory (so sibling test
# modules can import each other) both land on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for path in (SRC_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@pytest.fixture()
def project_root() -> Path:
    """Absolute path to the repository root."""
    return PROJECT_ROOT


@pytest.fixture()
def package_root() -> Path:
    """Absolute path to the canvas_copilot package root (src/canvas_copilot)."""
    return PROJECT_ROOT / "src" / "canvas_copilot"
