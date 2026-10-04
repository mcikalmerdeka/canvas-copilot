"""Pytest fixtures shared across the Canvas Copilot test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the project root importable regardless of pytest invocation directory,
# since the app is a flat script layout rather than an installed package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture()
def project_root() -> Path:
    """Absolute path to the repository root."""
    return PROJECT_ROOT
