"""Live OpenAI API smoke test for the Prompt Generation pipeline.

Runs a real (cheap) call to `gpt-6-luna` — skipped entirely when no API key is
present so `pytest` stays green/offline by default. Deselect with
`pytest -m "not integration"`.
"""

from __future__ import annotations

import os

import pytest

from conftest import PROJECT_ROOT

from dotenv import load_dotenv  # noqa: E402

# Load .env before the skip check so a key stored in the file is honored.
load_dotenv(PROJECT_ROOT / ".env")

pytestmark = pytest.mark.integration

if not os.getenv("OPENAI_API_KEY"):
    pytest.skip("OPENAI_API_KEY not set — skipping live integration test", allow_module_level=True)

from openai import OpenAI  # noqa: E402

import prompt_generation as pg  # noqa: E402


def test_live_upgrade_prompt_with_gpt_6_luna() -> None:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    result = pg.generate_prompt(
        client,
        pg.PIPELINE_TEXT,
        "format_2",
        raw_prompt="a goldfish swimming inside a coffee cup on a kitchen table, morning light",
    )
    assert isinstance(result, str)
    assert len(result) > 200, "expected a rich formatted prompt, got something too short"
    # Format 2 exemplar opens with a "Direction:" section.
    assert "Direction:" in result
