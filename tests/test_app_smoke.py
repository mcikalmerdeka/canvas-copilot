"""Smoke tests: modules import cleanly and both modes render in bare mode.

Streamlit widgets outside a `streamlit run` session return their defaults and
log warnings — that is exactly what makes this a cheap structural test: any
import-time crash, missing constant, or wiring mistake fails loudly here.
"""

from __future__ import annotations

import importlib

from test_prompt_generation import RecordingClient


def test_app_module_imports_and_exposes_modes() -> None:
    app = importlib.import_module("app")
    assert app.MODES == ("Image Generation", "Prompt Generation")


def test_mode_modules_expose_render() -> None:
    image_generation = importlib.import_module("image_generation")
    prompt_generation = importlib.import_module("prompt_generation")
    assert callable(image_generation.render)
    assert callable(prompt_generation.render)


def test_image_generation_renders_in_bare_mode_without_api_calls() -> None:
    image_generation = importlib.import_module("image_generation")
    client = RecordingClient()
    image_generation.render(client)
    assert client.responses.calls == []  # nothing generated until the button is pressed


def test_prompt_generation_renders_in_bare_mode_without_api_calls() -> None:
    prompt_generation = importlib.import_module("prompt_generation")
    client = RecordingClient()
    prompt_generation.render(client)
    assert client.responses.calls == []  # nothing generated until the button is pressed
