"""Smoke tests: modules import cleanly and both modes render in bare mode.

Streamlit widgets outside a `streamlit run` session return their defaults and
log warnings — that is exactly what makes this a cheap structural test: any
import-time crash, missing constant, or wiring mistake fails loudly here.
"""

from __future__ import annotations

import importlib

import canvas_copilot.settings.providers as providers
from test_prompt_generation import RecordingClient


def make_context(client, provider_id: str = providers.PROVIDER_OPENAI) -> providers.ProviderContext:
    """A bare-mode context: real fields, no env or network involved."""
    return providers.ProviderContext(provider_id=provider_id, api_key="test-key", client=client)


def test_app_module_imports_and_exposes_modes() -> None:
    app = importlib.import_module("canvas_copilot.app")
    assert app.MODES == ("Image Generation", "Prompt Generation")


def test_provider_registry_lists_both_providers() -> None:
    assert providers.PROVIDERS == ("OpenAI", "Evolink")


def test_mode_modules_expose_render() -> None:
    image_generation = importlib.import_module("canvas_copilot.modes.image_generation")
    prompt_generation = importlib.import_module("canvas_copilot.modes.prompt_generation")
    assert callable(image_generation.render)
    assert callable(prompt_generation.render)


def test_image_generation_renders_in_bare_mode_without_api_calls() -> None:
    image_generation = importlib.import_module("canvas_copilot.modes.image_generation")
    ctx = make_context(RecordingClient())
    image_generation.render(ctx)
    assert ctx.client.responses.calls == []  # nothing generated until the button is pressed


def test_image_generation_renders_in_bare_mode_for_evolink_too() -> None:
    image_generation = importlib.import_module("canvas_copilot.modes.image_generation")
    ctx = make_context(RecordingClient(), providers.PROVIDER_EVOLINK)
    image_generation.render(ctx)  # evolink branch selects its own menus without crashing


def test_prompt_generation_renders_in_bare_mode_without_api_calls() -> None:
    prompt_generation = importlib.import_module("canvas_copilot.modes.prompt_generation")
    ctx = make_context(RecordingClient())
    prompt_generation.render(ctx)
    assert ctx.client.responses.calls == []  # nothing generated until the button is pressed


# --- app.py wiring ----------------------------------------------------------------

def _patch_mode_renders(monkeypatch) -> list:
    """Patch both mode renders to record what app.py dispatches to them."""
    seen: list[tuple[str, object]] = []
    image_generation = importlib.import_module("canvas_copilot.modes.image_generation")
    prompt_generation = importlib.import_module("canvas_copilot.modes.prompt_generation")
    monkeypatch.setattr(image_generation, "render", lambda ctx: seen.append(("image", ctx)))
    monkeypatch.setattr(prompt_generation, "render", lambda ctx: seen.append(("prompt", ctx)))
    return seen


def test_stops_loudly_when_the_selected_provider_key_is_missing(monkeypatch) -> None:
    # st.stop() only halts real Streamlit sessions (a no-op in bare mode), so
    # the app-level behavior is pinned by observing the calls app.py makes.
    app = importlib.import_module("canvas_copilot.app")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: None)  # reload re-runs the from-import; .env must not restore the key
    stops: list[bool] = []
    errors: list[str] = []
    monkeypatch.setattr(app.st, "stop", lambda: stops.append(True))
    monkeypatch.setattr(app.st, "error", lambda message: errors.append(message))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("EVOLINK_API_KEY", raising=False)
    importlib.reload(app)
    assert stops == [True]  # the app halts instead of sending keyless requests
    assert any("OPENAI_API_KEY" in message for message in errors), errors


def test_dispatch_passes_a_provider_context_to_the_mode(monkeypatch) -> None:
    app = importlib.import_module("canvas_copilot.app")
    seen = _patch_mode_renders(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-bare-mode-test")
    monkeypatch.setenv("EVOLINK_API_KEY", "ek-bare-mode-test")
    importlib.reload(app)
    assert [target for target, _ in seen] == ["image"]  # default mode renders Image Generation
    (target, ctx) = seen[0]
    assert isinstance(ctx, providers.ProviderContext)
    assert ctx.provider_id == providers.PROVIDER_OPENAI
    assert ctx.api_key == "sk-bare-mode-test"
    assert ctx.client is not None
