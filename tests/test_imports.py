"""Import smoke tests — every package module imports cleanly and exposes its API.

These pin the src-layout package (`canvas_copilot`) itself: after any file
move or rename, running this file answers "does everything still import?"
"""

from __future__ import annotations

import canvas_copilot as package  # noqa: I001 — order matters below (package first)
import canvas_copilot.core.evolink_api as evolink_api
import canvas_copilot.modes.image_generation as image_generation
import canvas_copilot.modes.prompt_formats as prompt_formats
import canvas_copilot.modes.prompt_generation as prompt_generation
import canvas_copilot.settings.providers as providers


def test_package_is_importable_and_named() -> None:
    assert package.__name__ == "canvas_copilot"


def test_settings_providers_exposes_the_registry_api() -> None:
    assert providers.PROVIDERS == ("OpenAI", "Evolink")
    for name in ("get_api_key", "missing_key_message", "build_context"):
        assert callable(getattr(providers, name)), name
    assert providers.ProviderContext


def test_core_evolink_api_exposes_the_client_api() -> None:
    for name in (
        "to_data_url",
        "build_upload_payload",
        "build_generation_payload",
        "poll_task",
        "extract_result_urls",
        "describe_failure",
        "upload_image",
        "create_generation_task",
        "fetch_task",
        "download_image",
    ):
        assert callable(getattr(evolink_api, name)), name


def test_modes_expose_their_renderers_and_registry() -> None:
    assert callable(image_generation.render)
    assert callable(prompt_generation.render)
    assert callable(prompt_formats.get_format)
    assert callable(prompt_formats.list_labels)


def test_app_exposes_the_mode_dispatch() -> None:
    import canvas_copilot.app as app

    assert app.MODES == ("Image Generation", "Prompt Generation")
