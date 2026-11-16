"""AppTest coverage for the guided builder's interactive wiring.

These tests run the real Streamlit app with a fake `generate_prompt`, so they
exercise widget state, preset callbacks, validation, result rendering, and the
handoff to Image Generation without touching the network.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from canvas_copilot.modes import prompt_catalog as catalog
from canvas_copilot.modes import prompt_generation as pg

APP_PATH = Path(__file__).resolve().parents[1] / "src" / "canvas_copilot" / "app.py"


def _open_builder(monkeypatch, output: str = "FAKE PROMPT OUTPUT") -> tuple[AppTest, list[dict]]:
    """Open the app, switch to Prompt Generation, and capture generation calls."""
    calls: list[dict] = []

    def fake_generate(client, capability, format_id=None, **kwargs):
        calls.append({"capability": capability, "format_id": format_id, **kwargs})
        return output

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("EVOLINK_API_KEY", raising=False)
    monkeypatch.setattr(pg, "generate_prompt", fake_generate)

    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=20)
    at.segmented_control(key="app_mode").set_value("Prompt Generation").run(timeout=20)
    return at, calls


def _button(at: AppTest, label: str):
    return next(button for button in at.button if button.label == label)


def test_builder_is_the_default_capability(monkeypatch) -> None:
    at, _ = _open_builder(monkeypatch)
    assert not at.exception
    assert at.segmented_control(key="pg_capability").value == "Prompt Builder"
    assert at.text_area(key="pb_subject").value == ""
    assert at.text_area(key="pb_base_prompt").value == ""
    assert at.selectbox(key="pb_preset").value == "fresh"
    assert at.segmented_control(key="pb_section").value == "character"
    assert "Paste a starting prompt" in at.code[0].value


def test_preset_fills_subject_facets_and_brief(monkeypatch) -> None:
    at, _ = _open_builder(monkeypatch)
    at.selectbox(key="pb_preset").set_value("editorial_portrait").run(timeout=20)
    assert not at.exception
    assert "early 30s" in at.text_area(key="pb_subject").value
    assert at.selectbox(key="pb_character_outfit").value == "Tailored editorial"
    assert "A sharply tailored suit" in at.code[0].value
    assert "large softbox" in at.code[0].value


def test_subject_type_hides_character_facets(monkeypatch) -> None:
    at, _ = _open_builder(monkeypatch)
    at.text_area(key="pb_subject").set_value("A stone chapel on a ridge.").run(timeout=20)
    at.segmented_control(key="pb_subject_type").set_value("Product").run(timeout=20)
    keys = {box.key for box in at.selectbox}
    assert "pb_character_outfit" not in keys
    assert "pb_setting" in keys
    assert "SUBJECT TYPE\nProduct" in at.code[0].value


def test_empty_subject_warns_and_never_calls_the_model(monkeypatch) -> None:
    at, calls = _open_builder(monkeypatch)
    _button(at, "Generate Prompt").click().run(timeout=20)
    assert at.warning
    assert calls == []
    assert "pg_result" not in at.session_state


def test_generate_sends_the_brief_and_renders_the_result(monkeypatch) -> None:
    at, calls = _open_builder(monkeypatch)
    at.text_area(key="pb_subject").set_value(
        "A woman in her 30s on a rooftop at golden hour."
    ).run(timeout=20)
    at.selectbox(key="pb_character_outfit").set_value("Streetwear").run(timeout=20)
    _button(at, "Generate Prompt").click().run(timeout=20)

    assert not at.exception
    assert calls
    assert calls[0]["capability"] == pg.PIPELINE_GUIDED
    assert calls[0]["format_id"] == "format_1"
    assert "A woman in her 30s on a rooftop at golden hour." in calls[0]["guided_brief"]
    assert "Layered streetwear" in calls[0]["guided_brief"]
    assert at.session_state["pg_result"] == "FAKE PROMPT OUTPUT"
    assert [tab.label for tab in at.tabs] == ["Rendered", "Raw (copy-ready)"]
    assert any(expander.label == "Brief used" for expander in at.expander)


def test_surprise_me_fills_facets_but_preserves_free_text(monkeypatch) -> None:
    at, _ = _open_builder(monkeypatch)
    at.text_area(key="pb_subject").set_value("A woman on a rooftop.").run(timeout=20)
    at.text_area(key="pb_character_details").set_value("Glasses and freckles.").run(timeout=20)
    _button(at, "Surprise me").click().run(timeout=20)

    assert not at.exception
    assert at.text_area(key="pb_character_details").value == "Glasses and freckles."
    assert at.selectbox(key="pb_character_outfit").value != catalog.DEFAULT_LABEL
    at.segmented_control(key="pb_section").set_value("scene").run(timeout=20)
    assert at.selectbox(key="pb_setting").value != catalog.DEFAULT_LABEL


def test_selecting_a_facet_keeps_its_section_open(monkeypatch) -> None:
    at, _ = _open_builder(monkeypatch)
    at.segmented_control(key="pb_section").set_value("camera").run(timeout=20)
    at.selectbox(key="pb_lens").set_value("85mm portrait").run(timeout=20)

    assert not at.exception
    assert at.segmented_control(key="pb_section").value == "camera"
    assert at.selectbox(key="pb_lens").value == "85mm portrait"


def test_starting_prompt_is_preserved_and_allows_generation(monkeypatch) -> None:
    at, calls = _open_builder(monkeypatch)
    at.text_area(key="pb_base_prompt").set_value(
        "A quiet harbor at dawn, fishing boats resting on glassy water."
    ).run(timeout=20)
    at.segmented_control(key="pb_section").set_value("camera").run(timeout=20)
    at.selectbox(key="pb_lens").set_value("85mm portrait").run(timeout=20)
    _button(at, "Generate Prompt").click().run(timeout=20)

    assert not at.exception
    assert calls
    brief = calls[0]["guided_brief"]
    assert brief.startswith("STARTING PROMPT\nA quiet harbor at dawn")
    assert "85mm portrait lens" in brief
    assert "SUBJECT\n" not in brief
    assert at.session_state["pg_result"] == "FAKE PROMPT OUTPUT"


def test_switching_format_after_generation_keeps_the_original_rendering(monkeypatch) -> None:
    output = "Direction: A quiet studio portrait."
    at, _ = _open_builder(monkeypatch, output=output)
    at.text_area(key="pb_subject").set_value("A studio portrait.").run(timeout=20)
    _button(at, "Generate Prompt").click().run(timeout=20)
    at.selectbox(key="pg_format").set_value("Format 3 · Structured JSON").run(timeout=20)

    assert not at.exception
    assert not at.error
    assert at.session_state["pg_result_format"] == "format_1"
    assert any(output in block.value for block in at.markdown)


def test_format_3_invalid_json_falls_back_to_raw(monkeypatch) -> None:
    at, calls = _open_builder(monkeypatch, output="this is not json")
    at.selectbox(key="pg_format").set_value("Format 3 · Structured JSON").run(timeout=20)
    at.text_area(key="pb_subject").set_value("A ceramic cup on a plinth.").run(timeout=20)
    _button(at, "Generate Prompt").click().run(timeout=20)

    assert not at.exception
    assert calls[0]["format_id"] == "format_3"
    assert at.error
    assert any("this is not json" in code.value for code in at.code)


def test_use_in_image_generation_hands_off_prompt_and_mode(monkeypatch) -> None:
    at, _ = _open_builder(monkeypatch)
    at.text_area(key="pb_subject").set_value("A lone chapel on a ridge.").run(timeout=20)
    _button(at, "Generate Prompt").click().run(timeout=20)
    _button(at, "Use in Image Generation").click().run(timeout=20)

    assert not at.exception
    assert at.session_state["app_mode"] == "Image Generation"
    assert at.session_state["image_gen_prompt"] == "FAKE PROMPT OUTPUT"
    assert at.text_area(key="image_gen_prompt").value == "FAKE PROMPT OUTPUT"
