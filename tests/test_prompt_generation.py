"""Tests for Prompt Generation logic (message building, API payload, output handling).

These tests stub the OpenAI client so no network is involved; they verify the
exact payload the app sends to `gpt-6-luna` and how responses are cleaned up.
"""

from __future__ import annotations

import base64
import json
import logging
from types import SimpleNamespace

import canvas_copilot.modes.prompt_generation as pg
from canvas_copilot.modes.prompt_formats import get_format


class RecordingResponses:
    """Fake `client.responses` that records kwargs and returns a canned output."""

    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.output_text)


class RecordingClient:
    """Fake OpenAI client exposing only `.responses.create`."""

    def __init__(self, output_text: str = "SAMPLE PROMPT OUTPUT") -> None:
        self.responses = RecordingResponses(output_text)


class TestModelConfiguration:
    def test_model_is_gpt_6_luna(self) -> None:
        client = RecordingClient()
        pg.generate_prompt(client, capability="text", format_id="format_2", raw_prompt="a cat")
        assert client.responses.calls[0]["model"] == "gpt-6-luna"
        assert pg.MODEL == "gpt-6-luna"

    def test_reasoning_effort_is_medium(self) -> None:
        client = RecordingClient()
        pg.generate_prompt(client, capability="text", format_id="format_1", raw_prompt="a cat")
        assert client.responses.calls[0]["reasoning"] == {"effort": "medium"}

    def test_max_output_tokens_is_bounded(self) -> None:
        client = RecordingClient()
        pg.generate_prompt(client, capability="text", format_id="format_1", raw_prompt="a cat")
        assert client.responses.calls[0]["max_output_tokens"] == pg.MAX_OUTPUT_TOKENS
        assert 0 < pg.MAX_OUTPUT_TOKENS <= 128_000


class TestEditSystemPrompt:
    def test_contains_edit_exemplar_verbatim(self) -> None:
        from canvas_copilot.modes.prompt_formats import EDIT_EXEMPLAR

        system_prompt = pg.build_edit_system_prompt()
        assert EDIT_EXEMPLAR in system_prompt

    def test_demands_plain_text_output_not_json(self) -> None:
        system_prompt = pg.build_edit_system_prompt()
        assert "code fences" in system_prompt.lower() or "plain text" in system_prompt.lower()

    def test_frames_image_as_source_to_preserve(self) -> None:
        system_prompt = pg.build_edit_system_prompt()
        assert "preserve" in system_prompt.lower()
        assert "source" in system_prompt.lower()


class TestMessageBuilding:
    def test_messages_start_with_system_then_user(self) -> None:
        client = RecordingClient()
        pg.generate_prompt(client, "text", "format_2", raw_prompt="idea")
        messages = client.responses.calls[0]["input"]
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"

    def test_system_message_contains_exemplar_verbatim(self) -> None:
        for format_id in ("format_1", "format_2", "format_3"):
            client = RecordingClient()
            pg.generate_prompt(client, "text", format_id, raw_prompt="idea")
            system_content = client.responses.calls[0]["input"][0]["content"]
            assert get_format(format_id)["exemplar_text"] in system_content

    def test_text_user_message_includes_raw_prompt(self) -> None:
        client = RecordingClient()
        pg.generate_prompt(client, "text", "format_1", raw_prompt="a harbor at dawn")
        user_content = client.responses.calls[0]["input"][1]["content"]
        assert "a harbor at dawn" in user_content

    def test_image_user_message_carries_input_image_part(self) -> None:
        client = RecordingClient()
        image_bytes = b"\x89PNG fake bytes"
        pg.generate_prompt(client, "image", "format_2", image_bytes=image_bytes, image_mime="image/png")
        parts = client.responses.calls[0]["input"][1]["content"]
        assert parts[0]["type"] == "input_text"
        expected_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
        assert parts[1]["type"] == "input_image"
        assert parts[1]["image_url"] == expected_url

    def test_edit_user_message_carries_request_and_image_part(self) -> None:
        image_bytes = b"\x89PNG fake bytes"
        parts = pg.build_user_content(
            pg.PIPELINE_EDIT,
            image_bytes=image_bytes,
            image_mime="image/png",
            edit_request="remove the QR code and all text",
        )
        assert parts[0]["type"] == "input_text"
        assert "remove the QR code and all text" in parts[0]["text"]
        expected_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
        assert parts[1]["type"] == "input_image"
        assert parts[1]["image_url"] == expected_url

    def test_edit_user_message_does_not_include_image_to_prompt_instructions(self) -> None:
        parts = pg.build_user_content(
            pg.PIPELINE_EDIT,
            image_bytes=b"x",
            image_mime="image/jpeg",
            edit_request="extend downward",
        )
        text = parts[0]["text"]
        assert "reverse-engineer" not in text.lower()
        assert "extend downward" in text

    def test_image_user_message_includes_extra_direction_when_given(self) -> None:
        client = RecordingClient()
        pg.generate_prompt(client, "image", "format_2", image_bytes=b"x", image_mime="image/jpeg", extra_direction="focus on the lighting")
        text_part = client.responses.calls[0]["input"][1]["content"][0]["text"]
        assert "focus on the lighting" in text_part


class TestEditPipeline:
    def test_edit_capability_sends_edit_system_prompt_and_user_parts(self) -> None:
        client = RecordingClient()
        image_bytes = b"\x89PNG fake bytes"
        pg.generate_prompt(
            client,
            pg.PIPELINE_EDIT,
            image_bytes=image_bytes,
            image_mime="image/png",
            edit_request="remove the text overlay",
        )
        messages = client.responses.calls[0]["input"]
        assert pg.build_edit_system_prompt() == messages[0]["content"]
        parts = messages[1]["content"]
        assert "remove the text overlay" in parts[0]["text"]
        assert parts[1]["type"] == "input_image"

    def test_edit_capability_needs_no_format_id(self) -> None:
        client = RecordingClient()
        pg.generate_prompt(client, pg.PIPELINE_EDIT, image_bytes=b"x", image_mime="image/png", edit_request="extend downward")
        # No KeyError and the three-format registry was never consulted.
        assert client.responses.calls[0]["input"][0]["role"] == "system"

    def test_upgrade_capability_still_uses_format_exemplar(self) -> None:
        from canvas_copilot.modes.prompt_formats import get_format

        client = RecordingClient()
        pg.generate_prompt(client, pg.PIPELINE_TEXT, "format_2", raw_prompt="idea")
        assert get_format("format_2")["exemplar_text"] in client.responses.calls[0]["input"][0]["content"]


class TestOutputHandling:
    def test_returns_cleaned_model_text(self) -> None:
        client = RecordingClient(output_text="Direction: something\nMood: calm")
        result = pg.generate_prompt(client, "text", "format_2", raw_prompt="idea")
        assert result == "Direction: something\nMood: calm"

    def test_code_fences_are_stripped_from_output(self) -> None:
        client = RecordingClient(output_text="```json\n{\"a\": 1}\n```")
        result = pg.generate_prompt(client, "text", "format_3", raw_prompt="idea")
        assert result == '{"a": 1}'

    def test_format_3_output_is_valid_json_ready_for_download(self) -> None:
        payload = json.dumps({"task": "Image Generation", "aspect_ratio": "9:16"})
        client = RecordingClient(output_text=payload)
        result = pg.generate_prompt(client, "text", "format_3", raw_prompt="idea")
        assert json.loads(result)["aspect_ratio"] == "9:16"


class TestJsonParsing:
    def test_try_parse_json_accepts_object(self) -> None:
        assert pg.try_parse_json('{"a": 1}') == {"a": 1}

    def test_try_parse_json_accepts_fenced_object(self) -> None:
        assert pg.try_parse_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_try_parse_json_rejects_prose(self) -> None:
        assert pg.try_parse_json("Direction: a train platform\nMood: calm") is None

    def test_try_parse_json_rejects_non_object_json(self) -> None:
        assert pg.try_parse_json("[1, 2, 3]") is None


class TestFilenames:
    def test_result_filename_uses_format_id_and_extension(self) -> None:
        assert pg.result_filename(get_format("format_1")) == "prompt_format_1.md"
        assert pg.result_filename(get_format("format_2")) == "prompt_format_2.md"
        assert pg.result_filename(get_format("format_3")) == "prompt_format_3.json"

    def test_edit_result_meta_renders_markdown_and_names_download(self) -> None:
        assert pg.EDIT_RESULT_META["is_json"] is False
        assert pg.result_filename(pg.EDIT_RESULT_META) == "prompt_edit.md"


# --- Provider-aware image references ----------------------------------------------

import canvas_copilot.core.evolink_api as ev  # noqa: E402
import canvas_copilot.settings.providers as pr  # noqa: E402

EVOLINK_CTX = pr.ProviderContext(
    provider_id=pr.PROVIDER_EVOLINK, api_key="ek-test", client=RecordingClient()
)
OPENAI_CTX = pr.ProviderContext(
    provider_id=pr.PROVIDER_OPENAI, api_key="sk-test", client=RecordingClient()
)


class TestImageReferenceResolution:
    """Evolink needs a downloadable URL for `input_image`; OpenAI takes data URLs."""

    def test_openai_resolves_to_a_data_url(self) -> None:
        resolved = pg.resolve_image_ref(OPENAI_CTX, b"\x89PNG bytes", "image/png", "source.png")
        assert resolved == "data:image/png;base64," + base64.b64encode(b"\x89PNG bytes").decode("ascii")

    def test_evolink_uploads_the_bytes_and_uses_the_file_url(self, monkeypatch) -> None:
        uploads: list[tuple] = []

        def fake_upload(api_key, image_bytes, image_mime, file_name=None):
            uploads.append((api_key, image_bytes, image_mime, file_name))
            return f"https://files.evolink.ai/u/{file_name}"

        monkeypatch.setattr(ev, "upload_image", fake_upload)
        resolved = pg.resolve_image_ref(EVOLINK_CTX, b"jpeg-bytes", "image/jpeg", "photo.jpg")
        assert resolved == "https://files.evolink.ai/u/photo.jpg"
        assert uploads[0] == ("ek-test", b"jpeg-bytes", "image/jpeg", "photo.jpg")


class TestUserContentWithImageUrl:
    def test_image_capability_uses_the_given_url_verbatim(self) -> None:
        parts = pg.build_user_content(pg.PIPELINE_IMAGE, image_url="https://files.evolink.ai/u/photo.jpg")
        assert parts[1] == {"type": "input_image", "image_url": "https://files.evolink.ai/u/photo.jpg"}

    def test_edit_capability_uses_the_given_url_verbatim(self) -> None:
        parts = pg.build_user_content(
            pg.PIPELINE_EDIT,
            edit_request="remove the text",
            image_url="https://files.evolink.ai/u/source.png",
        )
        assert parts[1] == {"type": "input_image", "image_url": "https://files.evolink.ai/u/source.png"}

    def test_generate_prompt_forwards_the_resolved_url(self) -> None:
        client = RecordingClient()
        pg.generate_prompt(client, "image", "format_2", image_url="https://files.evolink.ai/u/photo.jpg")
        parts = client.responses.calls[0]["input"][1]["content"]
        assert parts[1]["type"] == "input_image"
        assert parts[1]["image_url"] == "https://files.evolink.ai/u/photo.jpg"


class TestGenerationLogging:
    def test_start_and_finish_are_logged(self, caplog) -> None:
        """Started/finished lines reach the terminal log; output is unchanged."""
        client = RecordingClient(output_text="SAMPLE PROMPT OUTPUT")
        with caplog.at_level(logging.INFO, logger="canvas_copilot.modes.prompt_generation"):
            result = pg.generate_prompt(client, capability="text", format_id="format_2", raw_prompt="idea")
        messages = [
            rec.getMessage()
            for rec in caplog.records
            if rec.name == "canvas_copilot.modes.prompt_generation"
        ]
        assert any(msg.startswith("Prompt generation started (capability=text, format=format_2)") for msg in messages)
        assert any(msg.startswith("Prompt finished") and "20 characters" in msg for msg in messages)
        assert result == "SAMPLE PROMPT OUTPUT"
