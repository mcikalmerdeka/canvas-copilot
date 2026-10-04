"""Tests for Prompt Generation logic (message building, API payload, output handling).

These tests stub the OpenAI client so no network is involved; they verify the
exact payload the app sends to `gpt-6-luna` and how responses are cleaned up.
"""

from __future__ import annotations

import base64
import json
from types import SimpleNamespace

import prompt_generation as pg
from prompt_formats import get_format


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

    def test_image_user_message_includes_extra_direction_when_given(self) -> None:
        client = RecordingClient()
        pg.generate_prompt(client, "image", "format_2", image_bytes=b"x", image_mime="image/jpeg", extra_direction="focus on the lighting")
        text_part = client.responses.calls[0]["input"][1]["content"][0]["text"]
        assert "focus on the lighting" in text_part


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
