"""Tests for Image Generation logic across both providers.

OpenAI keeps the sync multipart flow; Evolink runs on its async task API
(fake module functions here, so no network happens) — these tests pin the
composition: uploads → payload → poll → downloads, and the per-provider
option menus.
"""

from __future__ import annotations

import base64
import io
from types import SimpleNamespace

import canvas_copilot.core.evolink_api as ev
import canvas_copilot.core.model_registry as mr
import canvas_copilot.modes.image_generation as ig

COMPLETED_TASK = {
    "id": "task-unified-1",
    "status": "completed",
    "progress": 100,
    "results": ["https://files.example/one.png", "https://files.example/two.png"],
}


def collect_calls(result) -> dict:
    """Monkeypatch helper that records calls to an evolink_api function."""
    calls: list[dict] = []

    def recording(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return result(*args, **kwargs) if callable(result) else result

    return calls, recording


class TestCollectInputRecords:
    def test_no_inputs_yields_empty_list(self) -> None:
        assert ig.collect_input_records(None, None) == []

    def test_handoff_image_becomes_first_record(self) -> None:
        handoff = {"bytes": b"\x89PNG handoff", "mime": "image/png", "name": "source.png"}
        records = ig.collect_input_records(None, handoff)
        assert records == [{"bytes": b"\x89PNG handoff", "mime": "image/png", "name": "source.png"}]

    def test_uploaded_references_follow_the_handoff_image(self) -> None:
        upload = SimpleNamespace(getvalue=lambda: b"upload-bytes", name="upload.jpg", type="image/jpeg")
        handoff = {"bytes": b"handoff-bytes", "mime": "image/png", "name": "handoff.png"}
        records = ig.collect_input_records([upload], handoff)
        assert records[0] == {"bytes": b"handoff-bytes", "mime": "image/png", "name": "handoff.png"}
        assert records[1] == {"bytes": b"upload-bytes", "mime": "image/jpeg", "name": "upload.jpg"}

    def test_multiple_uploads_keep_their_order(self) -> None:
        first = SimpleNamespace(getvalue=lambda: b"one", name="one.png", type="image/png")
        second = SimpleNamespace(getvalue=lambda: b"two", name="two.png", type="image/png")
        records = ig.collect_input_records([first, second], None)
        assert [r["name"] for r in records] == ["one.png", "two.png"]

    def test_uploaded_file_without_mime_defaults_to_png(self) -> None:
        mystery = SimpleNamespace(getvalue=lambda: b"x", name="mystery", type=None)
        records = ig.collect_input_records([mystery], None)
        assert records[0]["mime"] == "image/png"


class TestToNamedBuffer:
    def test_wraps_record_bytes_as_a_named_buffer(self) -> None:
        buffer = ig.to_named_buffer({"bytes": b"abc", "mime": "image/png", "name": "a.png"})
        assert isinstance(buffer, io.BytesIO)
        assert buffer.getvalue() == b"abc"
        assert buffer.name == "a.png"


class TestValidateReferenceCount:
    def test_within_cap_is_accepted(self) -> None:
        spec = mr.EVOLINK_MODELS["gemini-nano-banana-2.1"]
        assert ig.validate_reference_count(spec, 14) is None
        assert ig.validate_reference_count(spec, 0) is None

    def test_over_cap_returns_a_message_naming_the_cap(self) -> None:
        spec = mr.EVOLINK_MODELS["grok-imagine-image-2.0"]
        message = ig.validate_reference_count(spec, 5)
        assert message is not None
        assert "3" in message
        assert spec.label in message


class TestGenerateImagesEvolink:
    def _run(
        self,
        monkeypatch,
        model_id="gpt-image-2.5-sunburst",
        input_records=None,
        size="1024x1024",
        quality="medium",
        n=2,
        output_format="png",
        resolution=None,
        model_params=None,
        negative_prompt=None,
    ) -> tuple[list[bytes], dict]:
        """Wire fake evolink_api functions and run one Evolink generation."""
        spec = mr.EVOLINK_MODELS[model_id]
        upload_calls, upload_recording = collect_calls(
            lambda api_key, image_bytes, image_mime, file_name=None: f"https://files.evolink.ai/{file_name}"
        )
        create_calls, create_recording = collect_calls(COMPLETED_TASK)
        download_calls, download_recording = collect_calls(
            lambda url: b"image-bytes-for-" + url.rsplit("/", 1)[-1].encode()
        )
        monkeypatch.setattr(ev, "upload_image", upload_recording)
        monkeypatch.setattr(ev, "create_generation_task", create_recording)

        created = {"id": "task-1", "status": "pending"}
        monkeypatch.setattr(ev, "fetch_task", lambda api_key, task_id: created)
        monkeypatch.setattr(ev, "download_image", download_recording)

        progress_values: list[int] = []
        monkeypatch.setattr(
            ev, "poll_task",
            lambda fetch_task, task_id, **kwargs: COMPLETED_TASK,
        )
        images = ig.generate_images_evolink(
            "ek-key",
            spec,
            "a quiet harbor",
            input_records or [],
            size=size,
            quality=quality,
            n=n,
            output_format=output_format,
            resolution=resolution,
            model_params=model_params,
            negative_prompt=negative_prompt,
            on_progress=lambda task: progress_values.append(task["progress"]),
        )
        details = {
            "upload_calls": upload_calls,
            "create_calls": create_calls,
            "download_calls": download_calls,
            "progress_values": progress_values,
        }
        return images, details

    def test_uploads_references_and_passes_their_urls(self, monkeypatch) -> None:
        records = [
            {"bytes": b"handoff", "mime": "image/png", "name": "handoff.png"},
            {"bytes": b"upload", "mime": "image/jpeg", "name": "ref.jpg"},
        ]
        images, details = self._run(monkeypatch, input_records=records)
        assert [c["kwargs"].get("file_name") or c["args"][3] for c in details["upload_calls"]] == [
            "handoff.png",
            "ref.jpg",
        ]
        payload = details["create_calls"][0]["args"][1]
        assert payload["image_urls"] == [
            "https://files.evolink.ai/handoff.png",
            "https://files.evolink.ai/ref.jpg",
        ]

    def test_text_only_flow_skips_the_file_uploads(self, monkeypatch) -> None:
        images, details = self._run(monkeypatch, input_records=[])
        assert details["upload_calls"] == []
        assert "image_urls" not in details["create_calls"][0]["args"][1]

    def test_downloads_every_result_url_in_order(self, monkeypatch) -> None:
        images, details = self._run(monkeypatch)
        assert images == [b"image-bytes-for-one.png", b"image-bytes-for-two.png"]
        urls = [c["args"][0] for c in details["download_calls"]]
        assert urls == ["https://files.example/one.png", "https://files.example/two.png"]

    def test_gpt_payload_carries_the_request_parameters(self, monkeypatch) -> None:
        images, details = self._run(monkeypatch, output_format="jpeg")
        payload = details["create_calls"][0]["args"][1]
        assert payload["model"] == "gpt-image-2.5-sunburst"
        assert payload["prompt"] == "a quiet harbor"
        assert payload["size"] == "1024x1024"
        assert payload["quality"] == "medium"
        assert payload["n"] == 2
        assert payload["output_format"] == "jpeg"
        assert "moderation" not in payload
        assert "output_compression" not in payload

    def test_grok_sends_its_separate_resolution_param(self, monkeypatch) -> None:
        images, details = self._run(
            monkeypatch, model_id="grok-imagine-image-2.0", size="16:9", quality="low", resolution="2K", n=4,
        )
        payload = details["create_calls"][0]["args"][1]
        assert payload["model"] == "grok-imagine-image-2.0"
        assert payload["resolution"] == "2K"
        assert payload["size"] == "16:9"
        assert payload["quality"] == "low"
        assert payload["n"] == 4
        assert "output_format" not in payload  # not supported by this model

    def test_nanobanana_never_sends_n_or_output_format(self, monkeypatch) -> None:
        images, details = self._run(
            monkeypatch, model_id="gemini-nano-banana-2.1", size="16:9", quality="2K", n=None, output_format=None,
        )
        payload = details["create_calls"][0]["args"][1]
        assert payload["model"] == "gemini-nano-banana-2.1"
        assert "n" not in payload
        assert "output_format" not in payload
        assert payload["size"] == "16:9"
        assert payload["quality"] == "2K"

    def test_qwen_sends_negative_prompt_and_n(self, monkeypatch) -> None:
        images, details = self._run(
            monkeypatch, model_id="qwen-image-3.0-pro", size="1:1", quality="1K", n=3,
            negative_prompt="blurry, extra fingers",
        )
        payload = details["create_calls"][0]["args"][1]
        assert payload["negative_prompt"] == "blurry, extra fingers"
        assert payload["n"] == 3
        assert "resolution" not in payload

    def test_midjourney_folds_references_and_aspect_into_the_prompt(self, monkeypatch) -> None:
        records = [{"bytes": b"ref", "mime": "image/png", "name": "ref.png"}]
        images, details = self._run(
            monkeypatch, model_id="mj-v8.2", input_records=records,
            size="16:9", quality="standard", n=None, output_format=None,
            model_params={"speed": "fast"},
        )
        payload = details["create_calls"][0]["args"][1]
        assert payload["prompt"] == "https://files.evolink.ai/ref.png a quiet harbor --ar 16:9"
        assert "image_urls" not in payload
        assert "size" not in payload
        assert "n" not in payload
        assert payload["model_params"] == {"speed": "fast"}
        assert payload["quality"] == "standard"

    def test_midjourney_auto_aspect_adds_no_flag(self, monkeypatch) -> None:
        images, details = self._run(
            monkeypatch, model_id="mj-v8.2", size="auto", quality="standard", n=None, output_format=None,
        )
        payload = details["create_calls"][0]["args"][1]
        assert payload["prompt"] == "a quiet harbor"

    def test_midjourney_prompt_budget_is_enforced(self, monkeypatch) -> None:
        long_prompt = "water " * 500  # > 2048 chars
        try:
            ig.generate_images_evolink(
                "ek-key", mr.EVOLINK_MODELS["mj-v8.2"], long_prompt, [],
                size="16:9", quality="standard", n=None, output_format=None,
            )
        except RuntimeError as error:
            assert "2048" in str(error)
        else:
            raise AssertionError("expected RuntimeError for an oversized MJ prompt")

    def test_failed_task_raises_with_the_server_reason(self, monkeypatch) -> None:
        failed = {
            "id": "task-1",
            "status": "failed",
            "error": {"code": "rate_limit_exceeded", "message": "Too many requests"},
        }
        monkeypatch.setattr(ev, "upload_image", lambda *a, **k: "https://u")
        monkeypatch.setattr(ev, "create_generation_task", lambda api_key, payload: {"id": "t", "status": "pending"})
        monkeypatch.setattr(ev, "fetch_task", lambda api_key, task_id: failed)
        monkeypatch.setattr(ev, "poll_task", lambda fetch_task, task_id, **kwargs: failed)
        try:
            ig.generate_images_evolink(
                "ek-key", mr.EVOLINK_MODELS["gpt-image-2.5-sunburst"], "p", [],
                size="auto", quality="low", n=1, output_format="png",
            )
        except RuntimeError as error:
            assert "rate_limit_exceeded" in str(error)
            assert "Too many requests" in str(error)
        else:
            raise AssertionError("expected RuntimeError for a failed task")

    def test_completed_task_without_urls_raises(self, monkeypatch) -> None:
        empty = {"id": "t", "status": "completed", "results": []}
        monkeypatch.setattr(ev, "create_generation_task", lambda api_key, payload: {"id": "t", "status": "pending"})
        monkeypatch.setattr(ev, "fetch_task", lambda api_key, task_id: empty)
        monkeypatch.setattr(ev, "poll_task", lambda fetch_task, task_id, **kwargs: empty)
