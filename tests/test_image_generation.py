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


class TestPerProviderQualityMenus:
    def test_openai_menu_still_offers_auto(self) -> None:
        assert "auto" in ig.QUALITY_OPTIONS

    def test_evolink_menu_holds_the_five_supported_tiers(self) -> None:
        assert list(ig.QUALITY_OPTIONS_EVOLINK) == ["low", "medium", "high", "xhigh", "max"]
        assert "auto" not in ig.QUALITY_OPTIONS_EVOLINK


class TestGenerateImagesEvolink:
    def _run(
        self,
        monkeypatch,
        input_records=None,
        output_format="png",
    ) -> tuple[list[bytes], dict]:
        """Wire fake evolink_api functions and run one Evolink generation."""
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
            "a quiet harbor",
            input_records or [],
            model="gpt-image-2.5-sunburst",
            size="1024x1024",
            quality="medium",
            n=2,
            output_format=output_format,
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

    def test_payload_carries_the_request_parameters(self, monkeypatch) -> None:
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
                "ek-key", "p", [],
                model="gpt-image-2.5-sunburst", size="auto", quality="low", n=1, output_format="png",
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
