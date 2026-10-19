"""Tests for Evolink image-task logic: payloads, polling, result extraction.

All network access is fake here — fetch_task and sleep are injected callables,
so these tests exercise the real request/poll logic offline.
"""

from __future__ import annotations

import base64

import canvas_copilot.core.evolink_api as ev

COMPLETED_TASK = {
    "id": "task-unified-1",
    "status": "completed",
    "results": ["https://files.example/one.png", "https://files.example/two.png"],
}
FAILED_TASK = {
    "id": "task-unified-2",
    "status": "failed",
    "error": {
        "code": "content_policy_violation",
        "message": "Your request was blocked by safety filters.",
        "details": "prompt section 2",
    },
}


class TestDataUrlEncoding:
    def test_encodes_bytes_and_mime(self) -> None:
        encoded = ev.to_data_url(b"\x89PNG", "image/png")
        assert encoded == "data:image/png;base64," + base64.b64encode(b"\x89PNG").decode()


class TestUploadPayload:
    def test_carries_a_data_url_of_the_image(self) -> None:
        payload = ev.build_upload_payload(b"\x89PNG", "image/png")
        expected = "data:image/png;base64," + base64.b64encode(b"\x89PNG").decode()
        assert payload["base64_data"] == expected

    def test_optional_name_is_passed_through(self) -> None:
        assert "file_name" not in ev.build_upload_payload(b"x", "image/png")
        styled = ev.build_upload_payload(b"x", "image/jpeg", file_name="ref.jpg")
        assert styled["file_name"] == "ref.jpg"


class TestGenerationPayload:
    def test_minimal_text_to_image_request(self) -> None:
        payload = ev.build_generation_payload(
            model="gpt-image-2.5-sunburst",
            prompt="a quiet harbor",
            size="auto",
            quality="medium",
            n=1,
            output_format="png",
        )
        assert payload == {
            "model": "gpt-image-2.5-sunburst",
            "prompt": "a quiet harbor",
            "size": "auto",
            "quality": "medium",
            "n": 1,
            "output_format": "png",
        }

    def test_reference_images_land_in_image_urls(self) -> None:
        payload = ev.build_generation_payload(
            model="gpt-image-2.5-sunburst",
            prompt="edit this",
            image_urls=["https://files.evolink.ai/a.png", "https://files.evolink.ai/b.png"],
            size="1024x1024",
            quality="low",
            n=2,
            output_format="png",
        )
        assert payload["image_urls"] == ["https://files.evolink.ai/a.png", "https://files.evolink.ai/b.png"]

    def test_no_moderation_field_is_ever_sent(self) -> None:
        payload = ev.build_generation_payload(
            model="m", prompt="p", size="1024x1024", quality="xhigh", n=1, output_format="jpeg",
            image_urls=["https://x"],
        )
        assert "moderation" not in payload


class TestPollTask:
    @staticmethod
    def fake_fetch(statuses):
        """Build a fetch_task that replays the scripted statuses in order."""
        calls = {"i": 0}

        def fetch(task_id: str) -> dict:
            status = statuses[min(calls["i"], len(statuses) - 1)]
            calls["i"] += 1
            return {"id": task_id, "status": status}

        return fetch

    def test_completed_task_returns_immediately_without_sleeping(self) -> None:
        sleeps: list[float] = []
        task = ev.poll_task(
            self.fake_fetch(["completed"]), "task-1",
            sleep=sleeps.append, interval_s=3.0, timeout_s=600.0,
        )
        assert task["status"] == "completed"
        assert sleeps == []

    def test_failed_task_returns_immediately(self) -> None:
        task = ev.poll_task(
            self.fake_fetch(["failed"]), "task-1",
            sleep=lambda _s: None, interval_s=3.0, timeout_s=600.0,
        )
        assert task["status"] == "failed"

    def test_processes_pending_then_processing_until_completed(self) -> None:
        sleeps: list[float] = []
        task = ev.poll_task(
            self.fake_fetch(["pending", "processing", "completed"]), "t",
            sleep=sleeps.append, interval_s=2.5, timeout_s=600.0,
        )
        assert task["status"] == "completed"
        assert sleeps == [2.5, 2.5]

    def test_progress_callback_sees_every_fetch(self) -> None:
        replies = [
            {"id": "t", "status": "processing", "progress": 40},
            {"id": "t", "status": "completed", "progress": 100},
        ]
        remaining = list(replies)

        def fetch(task_id: str) -> dict:
            return remaining.pop(0)

        seen: list[dict] = []
        task = ev.poll_task(
            fetch, "t", sleep=lambda _s: None, interval_s=1.0, timeout_s=600.0, on_progress=seen.append,
        )
        assert task["status"] == "completed"
        assert seen == replies

    def test_timeout_raises_after_spending_the_budget(self) -> None:
        def endless_fetch(task_id: str) -> dict:
            return {"id": task_id, "status": "processing", "progress": 10}

        try:
            ev.poll_task(
                endless_fetch, "task-unified-42",
                sleep=lambda _s: None, interval_s=3.0, timeout_s=9.0,
            )
        except ev.TaskPollTimeout as error:
            assert "task-unified-42" in str(error)
        else:
            raise AssertionError("expected TaskPollTimeout for an endless task")


class TestExtractResultUrls:
    def test_prefers_the_results_list(self) -> None:
        assert ev.extract_result_urls(COMPLETED_TASK) == [
            "https://files.example/one.png",
            "https://files.example/two.png",
        ]

    def test_falls_back_to_result_data_urls(self) -> None:
        task = {"status": "completed", "result_data": [{"url": "https://x/1.png"}]}
        assert ev.extract_result_urls(task) == ["https://x/1.png"]

    def test_missing_results_yield_empty_list(self) -> None:
        assert ev.extract_result_urls({"status": "completed"}) == []


class TestDescribeFailure:
    def test_joins_code_message_and_details(self) -> None:
        text = ev.describe_failure(FAILED_TASK)
        assert "content_policy_violation" in text
        assert "blocked by safety filters" in text
        assert "prompt section 2" in text

    def test_handles_an_errorless_failed_task(self) -> None:
        assert isinstance(ev.describe_failure({"status": "failed"}), str)


# ---------------------------------------------------------------------------
# Thin HTTP wrappers — requests is faked at the module boundary so the tests
# stay offline while pinning URLs, auth headers, and response parsing.
# ---------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, json_data=None, content=b"", status_code: int = 200) -> None:
        self._json_data = json_data
        self.content = content
        self.status_code = status_code

    def json(self) -> dict:
        return self._json_data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = ev.requests.HTTPError(f"HTTP {self.status_code}")
            error.response = self
            raise error


class TestUploadImage:
    def test_posts_data_url_to_the_file_service_and_returns_the_url(self, monkeypatch) -> None:
        calls: list[dict] = []

        def fake_post(url, json=None, headers=None, timeout=None):
            calls.append({"url": url, "json": json, "headers": headers})
            return FakeResponse(
                json_data={
                    "data": {"file_url": "https://files.evolink.ai/u/source.png", "file_id": "f1"}
                }
            )

        monkeypatch.setattr(ev.requests, "post", fake_post)
        url = ev.upload_image("ek-key", b"\x89PNG bytes", "image/png", "source.png")
        assert url == "https://files.evolink.ai/u/source.png"
        assert calls[0]["url"] == "https://files-api.evolink.ai/api/v1/files/upload/base64"
        assert calls[0]["headers"]["Authorization"] == "Bearer ek-key"
        assert calls[0]["json"]["file_name"] == "source.png"
        assert calls[0]["json"]["base64_data"].startswith("data:image/png;base64,")

    def test_raises_when_the_upload_returns_no_file_url(self, monkeypatch) -> None:
        monkeypatch.setattr(
            ev.requests, "post", lambda *args, **kwargs: FakeResponse(json_data={"data": {}})
        )
        try:
            ev.upload_image("ek-key", b"x", "image/png")
        except RuntimeError as error:
            assert "upload" in str(error).lower()
        else:
            raise AssertionError("expected RuntimeError when file_url is absent")


class TestCreateGenerationTask:
    def test_posts_the_payload_and_returns_the_task(self, monkeypatch) -> None:
        calls: list[dict] = []
        task = {"id": "task-unified-9", "status": "pending"}

        def fake_post(url, json=None, headers=None, timeout=None):
            calls.append({"url": url, "json": json, "headers": headers})
            return FakeResponse(json_data=task)

        monkeypatch.setattr(ev.requests, "post", fake_post)
        payload = ev.build_generation_payload(
            model="gpt-image-2.5-flare", prompt="p", size="auto", quality="low", n=1, output_format="png",
        )
        returned = ev.create_generation_task("ek-key", payload)
        assert returned == task
        assert calls[0]["url"] == "https://api.evolink.ai/v1/images/generations"
        assert calls[0]["headers"]["Authorization"] == "Bearer ek-key"
        assert calls[0]["json"]["model"] == "gpt-image-2.5-flare"

    def test_raises_when_no_task_id_comes_back(self, monkeypatch) -> None:
        monkeypatch.setattr(
            ev.requests, "post", lambda *args, **kwargs: FakeResponse(json_data={"status": "pending"})
        )
        try:
            ev.create_generation_task("ek-key", {"model": "m", "prompt": "p"})
        except RuntimeError as error:
            assert "task" in str(error).lower()
        else:
            raise AssertionError("expected RuntimeError when task id is absent")


class TestFetchTask:
    def test_queries_the_task_endpoint_with_bearer_auth(self, monkeypatch) -> None:
        calls: list[dict] = []

        def fake_get(url, headers=None, timeout=None):
            calls.append({"url": url, "headers": headers})
            return FakeResponse(json_data={"id": "task-1", "status": "processing"})

        monkeypatch.setattr(ev.requests, "get", fake_get)
        task = ev.fetch_task("ek-key", "task-unified-9")
        assert task["status"] == "processing"
        assert calls[0]["url"] == "https://api.evolink.ai/v1/tasks/task-unified-9"
        assert calls[0]["headers"]["Authorization"] == "Bearer ek-key"


class TestDownloadImage:
    def test_gets_the_url_and_returns_the_bytes(self, monkeypatch) -> None:
        calls: list[dict] = []

        def fake_get(url, headers=None, timeout=None):
            calls.append({"url": url, "headers": headers})
            return FakeResponse(content=b"\x89PNG-image-bytes")

        monkeypatch.setattr(ev.requests, "get", fake_get)
        assert ev.download_image("https://files.evolink.ai/u/a.png") == b"\x89PNG-image-bytes"
        assert calls[0]["url"] == "https://files.evolink.ai/u/a.png"
        # Result URLs are pre-signed and may reject unexpected headers.
        assert calls[0]["headers"] is None
