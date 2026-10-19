"""Evolink (api.evolink.ai) image-series API client for Canvas Copilot.

The gpt-image-2.5 models on Evolink run on an **async task API**, not the
OpenAI `/images` endpoints:

1. `POST /v1/images/generations` creates a task and returns its `id`.
2. `GET /v1/tasks/{id}` is polled until the task is `completed` or `failed`;
   result image URLs are valid for 24 hours.
3. Reference images must reach Evolink as **publicly downloadable URLs**, so
   local bytes are first uploaded through their file service (also used by
   Prompt Generation, since `input_image` needs a URL over there too).

Pure helpers here (payload building, poll loop, result extraction) are tested
with injected fetch/sleep callables, keeping the offline test suite offline.
"""

from __future__ import annotations

import base64
import time
from collections.abc import Callable
from typing import Any

import requests

# Task lifecycle statuses returned by GET /v1/tasks/{id}.
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
ACTIVE_STATUSES = frozenset({"pending", "processing"})

DEFAULT_POLL_INTERVAL_S = 3.0
DEFAULT_POLL_TIMEOUT_S = 600.0

BASE_URL = "https://api.evolink.ai"
FILE_API_URL = "https://files-api.evolink.ai"
UPLOAD_TIMEOUT_S = 120.0
REQUEST_TIMEOUT_S = 120.0


class TaskPollTimeout(RuntimeError):
    """A task stayed non-terminal until the polling budget ran out."""


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested offline)
# ---------------------------------------------------------------------------

def to_data_url(image_bytes: bytes, image_mime: str) -> str:
    """Encode raw image bytes as a `data:...;base64,...` URL."""
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{image_mime};base64,{encoded}"


def build_upload_payload(image_bytes: bytes, image_mime: str, file_name: str | None = None) -> dict[str, str]:
    """Body for POST /api/v1/files/upload/base64 (files-api.evolink.ai)."""
    payload = {"base64_data": to_data_url(image_bytes, image_mime)}
    if file_name:
        payload["file_name"] = file_name
    return payload


def build_generation_payload(
    model: str,
    prompt: str,
    *,
    size: str | None = None,
    quality: str | None = None,
    n: int | None = None,
    output_format: str | None = None,
    image_urls: list[str] | None = None,
    resolution: str | None = None,
    model_params: dict[str, Any] | None = None,
    negative_prompt: str | None = None,
) -> dict[str, Any]:
    """Body for POST /v1/images/generations — model-agnostic.

    Every optional field is omitted when not given, so each model's exact
    schema is respected by the caller passing only what that model supports
    (see `core/model_registry.py`). Note Evolink has no `moderation` field and
    no `output_compression`, and `quality` has no `auto` tier.
    """
    payload: dict[str, Any] = {"model": model, "prompt": prompt}
    if size is not None:
        payload["size"] = size
    if quality is not None:
        payload["quality"] = quality
    if n is not None:
        payload["n"] = n
    if output_format is not None:
        payload["output_format"] = output_format
    if image_urls:
        payload["image_urls"] = list(image_urls)
    if resolution is not None:
        payload["resolution"] = resolution
    if model_params:
        payload["model_params"] = dict(model_params)
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    return payload


def poll_task(
    fetch_task: Callable[[str], dict],
    task_id: str,
    *,
    sleep: Callable[[float], None] = time.sleep,
    interval_s: float = DEFAULT_POLL_INTERVAL_S,
    timeout_s: float = DEFAULT_POLL_TIMEOUT_S,
    on_progress: Callable[[dict], None] | None = None,
) -> dict:
    """Poll `GET /v1/tasks/{id}` until a terminal status or the budget runs out.

    `fetch_task` performs one status query; `sleep` is injectable so tests can
    run this loop fully offline. The final task dict is returned for both
    `completed` and `failed` — callers decide how to handle a failure.
    """
    elapsed_s = 0.0
    while True:
        task = fetch_task(task_id)
        if on_progress is not None:
            on_progress(task)
        status = task.get("status")
        if status in (STATUS_COMPLETED, STATUS_FAILED):
            return task
        if elapsed_s + interval_s > timeout_s:
            raise TaskPollTimeout(
                f"Evolink task {task_id} did not finish within {int(timeout_s)}s "
                f"(last status: {status})."
            )
        sleep(interval_s)
        elapsed_s += interval_s


def extract_result_urls(task: dict) -> list[str]:
    """Image URLs from a completed task; `results`, else `result_data` URLs."""
    results = task.get("results")
    if results:
        return [str(url) for url in results]
    result_data = task.get("result_data") or []
    return [item["url"] for item in result_data if item.get("url")]


def describe_failure(task: dict) -> str:
    """Human-readable message from a failed task's error block."""
    error = task.get("error") or {}
    parts = [error.get("code"), error.get("message"), error.get("details")]
    joined = " — ".join(str(part) for part in parts if part)
    return joined or f"Task {task.get('id', '')} failed without error details."


# ---------------------------------------------------------------------------
# HTTP wrappers
# ---------------------------------------------------------------------------

def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def upload_image(
    api_key: str,
    image_bytes: bytes,
    image_mime: str,
    file_name: str | None = None,
) -> str:
    """Upload local image bytes and return a publicly downloadable URL."""
    response = requests.post(
        f"{FILE_API_URL}/api/v1/files/upload/base64",
        json=build_upload_payload(image_bytes, image_mime, file_name),
        headers=_headers(api_key),
        timeout=UPLOAD_TIMEOUT_S,
    )
    response.raise_for_status()
    data = response.json().get("data") or {}
    file_url = data.get("file_url")
    if not file_url:
        raise RuntimeError(f"Evolink file upload returned no URL: {response.json()}")
    return str(file_url)


def create_generation_task(api_key: str, payload: dict[str, Any]) -> dict:
    """Submit POST /v1/images/generations and return the created task."""
    response = requests.post(
        f"{BASE_URL}/v1/images/generations",
        json=payload,
        headers=_headers(api_key),
        timeout=REQUEST_TIMEOUT_S,
    )
    response.raise_for_status()
    task = response.json()
    if not task.get("id"):
        raise RuntimeError(f"Evolink did not return a task id: {task}")
    return task


def fetch_task(api_key: str, task_id: str) -> dict:
    """Query GET /v1/tasks/{id} for one status update."""
    response = requests.get(
        f"{BASE_URL}/v1/tasks/{task_id}",
        headers=_headers(api_key),
        timeout=REQUEST_TIMEOUT_S,
    )
    response.raise_for_status()
    return response.json()


def download_image(url: str) -> bytes:
    """Fetch a result image (no auth header — the URLs are pre-signed)."""
    response = requests.get(url, timeout=REQUEST_TIMEOUT_S)
    response.raise_for_status()
    return response.content
