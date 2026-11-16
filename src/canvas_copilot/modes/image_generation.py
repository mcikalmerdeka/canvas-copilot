"""Image Generation mode — text-to-image generation with GPT Image models.

Works with both providers selected in the sidebar:

- **OpenAI** (default): the sync `/images` endpoints (`images.generate` / `images.edit`),
  multipart uploads for reference images, responses come back base64.
- **Evolink**: the async task API (`evolink_api.py`) with a **model catalog**
  (`model_registry.py`) — GPT Image 2.6 / 2.5, Nano Banana 2.1, Grok Imagine
  2.0, Midjourney v8.2, and Qwen Image 3.0 / Pro. Each model's option menus,
  limits, and payload shape come from its registry spec.
"""

from __future__ import annotations

import io
import logging

from canvas_copilot.core import evolink_api as ev
from canvas_copilot.core import model_registry as mr

logger = logging.getLogger(__name__)

MODEL_OPTIONS = {
    "gpt-image-2.6-sunburst": "GPT Sunburst 2.6",
    "gpt-image-2.6-flare": "GPT Flare 2.6",
    "gpt-image-2.5-sunburst": "GPT Sunburst 2.5",
    "gpt-image-2.5-flare": "GPT Flare 2.5",
}
SIZE_OPTIONS = {
    "auto": "Auto",
    "1024x1024": "1024 × 1024 · Square",
    "1024x1536": "1024 × 1536 · Portrait",
    "1536x1024": "1536 × 1024 · Landscape",
    "2048x1152": "2048 × 1152 · Widescreen (16:9)",
    "1152x2048": "1152 × 2048 · Phone (9:16)",
}
QUALITY_OPTIONS = {
    "auto": "Auto",
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "xhigh": "X-High",
    "max": "Max",
}
MODERATION_OPTIONS = {
    "auto": "Auto (standard filtering)",
    "low": "Low (less restrictive)",
}
FORMAT_OPTIONS = {
    "png": "PNG",
    "jpeg": "JPEG",
    "webp": "WEBP",
}
FORMAT_MIME = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}

# Midjourney's API route has no size parameter: the aspect ratio is written
# into the prompt as --ar (see model_registry.ASPECT_RATIOS_MJ).
MJ_ASPECT_OPTIONS = tuple([("auto", "Auto"), *[(ratio, ratio) for ratio in mr.ASPECT_RATIOS_MJ]])


def collect_input_records(reference_files, handoff_image: dict | None) -> list[dict]:
    """Merge the handoff image (from Prompt Generation) and uploaded references
    into one ordered list of `{bytes, mime, name}` input records."""
    records: list[dict] = []
    if handoff_image:
        records.append({
            "bytes": handoff_image["bytes"],
            "mime": handoff_image["mime"],
            "name": handoff_image["name"],
        })
    for file in reference_files or []:
        records.append({
            "bytes": file.getvalue(),
            "mime": file.type or "image/png",
            "name": file.name,
        })
    return records


def validate_reference_count(spec, count: int) -> str | None:
    """None when the reference count fits the model, else a warning message."""
    cap = spec.max_reference_images
    if count > cap:
        return f"{spec.label} accepts up to {cap} reference image(s) — remove {count - cap}."
    return None


def to_named_buffer(record: dict) -> io.BytesIO:
    """Wrap an input record's bytes as a named buffer, as the edits endpoint expects."""
    buffer = io.BytesIO(record["bytes"])
    buffer.name = record["name"]
    return buffer


def generate_images_evolink(
    api_key: str,
    spec,
    prompt: str,
    input_records: list[dict],
    *,
    size: str | None = None,
    quality: str | None = None,
    resolution: str | None = None,
    n: int | None = None,
    output_format: str | None = None,
    model_params: dict | None = None,
    negative_prompt: str | None = None,
    on_progress=None,
) -> list[bytes]:
    """One Evolink image task end-to-end: upload refs → create → poll → download.

    `spec` (a model_registry.EvolinkModel) drives the payload shape: params the
    model doesn't support are omitted, Midjourney's references travel inside
    the prompt and its aspect ratio becomes an --ar flag.

    Returns the finished image bytes in the order the server returned them.
    Raises RuntimeError with the server's reason when the task fails or comes
    back empty.
    """
    reference_urls: list[str] = []
    if input_records and spec.max_reference_images > 0:
        for reference_index, record in enumerate(input_records, start=1):
            reference_urls.append(
                ev.upload_image(api_key, record["bytes"], record["mime"], record["name"])
            )
            logger.info(
                "Reference image uploaded (%d/%d): %s",
                reference_index,
                len(input_records),
                record["name"],
            )

    model_prompt = mr.build_model_prompt(spec, prompt, reference_urls)
    if spec.aspect_via_prompt:
        aspect_flag = mr.aspect_prompt_flag(size)
        if aspect_flag:
            model_prompt = f"{model_prompt} {aspect_flag}".strip()
        payload_size = None
    else:
        payload_size = size

    if spec.aspect_via_prompt and len(model_prompt) > 2048:
        raise RuntimeError(
            f"Midjourney prompts are limited to 2048 characters (got {len(model_prompt)}). "
            "Shorten the prompt or remove reference images."
        )

    payload = ev.build_generation_payload(
        spec.model_id,
        model_prompt,
        size=payload_size,
        quality=quality,
        n=n,
        output_format=output_format if spec.supports_output_format else None,
        # Midjourney receives references inside the prompt — no image_urls key.
        image_urls=None if spec.references_in_prompt else (reference_urls or None),
        resolution=resolution,
        model_params=model_params,
        negative_prompt=negative_prompt,
    )
    task = ev.create_generation_task(api_key, payload)
    logger.info("Generation task created: %s", task["id"])
    final_task = ev.poll_task(
        lambda task_id: ev.fetch_task(api_key, task_id),
        task["id"],
        on_progress=on_progress,
    )
    if final_task["status"] != ev.STATUS_COMPLETED:
        raise RuntimeError(ev.describe_failure(final_task))
    urls = ev.extract_result_urls(final_task)
    if not urls:
        raise RuntimeError("Evolink marked the task completed but returned no images.")
    logger.info("Task completed: %d image(s) ready to download", len(urls))
    return [ev.download_image(url) for url in urls]


def render(ctx) -> None:
    """Render the Image Generation workspace (UI lives in image_views)."""
    from canvas_copilot.modes import image_views

    image_views.render_mode(ctx)
