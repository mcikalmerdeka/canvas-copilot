"""Image Generation mode — text-to-image generation with GPT Image models.

Works with both providers selected in the sidebar:

- **OpenAI** (default): the sync `/images` endpoints (`images.generate` / `images.edit`),
  multipart uploads for reference images, responses come back base64.
- **Evolink**: the async task API (`evolink_api.py`) with a **model catalog**
  (`model_registry.py`) — GPT Image 2.5, Nano Banana 2.1, Grok Imagine 2.0,
  Midjourney v8.2, and Qwen Image 3.0 / Pro. Each model's option menus, limits,
  and payload shape come from its registry spec.
"""

from __future__ import annotations

import base64
import io
import logging
import time

import streamlit as st
from openai import BadRequestError

from canvas_copilot.core import evolink_api as ev
from canvas_copilot.core import model_registry as mr

logger = logging.getLogger(__name__)

MODEL_OPTIONS = {
    "gpt-image-2.5-sunburst": "Sunburst (best editing precision)",
    "gpt-image-2.5-flare": "Flare (faster, everyday quality)",
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
    """Render the mode. `ctx` is a providers.ProviderContext."""
    is_evolink = ctx.is_evolink
    client = ctx.client

    st.title("🎨 Canvas Copilot")
    st.caption("Describe anything — GPT Image paints it.")

    # --- Per-model spec (Evolink) ----------------------------------------------

    if is_evolink:
        model_id = st.selectbox(
            "Model",
            list(mr.EVOLINK_MODELS),
            format_func=lambda mid: mr.EVOLINK_MODELS[mid].label,
            key="ig_model_id",
            index=0,
        )
        spec = mr.EVOLINK_MODELS[model_id]
        size_menu = dict(spec.size_options) if spec.size_options else None
        aspect_menu = MJ_ASPECT_OPTIONS if spec.aspect_via_prompt else None
        quality_menu = dict(spec.quality_options)
        quality_index = list(quality_menu).index(spec.quality_default)
        resolution_menu = dict(spec.resolution_options) if spec.resolution_options else None
        speed_menu = dict(spec.speed_options) if spec.speed_options else None
    else:
        spec = None
        size_menu = SIZE_OPTIONS
        aspect_menu = None
        quality_menu = QUALITY_OPTIONS
        quality_index = 0  # `auto` for the official API
        resolution_menu = None
        speed_menu = None

    # --- Input section ----------------------------------------------------------

    prompt = st.text_area(
        "Prompt",
        key="image_gen_prompt",
        placeholder="A quiet harbor at dawn, fishing boats resting on glassy water…",
        height=120,
    )

    # Optional reference images. With inputs attached, OpenAI switches from the
    # generations endpoint to the edits endpoint; Evolink models receive them
    # as uploaded file URLs (Midjourney gets them inside the prompt text).
    reference_cap = spec.max_reference_images if spec else 16
    reference_help = (
        f"Attach up to {reference_cap} reference images (PNG, JPEG, or WEBP, <50MB each). "
        "The model will use them as style, subject, or composition references."
    )
    if spec and spec.references_in_prompt:
        reference_help += " Their URLs are added at the start of your prompt (Midjourney's native image-to-image)."
    reference_files = st.file_uploader(
        "Reference images (optional)",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        help=reference_help,
    )

    if reference_files:
        ref_cols = st.columns(len(reference_files))
        for col, file in zip(ref_cols, reference_files):
            with col:
                st.image(file, caption=file.name, width="stretch")

    # Image handed off from Prompt Generation ("💡 Use in Image Generation").
    # Streamlit's uploader can't be pre-filled, so the image travels via session
    # state and stays attached until removed or replaced by another handoff.
    if st.session_state.get("pending_handoff_image"):
        st.session_state["handoff_image"] = st.session_state.pop("pending_handoff_image")
    handoff_image = st.session_state.get("handoff_image")
    if handoff_image:
        st.image(
            handoff_image["bytes"],
            caption=f"{handoff_image['name']} · attached from Prompt Generation",
            width="stretch",
        )
        if st.button("✕ Remove attached image"):
            st.session_state.pop("handoff_image", None)
            st.rerun()

    # --- Option menus -------------------------------------------------------------

    # Row 1: model geometry. Size/aspect in the second slot; quality third;
    # a fourth slot appears only for models with their own resolution or speed
    # parameter (Grok resolution, Midjourney speed).
    has_fourth = bool(resolution_menu or speed_menu)
    columns = st.columns(4) if has_fourth else st.columns(3)
    with columns[0]:
        if is_evolink:
            st.caption("")  # keeps the Model selectbox (rendered above) aligned
        else:
            model = st.selectbox(
                "Model",
                list(MODEL_OPTIONS),
                format_func=MODEL_OPTIONS.get,
                index=0,
                help="Sunburst wins on editing precision. Flare trades precision for faster everyday generation.",
            )
    with columns[1]:
        if size_menu:
            size = st.selectbox(
                "Size", list(size_menu), format_func=size_menu.get, key=f"ig_size_{spec.model_id if spec else 'openai'}"
            )
        elif aspect_menu:
            size = st.selectbox(
                "Aspect ratio (added as --ar to the prompt)",
                [value for value, _ in aspect_menu],
                format_func=dict(aspect_menu).get,
                key=f"ig_aspect_{spec.model_id}",
            )
        else:
            size = "auto"
    with columns[2]:
        quality = st.selectbox(
            "Quality",
            list(quality_menu),
            format_func=quality_menu.get,
            index=quality_index,
            help=(
                "Rendering quality — higher tiers cost more output tokens."
                if is_evolink
                else "OpenAI decides automatically; tiers trade cost for detail."
            ),
        )
    if resolution_menu or speed_menu:
        with columns[3]:
            if resolution_menu:
                resolution = st.selectbox(
                    "Resolution",
                    list(resolution_menu),
                    format_func=resolution_menu.get,
                    index=list(resolution_menu).index(spec.resolution_default),
                )
            else:
                speed = st.selectbox(
                    "Speed",
                    list(speed_menu),
                    format_func=speed_menu.get,
                    index=0,  # default first
                    help="Draft returns 24 lightweight 0.5K sketch images instead of 4 finished ones.",
                )

    # Row 2: variations + format + compression.
    variations_label = "Variations"
    variations_help = "Generate multiple candidates in one request. Each variation is an additional API cost."
    variations_max = spec.max_variations if spec else 10
    row2 = st.columns(3)
    with row2[0]:
        if variations_max > 1:
            variations = st.slider(
                variations_label, min_value=1, max_value=variations_max, value=1, help=variations_help,
            )
        elif spec and spec.references_in_prompt:
            st.caption("Midjourney returns 4 images per generation (24 in Draft mode).")
            variations = None
        else:
            st.caption("This model returns one image per request.")
            variations = None
    with row2[1]:
        if is_evolink and spec and spec.supports_output_format:
            output_format = st.selectbox(
                "Format",
                list(FORMAT_OPTIONS),
                format_func=FORMAT_OPTIONS.get,
                help="JPEG renders faster than PNG; WEBP and JPEG support compression.",
            )
        elif is_evolink:
            st.caption("PNG (fixed by this model)")
            output_format = "png"
        else:
            output_format = st.selectbox(
                "Format",
                list(FORMAT_OPTIONS),
                format_func=FORMAT_OPTIONS.get,
                help="JPEG renders faster than PNG; WEBP and JPEG support compression.",
            )
    with row2[2]:
        if not is_evolink and output_format in ("jpeg", "webp"):
            compression = st.slider(
                "Compression",
                0,
                100,
                100,
                step=5,
                help="Lower values compress the image harder (smaller files, more artifacts). 100 = best quality.",
            )
        elif is_evolink and spec and spec.supports_output_format and output_format in ("jpeg", "webp"):
            compression = st.slider(
                "Compression",
                0,
                100,
                100,
                step=5,
                help="Lower values compress the image harder (smaller files, more artifacts). 100 = best quality.",
            )
        elif is_evolink:
            st.caption("Compression is not offered by Evolink.")
            compression = None
        else:
            st.caption("Compression applies to JPEG / WEBP only.")
            compression = None

    # Per-model extras.
    negative_prompt = None
    if spec and spec.supports_negative_prompt:
        negative_prompt = st.text_area(
            "Negative prompt (optional)",
            key="ig_negative_prompt",
            placeholder="low resolution, extra fingers, watermark…",
            height=70,
            help="Content to keep out of the image (max 500 characters).",
        ) or None

    # OpenAI only: Evolink's task API has no moderation field (and the tests
    # pin that it must never send one).
    moderation = None
    if not is_evolink:
        moderation = st.selectbox(
            "Moderation",
            list(MODERATION_OPTIONS),
            format_func=MODERATION_OPTIONS.get,
            help="Low relaxes the standard content filter for edge-case requests.",
        )

    generate = st.button("Generate", type="primary", width="stretch")

    if generate:
        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            st.warning("Please enter a prompt describing the image you want to create.")
        else:
            too_many = validate_reference_count(spec, len(collect_input_records(reference_files, handoff_image))) if spec else None
            if too_many:
                st.warning(too_many)
            else:
                try:
                    input_records = collect_input_records(reference_files, handoff_image)
                    start_time = time.perf_counter()
                    logger.info(
                        "Image generation started (model=%s, size=%s, n=%s, refs=%d)",
                        spec.model_id if spec else model,
                        size,
                        variations,
                        len(input_records),
                    )
                    with st.spinner("Painting your image…"):
                        progress_bar = st.progress(0)
                        if is_evolink:
                            generated_images = generate_images_evolink(
                                ctx.api_key,
                                spec,
                                cleaned_prompt,
                                input_records,
                                size=size,
                                quality=quality,
                                n=variations,
                                output_format=output_format if spec.supports_output_format else None,
                                resolution=resolution if resolution_menu else None,
                                model_params={"speed": speed} if speed_menu else None,
                                negative_prompt=negative_prompt,
                                on_progress=lambda task: progress_bar.progress(
                                    min(int(task.get("progress") or 0), 100)
                                ),
                            )
                            progress_bar.progress(100)
                        else:
                            # The edits endpoint expects file-like objects, not raw bytes.
                            input_buffers = [to_named_buffer(record) for record in input_records]
                            # Compression only applies to JPEG and WEBP; omit it for PNG so the
                            # server default (none) applies instead of sending an explicit 0.
                            compression_payload = (
                                {"output_compression": compression}
                                if compression is not None
                                else {}
                            )
                            if input_buffers:
                                result = client.images.edit(
                                    model=model,
                                    image=input_buffers,
                                    prompt=cleaned_prompt,
                                    size=size,
                                    quality=quality,
                                    n=variations,
                                    output_format=output_format,
                                    # SDK 3.13 has no typed `moderation` param on edit();
                                    # the /images/edits endpoint still accepts it in the body.
                                    extra_body={"moderation": moderation},
                                    **compression_payload,
                                )
                            else:
                                result = client.images.generate(
                                    model=model,
                                    prompt=cleaned_prompt,
                                    size=size,
                                    quality=quality,
                                    moderation=moderation,
                                    n=variations,
                                    output_format=output_format,
                                    **compression_payload,
                                )
                            generated_images = [
                                base64.b64decode(item.b64_json) for item in result.data
                            ]
                    logger.info(
                        "Image generation finished: %d image(s) in %.1fs",
                        len(generated_images),
                        time.perf_counter() - start_time,
                    )
                    # Keep the results so they survive reruns (e.g. download-button clicks).
                    st.session_state["last_images"] = generated_images
                    st.session_state["last_prompt"] = cleaned_prompt
                    st.session_state["last_format"] = output_format
                except BadRequestError as error:
                    if error.code == "moderation_blocked":
                        # Per the OpenAI docs: keep the primary message generic and use
                        # moderation_details (stage + categories) for remediation hints.
                        error_body = error.body if isinstance(error.body, dict) else {}
                        details = error_body.get("moderation_details") or {}
                        categories = details.get("categories") or []
                        stage = details.get("moderation_stage")

                        logger.warning(
                            "Request blocked by moderation (stage=%s, categories=%s)",
                            stage,
                            categories,
                        )

                        hint = "Try changing the prompt and generating again."
                        if "harassment" in categories:
                            hint = "Try removing abusive or targeting language and focus on neutral visual details instead."
                        elif stage == "input":
                            hint = "Try revising the prompt or reference images and submit the request again."
                        elif stage == "output":
                            hint = "The generated result was blocked by a safety check. Try changing the prompt and generating again."

                        st.error(f"Your request was blocked by content moderation. {hint}")
                        with st.expander("Show moderation details"):
                            st.json({"moderation_stage": stage, "categories": categories})
                    else:
                        logger.error("Image generation failed: %s", error)
                        st.error(f"Image generation failed: {error}")
                except Exception as exc:  # noqa: BLE001 — surface any API failure in the UI
                    logger.error("Image generation failed: %s", exc)
                    st.error(f"Image generation failed: {exc}")

    # --- Result section -----------------------------------------------------------

    if st.session_state.get("last_images"):
        st.divider()
        display_format = st.session_state.get("last_format", "png")
        image_mime = FORMAT_MIME.get(display_format, "image/png")
        last_images = st.session_state["last_images"]
        prompt_caption = st.session_state.get("last_prompt")

        if len(last_images) == 1:
            st.image(last_images[0], caption=prompt_caption, width="stretch")
            st.download_button(
                "Download image",
                data=last_images[0],
                file_name=f"generated_image.{display_format}",
                mime=image_mime,
                width="stretch",
            )
        else:
            st.caption(prompt_caption)
            for row in range(0, len(last_images), 2):
                tiles = st.columns(2)
                for tile, idx in zip(tiles, range(row, row + 2)):
                    if idx >= len(last_images):
                        continue
                    with tile:
                        st.image(last_images[idx], caption=f"Variant {idx + 1}", width="stretch")
                        st.download_button(
                            f"Download #{idx + 1}",
                            data=last_images[idx],
                            file_name=f"generated_image_{idx + 1}.{display_format}",
                            mime=image_mime,
                            key=f"download_{idx}",
                            width="stretch",
                        )
