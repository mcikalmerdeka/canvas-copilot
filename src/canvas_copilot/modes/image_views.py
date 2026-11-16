"""Image Generation UI.

All request shaping and provider branching lives in `image_generation.py`; this
module only lays out the prompt, references, settings, and results. Keeping the
view separate makes the generation logic testable without Streamlit.
"""

from __future__ import annotations

import base64
import time

import streamlit as st
from openai import BadRequestError

from canvas_copilot import ui
from canvas_copilot.core import model_registry as mr
from canvas_copilot.modes import image_generation as ig

logger = ig.logger

MJ_ASPECT_OPTIONS = ig.MJ_ASPECT_OPTIONS


def _render_reference_previews(reference_files) -> None:
    if not reference_files:
        return
    for start in range(0, len(reference_files), 3):
        columns = st.columns(3, gap="small")
        for column, file in zip(columns, reference_files[start : start + 3]):
            with column:
                st.image(file, caption=file.name, width="stretch")


def _render_prompt_card() -> None:
    with st.container(border=True):
        ui.section(
            "Prompt",
            "Describe the image you want. Concrete nouns and a clear light direction beat adjective piles.",
        )
        st.text_area(
            "Prompt",
            key="image_gen_prompt",
            placeholder="A quiet harbor at dawn, fishing boats resting on glassy water…",
            height=220,
            label_visibility="collapsed",
        )


def _render_reference_card(spec) -> list:
    reference_cap = spec.max_reference_images if spec else 16
    reference_help = (
        f"Attach up to {reference_cap} reference images (PNG, JPEG, or WEBP, under 50MB each). "
        "The model uses them as style, subject, or composition references."
    )
    if spec and spec.references_in_prompt:
        reference_help += (
            " Their URLs are added at the start of your prompt (Midjourney's native image-to-image)."
        )

    with st.container(border=True):
        ui.section("Reference images", "Optional. Attach references for style, subject, or composition.")
        reference_files = st.file_uploader(
            "Reference images",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            help=reference_help,
            label_visibility="collapsed",
        )
        _render_reference_previews(reference_files)

        # Image handed off from Prompt Generation ("Use in Image Generation").
        # Streamlit's uploader can't be pre-filled, so the image travels via
        # session state and stays attached until removed or replaced.
        if st.session_state.get("pending_handoff_image"):
            st.session_state["handoff_image"] = st.session_state.pop("pending_handoff_image")
        handoff_image = st.session_state.get("handoff_image")
        if handoff_image:
            st.divider()
            st.image(
                handoff_image["bytes"],
                caption=f"{handoff_image['name']} · attached from Prompt Generation",
                width="stretch",
            )
            if st.button("Remove attached image", icon=":material/close:"):
                st.session_state.pop("handoff_image", None)
                st.rerun()

    return reference_files


def _render_settings(is_evolink: bool) -> dict[str, object]:
    """Render every output control and return the values the request needs."""
    settings: dict[str, object] = {}

    with st.container(border=True):
        ui.section("Output settings", "Model, geometry, and provider-specific controls.")

        if is_evolink:
            model_id = st.selectbox(
                "Model",
                list(mr.EVOLINK_MODELS),
                format_func=lambda mid: mr.EVOLINK_MODELS[mid].label,
                key="ig_model_id",
                index=0,
            )
            spec = mr.EVOLINK_MODELS[model_id]
        else:
            spec = None
            settings["model"] = st.selectbox(
                "Model",
                list(ig.MODEL_OPTIONS),
                format_func=ig.MODEL_OPTIONS.get,
                index=0,
                help="2.6 is the newest family; 2.5 remains selectable.",
            )

        settings["spec"] = spec
        size_menu = dict(spec.size_options) if spec and spec.size_options else (None if spec else ig.SIZE_OPTIONS)
        aspect_menu = MJ_ASPECT_OPTIONS if spec and spec.aspect_via_prompt else None
        quality_menu = dict(spec.quality_options) if spec and spec.quality_options else (None if spec else ig.QUALITY_OPTIONS)
        quality_index = list(quality_menu).index(spec.quality_default) if quality_menu and spec else 0
        resolution_menu = dict(spec.resolution_options) if spec and spec.resolution_options else None
        speed_menu = dict(spec.speed_options) if spec and spec.speed_options else None

        if size_menu:
            settings["size"] = st.selectbox(
                "Size",
                list(size_menu),
                format_func=size_menu.get,
                key=f"ig_size_{spec.model_id if spec else 'openai'}",
            )
        elif aspect_menu:
            settings["size"] = st.selectbox(
                "Aspect ratio (added as --ar to the prompt)",
                [value for value, _ in aspect_menu],
                format_func=dict(aspect_menu).get,
                key=f"ig_aspect_{spec.model_id}",
            )
        else:
            settings["size"] = "auto"

        if quality_menu:
            settings["quality"] = st.selectbox(
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
        else:
            st.caption("Quality is controlled by the resolution tier.")
            settings["quality"] = None

        if resolution_menu:
            settings["resolution"] = st.selectbox(
                "Resolution",
                list(resolution_menu),
                format_func=resolution_menu.get,
                index=list(resolution_menu).index(spec.resolution_default),
            )
        else:
            settings["resolution"] = None

        if speed_menu:
            settings["speed"] = st.selectbox(
                "Speed",
                list(speed_menu),
                format_func=speed_menu.get,
                index=0,
                help="Draft returns 24 lightweight 0.5K sketch images instead of 4 finished ones.",
            )
        else:
            settings["speed"] = None

        variations_max = spec.max_variations if spec else 10
        if variations_max > 1:
            settings["variations"] = st.slider(
                "Variations",
                min_value=1,
                max_value=variations_max,
                value=1,
                help="Generate multiple candidates in one request. Each variation is an additional API cost.",
            )
        elif spec and spec.references_in_prompt:
            st.caption("Midjourney returns 4 images per generation (24 in Draft mode).")
            settings["variations"] = None
        else:
            st.caption("This model returns one image per request.")
            settings["variations"] = None

        format_menu = dict(spec.output_format_options or ig.FORMAT_OPTIONS) if is_evolink and spec else ig.FORMAT_OPTIONS
        if is_evolink and spec and not spec.supports_output_format:
            st.caption("PNG (fixed by this model)")
            settings["output_format"] = "png"
        else:
            settings["output_format"] = st.selectbox(
                "Format",
                list(format_menu),
                format_func=format_menu.get,
                help="JPEG renders faster than PNG; WEBP and JPEG support compression.",
            )

        output_format = settings["output_format"]
        if not is_evolink and output_format in ("jpeg", "webp"):
            settings["compression"] = st.slider(
                "Compression",
                0,
                100,
                100,
                step=5,
                help="Lower values compress harder (smaller files, more artifacts). 100 = best quality.",
            )
        else:
            settings["compression"] = None

        if spec and spec.supports_negative_prompt:
            settings["negative_prompt"] = st.text_area(
                "Negative prompt (optional)",
                key="ig_negative_prompt",
                placeholder="low resolution, extra fingers, watermark…",
                height=70,
                help="Content to keep out of the image (max 500 characters).",
            ) or None
        else:
            settings["negative_prompt"] = None

        if is_evolink:
            # Evolink's task API has no moderation field (tests pin that it must
            # never send one), so the control is hidden.
            settings["moderation"] = None
        else:
            settings["moderation"] = st.selectbox(
                "Moderation",
                list(ig.MODERATION_OPTIONS),
                format_func=ig.MODERATION_OPTIONS.get,
                help="Low relaxes the standard content filter for edge-case requests.",
            )

        submitted = st.button(
            "Generate",
            type="primary",
            icon=":material/brush:",
            width="stretch",
        )

    settings["submitted"] = submitted
    return settings


def _generate(
    ctx,
    settings: dict[str, object],
    prompt: str,
    reference_files,
) -> None:
    is_evolink = ctx.is_evolink
    client = ctx.client
    spec = settings["spec"]
    output_format = settings["output_format"]

    cleaned_prompt = prompt.strip()
    if not cleaned_prompt:
        st.warning("Please enter a prompt describing the image you want to create.")
        return

    input_records = ig.collect_input_records(
        reference_files, st.session_state.get("handoff_image")
    )
    too_many = ig.validate_reference_count(spec, len(input_records)) if spec else None
    if too_many:
        st.warning(too_many)
        return

    try:
        start_time = time.perf_counter()
        logger.info(
            "Image generation started (model=%s, size=%s, n=%s, refs=%d)",
            spec.model_id if spec else settings["model"],
            settings["size"],
            settings["variations"],
            len(input_records),
        )
        with st.spinner("Painting your image…"):
            progress_bar = st.progress(0)
            if is_evolink:
                generated_images = ig.generate_images_evolink(
                    ctx.api_key,
                    spec,
                    cleaned_prompt,
                    input_records,
                    size=settings["size"],
                    quality=settings["quality"],
                    n=settings["variations"],
                    output_format=output_format if spec.supports_output_format else None,
                    resolution=settings["resolution"],
                    model_params={"speed": settings["speed"]} if settings["speed"] else None,
                    negative_prompt=settings["negative_prompt"],
                    on_progress=lambda task: progress_bar.progress(
                        min(int(task.get("progress") or 0), 100)
                    ),
                )
                progress_bar.progress(100)
            else:
                input_buffers = [ig.to_named_buffer(record) for record in input_records]
                compression_payload = (
                    {"output_compression": settings["compression"]}
                    if settings["compression"] is not None
                    else {}
                )
                if input_buffers:
                    result = client.images.edit(
                        model=settings["model"],
                        image=input_buffers,
                        prompt=cleaned_prompt,
                        size=settings["size"],
                        quality=settings["quality"],
                        n=settings["variations"],
                        output_format=output_format,
                        # SDK 3.13 has no typed `moderation` param on edit();
                        # the /images/edits endpoint still accepts it in the body.
                        extra_body={"moderation": settings["moderation"]},
                        **compression_payload,
                    )
                else:
                    result = client.images.generate(
                        model=settings["model"],
                        prompt=cleaned_prompt,
                        size=settings["size"],
                        quality=settings["quality"],
                        moderation=settings["moderation"],
                        n=settings["variations"],
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
        st.session_state["last_images"] = generated_images
        st.session_state["last_prompt"] = cleaned_prompt
        st.session_state["last_format"] = output_format
        st.toast(
            f"{len(generated_images)} image(s) ready",
            icon=":material/check_circle:",
        )
    except BadRequestError as error:
        if error.code == "moderation_blocked":
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


def _render_results() -> None:
    if not st.session_state.get("last_images"):
        ui.empty_state(
            "Nothing painted yet",
            "Describe a scene, attach references if you have them, then press Generate. Results stay here across downloads.",
        )
        return

    display_format = st.session_state.get("last_format", "png")
    image_mime = ig.FORMAT_MIME.get(display_format, "image/png")
    last_images = st.session_state["last_images"]
    prompt_caption = st.session_state.get("last_prompt", "")

    st.divider()
    ui.section("Results", f"{len(last_images)} image(s) · {display_format.upper()}")
    with st.expander("Prompt used"):
        st.code(prompt_caption, language="text", wrap_lines=True)

    if len(last_images) == 1:
        st.image(last_images[0], width="stretch")
        st.download_button(
            "Download image",
            data=last_images[0],
            file_name=f"generated_image.{display_format}",
            mime=image_mime,
            icon=":material/download:",
            width="stretch",
        )
        return

    for row in range(0, len(last_images), 2):
        tiles = st.columns(2, gap="medium")
        for tile, index in zip(tiles, range(row, row + 2)):
            if index >= len(last_images):
                continue
            with tile:
                st.image(last_images[index], caption=f"Variant {index + 1}", width="stretch")
                st.download_button(
                    f"Download #{index + 1}",
                    data=last_images[index],
                    file_name=f"generated_image_{index + 1}.{display_format}",
                    mime=image_mime,
                    key=f"download_{index}",
                    icon=":material/download:",
                    width="stretch",
                )


def render_mode(ctx) -> None:
    """Render the Image Generation workspace for the active provider."""
    is_evolink = ctx.is_evolink
    left, right = st.columns([1.15, 0.85], gap="large")

    # The model selector lives in the settings card, but references need the
    # model's cap for their help text. Read the current selection first so the
    # left card can render before the right card.
    if is_evolink:
        current_id = st.session_state.get("ig_model_id", next(iter(mr.EVOLINK_MODELS)))
        spec_for_refs = mr.EVOLINK_MODELS.get(current_id, next(iter(mr.EVOLINK_MODELS.values())))
    else:
        spec_for_refs = None

    with left:
        _render_prompt_card()
        reference_files = _render_reference_card(spec_for_refs)

    with right:
        settings = _render_settings(is_evolink)

    if settings["submitted"]:
        _generate(ctx, settings, st.session_state.get("image_gen_prompt", ""), reference_files)

    _render_results()
