"""Image Generation mode — text-to-image generation with OpenAI's GPT Image models."""

from __future__ import annotations

import base64
import io

import streamlit as st
from openai import BadRequestError

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


def render(client) -> None:
    st.title("🎨 Canvas Copilot")
    st.caption("Describe anything — GPT Image paints it.")

    # --- Input section ----------------------------------------------------------

    prompt = st.text_area(
        "Prompt",
        key="image_gen_prompt",
        placeholder="A quiet harbor at dawn, fishing boats resting on glassy water…",
        height=120,
    )

    # Optional reference images. With references attached, the app switches from
    # the generations endpoint to the edits endpoint (client.images.edit).
    reference_files = st.file_uploader(
        "Reference images (optional)",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        help=(
            "Attach up to 16 reference images (PNG, JPEG, or WEBP, <50MB each). "
            "The model will use them as style, subject, or composition references."
        ),
    )

    if reference_files:
        ref_cols = st.columns(len(reference_files))
        for col, file in zip(ref_cols, reference_files):
            with col:
                st.image(file, caption=file.name, width="stretch")

    model_col, size_col, quality_col, moderation_col = st.columns(4)
    with model_col:
        model = st.selectbox(
            "Model",
            list(MODEL_OPTIONS),
            format_func=MODEL_OPTIONS.get,
            index=0,
            help="Sunburst wins on editing precision. Flare trades precision for faster everyday generation.",
        )
    with size_col:
        size = st.selectbox("Size", list(SIZE_OPTIONS), format_func=SIZE_OPTIONS.get)
    with quality_col:
        quality = st.selectbox("Quality", list(QUALITY_OPTIONS), format_func=QUALITY_OPTIONS.get)
    with moderation_col:
        moderation = st.selectbox(
            "Moderation",
            list(MODERATION_OPTIONS),
            format_func=MODERATION_OPTIONS.get,
            help=(
                "Controls content-moderation strictness. auto: standard filtering "
                "that limits potentially age-inappropriate content. low: less "
                "restrictive filtering."
            ),
        )

    variations_col, format_col, compression_col = st.columns(3)
    with variations_col:
        variations = st.slider(
            "Variations",
            min_value=1,
            max_value=10,
            value=1,
            help="Generate multiple candidates in one request. Each variation is an additional API cost.",
        )
    with format_col:
        output_format = st.selectbox(
            "Format",
            list(FORMAT_OPTIONS),
            format_func=FORMAT_OPTIONS.get,
            help="JPEG renders faster than PNG; WEBP and JPEG support compression.",
        )
    with compression_col:
        if output_format in ("jpeg", "webp"):
            compression = st.slider(
                "Compression",
                0,
                100,
                100,
                step=5,
                help="Lower values compress the image harder (smaller files, more artifacts). 100 = best quality.",
            )
        else:
            st.caption("Compression applies to JPEG / WEBP only.")
            compression = None

    generate = st.button("Generate", type="primary", width="stretch")

    if generate:
        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            st.warning("Please enter a prompt describing the image you want to create.")
        else:
            try:
                # Compression only applies to JPEG and WEBP; omit it for PNG so the
                # server default (none) applies instead of sending an explicit 0.
                compression_payload = (
                    {"output_compression": compression}
                    if compression is not None
                    else {}
                )
                with st.spinner("Painting your image…"):
                    if reference_files:
                        # The edits endpoint expects file-like objects, not raw bytes.
                        input_images = []
                        for file in reference_files:
                            buffer = io.BytesIO(file.getvalue())
                            buffer.name = file.name
                            input_images.append(buffer)
                        result = client.images.edit(
                            model=model,
                            image=input_images,
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
                    st.error(f"Image generation failed: {error}")
            except Exception as exc:
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
