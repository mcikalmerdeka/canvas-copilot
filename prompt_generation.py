"""Prompt Generation mode — upgrade raw prompts or reverse-engineer prompts from images.

Pipeline: the chosen format's exemplar (see `prompt_formats.py`) is injected
directly into the `gpt-6-luna` context together with the source material
(a raw text prompt or an uploaded image), and the model returns the finished
prompt in exactly that format.
"""

from __future__ import annotations

import base64
import json

import streamlit as st

from prompt_formats import EDIT_EXEMPLAR, PROMPT_FORMATS, get_format, label_to_id

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

MODEL = "gpt-6-luna"
REASONING_EFFORT = "medium"
MAX_OUTPUT_TOKENS = 20_000

CAPABILITY_UPGRADE = "Upgrade Prompt"
CAPABILITY_IMAGE = "Image-to-Prompt"
CAPABILITY_EDIT = "Image-Edit Prompt"

# Internal pipeline capability codes (what generate_prompt/build_user_content use).
PIPELINE_TEXT = "text"
PIPELINE_IMAGE = "image"
PIPELINE_EDIT = "edit"

SYSTEM_PROMPT_TEMPLATE = """You are an expert prompt engineer for text-to-image generation models.

Your task: rewrite the given source material into ONE finished image-generation prompt that matches the exemplar format below.

Rules:
- Follow the exemplar's structure exactly: same sections/labels/field names, same order, same tone and level of detail.
- Map the source's subject, scene, action, mood, wardrobe, environment, and camera/lighting qualities into that structure.
- Expand plausibly where the source is vague, and never contradict what the source states.
- Depict only clearly adult subjects; never render minors.
- Output only the finished prompt itself — no preamble, no commentary, no code fences.
- {output_rule}

=== EXEMPLAR FORMAT ({format_id}) ===
{exemplar}
=== END EXEMPLAR ==="""

JSON_OUTPUT_RULE = (
    "Respond with a single valid JSON object matching the exemplar's schema and nesting, "
    "and nothing else. Do not wrap it in code fences."
)
TEXT_OUTPUT_RULE = "Respond with plain text styled exactly like the exemplar. Do not wrap it in code fences."

EDIT_SYSTEM_PROMPT_TEMPLATE = """You are an expert prompt engineer for image-EDITING models (image-to-image editing, inpainting, and outpainting).

The attached image is the exact SOURCE to be edited. Your task: rewrite the given edit request into ONE finished image-editing prompt that matches the exemplar style below.

Rules:
- The exemplar contains two finished examples: a removal/restoration edit and an extension/outpainting edit. Follow their structure, tone, and level of detail, keeping only the sections the requested edit needs.
- Treat the attached image as the exact source: every unobstructed part of it must be preserved. Only the elements and areas named in the edit request may change.
- Describe what you can actually see in the source image (subjects, background, palette, lighting, composition) so the edit prompt references the real content.
- Never invent changes beyond the edit request; when it is ambiguous, choose the most conservative interpretation that satisfies it.
- Depict only clearly adult subjects; never render minors.
- Output only the finished prompt itself — no preamble, no commentary. Respond with plain text styled exactly like the exemplars. Do not wrap it in code fences.

=== EDIT EXEMPLARS ===
{exemplar}
=== END EDIT EXEMPLARS ==="""

UPGRADE_USER_TEMPLATE = """Here is the raw prompt idea to upgrade:

\"\"\"
{raw_prompt}
\"\"\"

Upgrade it into the exemplar format above. Output only the upgraded prompt."""

EDIT_USER_TEMPLATE = """Here is the edit request to turn into a finished image-editing prompt:

\"\"\"
{edit_request}
\"\"\"

Study the attached source image closely — subjects, background, palette, lighting, and composition — then write ONE finished image-editing prompt in the exemplar style above. Output only the finished prompt."""

# Edit results render/download through the same result UI as the three formats;
# this stand-in carries the rendering flags the UI needs (no format selection).
EDIT_RESULT_META = {"id": "edit", "is_json": False}

IMAGE_USER_TEXT = (
    "Attached is one image. Study it closely — subject, framing, wardrobe, environment, "
    "lighting, camera characteristics, and mood — then reverse-engineer ONE finished "
    "image-generation prompt that would recreate a photo like it, in the exemplar format above. "
    "Output only the upgraded prompt."
)


# ---------------------------------------------------------------------------
# Pure logic (unit-tested)
# ---------------------------------------------------------------------------

def build_system_prompt(format_id: str) -> str:
    """Assemble the system prompt with the format exemplar injected verbatim."""
    fmt = get_format(format_id)
    output_rule = JSON_OUTPUT_RULE if fmt["is_json"] else TEXT_OUTPUT_RULE
    return SYSTEM_PROMPT_TEMPLATE.format(
        format_id=format_id,
        exemplar=fmt["exemplar_text"],
        output_rule=output_rule,
    )


def build_edit_system_prompt() -> str:
    """Assemble the Image-Edit system prompt with the edit exemplars injected verbatim."""
    return EDIT_SYSTEM_PROMPT_TEMPLATE.format(exemplar=EDIT_EXEMPLAR)


def build_image_data_url(image_bytes: bytes, image_mime: str) -> str:
    """Encode uploaded image bytes as a data URL for the Responses API."""
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{image_mime};base64,{encoded}"


def build_user_content(
    capability: str,
    raw_prompt: str | None = None,
    image_bytes: bytes | None = None,
    image_mime: str | None = None,
    extra_direction: str | None = None,
    edit_request: str | None = None,
) -> str | list[dict]:
    """Build the user message: plain text for upgrades, text + image for image-to-prompt and image-edit."""
    if capability == PIPELINE_IMAGE:
        text = IMAGE_USER_TEXT
        if extra_direction:
            text += f"\n\nExtra direction from the user: {extra_direction.strip()}"
        return [
            {"type": "input_text", "text": text},
            {"type": "input_image", "image_url": build_image_data_url(image_bytes, image_mime)},
        ]
    if capability == PIPELINE_EDIT:
        return [
            {"type": "input_text", "text": EDIT_USER_TEMPLATE.format(edit_request=(edit_request or "").strip())},
            {"type": "input_image", "image_url": build_image_data_url(image_bytes, image_mime)},
        ]
    return UPGRADE_USER_TEMPLATE.format(raw_prompt=raw_prompt.strip())


def strip_code_fences(text: str) -> str:
    """Remove a single wrapping markdown code fence, if the model added one."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        first_line_break = cleaned.find("\n")
        if first_line_break != -1:
            cleaned = cleaned[first_line_break + 1 :]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
    return cleaned.strip()


def try_parse_json(text: str) -> dict | None:
    """Parse model output as a JSON object; return None when it is not one."""
    try:
        parsed = json.loads(strip_code_fences(text))
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def result_filename(fmt: dict[str, object]) -> str:
    """Download filename for a generated prompt, extension driven by the format."""
    return f"prompt_{fmt['id']}{'.json' if fmt['is_json'] else '.md'}"


def extract_output_text(response) -> str:
    """Pull the final text out of a Responses API result object."""
    return getattr(response, "output_text", None) or ""


def generate_prompt(
    client,
    capability: str,
    format_id: str | None = None,
    raw_prompt: str | None = None,
    image_bytes: bytes | None = None,
    image_mime: str | None = None,
    extra_direction: str | None = None,
    edit_request: str | None = None,
) -> str:
    """One blocking call to `gpt-6-luna`; returns the finished prompt text."""
    system_prompt = build_edit_system_prompt() if capability == PIPELINE_EDIT else build_system_prompt(format_id)
    response = client.responses.create(
        model=MODEL,
        reasoning={"effort": REASONING_EFFORT},
        max_output_tokens=MAX_OUTPUT_TOKENS,
        input=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": build_user_content(
                    capability,
                    raw_prompt=raw_prompt,
                    image_bytes=image_bytes,
                    image_mime=image_mime,
                    extra_direction=extra_direction,
                    edit_request=edit_request,
                ),
            },
        ],
    )
    return strip_code_fences(extract_output_text(response))


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def render(client) -> None:
    st.title("🧩 Prompt Generation")
    st.caption(
        "Upgrade a raw idea into a rich prompt, reverse-engineer a prompt from an image, "
        "or turn an edit request into a structured image-editing prompt — with GPT-6 Luna."
    )

    # --- Sidebar: capability + format ---------------------------------------
    capability = st.sidebar.radio(
        "Capability", (CAPABILITY_UPGRADE, CAPABILITY_IMAGE, CAPABILITY_EDIT), key="pg_capability"
    )
    if capability == CAPABILITY_EDIT:
        # Edit prompts have their own exemplar structure; no format selection.
        fmt = EDIT_RESULT_META
        fmt_id = None
        st.sidebar.caption("Image-edit prompts (remove, restore, extend) with their own dedicated structure.")
    else:
        fmt_id = label_to_id(st.sidebar.radio("Prompt format", [f["label"] for f in PROMPT_FORMATS.values()], key="pg_format"))
        fmt = get_format(fmt_id)
        st.sidebar.caption(
            "One flowing prose paragraph ending with negative restrictions."
            if fmt_id == "format_1"
            else "Labeled sections (Direction, Mood, Camera, …)."
            if fmt_id == "format_2"
            else "Structured JSON object with nested sections."
        )

    # --- Main: source input --------------------------------------------------
    if capability == CAPABILITY_UPGRADE:
        raw_prompt = st.text_area(
            "Raw prompt",
            key="pg_raw_prompt",
            placeholder="A quiet harbor at dawn, fishing boats resting on glassy water…",
            height=120,
            help="Rough is fine — the model expands it into the selected format.",
        )
        image_file = None
        generate_label = "✨ Upgrade Prompt"
    elif capability == CAPABILITY_EDIT:
        raw_prompt = None
        image_file = st.file_uploader(
            "Image to edit",
            type=["png", "jpg", "jpeg", "webp"],
            key="pg_image",
            help="Upload the image you want to edit (PNG, JPEG, or WEBP, <50MB). It is treated as the exact source to preserve.",
        )
        if image_file:
            st.image(image_file, caption=image_file.name, width="stretch")
        st.text_area(
            "Edit request",
            key="pg_edit_request",
            placeholder=(
                "e.g. remove the text overlay and QR code, keep everything else identical — "
                "or: extend the canvas downward, keep the new city to the bottom 15% of the image"
            ),
            height=100,
            help="Rough is fine — the model turns it into a structured edit prompt that preserves the rest of the image.",
        )
        generate_label = "🪄 Generate Edit Prompt"
    else:
        raw_prompt = None
        image_file = st.file_uploader(
            "Image",
            type=["png", "jpg", "jpeg", "webp"],
            key="pg_image",
            help="Upload one image (PNG, JPEG, or WEBP, <50MB). The model reverse-engineers a prompt from it.",
        )
        if image_file:
            st.image(image_file, caption=image_file.name, width="stretch")
        extra_direction = st.text_input(
            "Extra direction (optional)",
            key="pg_extra_direction",
            placeholder="e.g. focus on the lighting",
        )
        generate_label = "🧩 Generate Prompt"

    if st.button(generate_label, type="primary", width="stretch"):
        _handle_generate(client, capability, fmt_id, raw_prompt, image_file)

    # --- Result (survives reruns) --------------------------------------------
    result = st.session_state.get("pg_result")
    if result and st.session_state.get("pg_result_capability") == capability:
        _render_result(result, fmt)


def _handle_generate(client, capability: str, fmt_id: str | None, raw_prompt, image_file) -> None:
    if capability == CAPABILITY_UPGRADE and not (raw_prompt or "").strip():
        st.warning("Please enter a raw prompt describing your idea.")
        return
    if capability == CAPABILITY_IMAGE and image_file is None:
        st.warning("Please upload an image to reverse-engineer a prompt from.")
        return
    if capability == CAPABILITY_EDIT:
        if image_file is None:
            st.warning("Please upload the image you want to edit.")
            return
        if not (st.session_state.get("pg_edit_request") or "").strip():
            st.warning("Please describe what you want changed in the image.")
            return

    try:
        with st.spinner("Composing your prompt…"):
            if capability == CAPABILITY_UPGRADE:
                generated = generate_prompt(client, PIPELINE_TEXT, fmt_id, raw_prompt=raw_prompt.strip())
                source = f"Raw prompt: {raw_prompt.strip()[:80]}…"
            elif capability == CAPABILITY_EDIT:
                generated = generate_prompt(
                    client,
                    PIPELINE_EDIT,
                    image_bytes=image_file.getvalue(),
                    image_mime=image_file.type or "image/png",
                    edit_request=(st.session_state.get("pg_edit_request") or "").strip(),
                )
                source = f"Edit of image: {image_file.name}"
            else:
                generated = generate_prompt(
                    client,
                    PIPELINE_IMAGE,
                    fmt_id,
                    image_bytes=image_file.getvalue(),
                    image_mime=image_file.type or "image/png",
                    extra_direction=st.session_state.get("pg_extra_direction") or None,
                )
                source = f"Image: {image_file.name}"
        st.session_state["pg_result"] = generated
        st.session_state["pg_result_capability"] = capability
        st.session_state["pg_source"] = source
        # Image capabilities keep the uploaded image around so the handoff to
        # Image Generation can attach it as an edit/reference input.
        if capability in (CAPABILITY_IMAGE, CAPABILITY_EDIT):
            st.session_state["pg_result_image"] = {
                "bytes": image_file.getvalue(),
                "mime": image_file.type or "image/png",
                "name": image_file.name,
            }
        else:
            st.session_state.pop("pg_result_image", None)
    except Exception as exc:  # noqa: BLE001 — surface any API/SDK failure in the UI
        st.error(f"Prompt generation failed: {exc}")


def _render_result(result: str, fmt: dict[str, object]) -> None:
    st.divider()
    st.subheader("Result")
    st.caption(st.session_state.get("pg_source", ""))

    rendered_tab, raw_tab = st.tabs(["Rendered", "Raw (copy-ready)"])
    with rendered_tab:
        if fmt["is_json"]:
            parsed = try_parse_json(result)
            if parsed is not None:
                st.json(parsed)
            else:
                st.error("The model returned text that is not valid JSON — showing the raw output instead.")
                st.code(result, language="text")
        else:
            st.markdown(result)
    with raw_tab:
        st.code(result, language="json" if fmt["is_json"] else "markdown")

    st.download_button(
        "⬇️ Download prompt",
        data=result,
        file_name=result_filename(fmt),
        mime="application/json" if fmt["is_json"] else "text/markdown",
        width="stretch",
    )

    if st.button("💡 Use in Image Generation", width="stretch"):
        st.session_state["pending_mode"] = "Image Generation"
        st.session_state["image_gen_prompt"] = result
        # Carry the uploaded image along so Image Generation already has it
        # attached as an edit/reference input (the uploader can't be pre-filled).
        handoff_image = st.session_state.get("pg_result_image")
        if handoff_image:
            st.session_state["pending_handoff_image"] = handoff_image
        st.rerun()
