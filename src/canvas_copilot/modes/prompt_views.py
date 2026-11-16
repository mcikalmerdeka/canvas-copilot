"""Prompt Generation UI: guided builder plus the three image/text workflows.

This module owns layout and widget state only. Prompt composition and API calls
live in `prompt_generation.py`; the option vocabulary lives in `prompt_catalog.py`.
Keeping the view separate keeps both modules small enough to reason about.
"""

from __future__ import annotations

import logging

import streamlit as st

from canvas_copilot import ui
from canvas_copilot.modes import prompt_catalog as catalog
from canvas_copilot.modes import prompt_generation as pg
from canvas_copilot.modes.prompt_formats import (
    PROMPT_FORMATS,
    get_format,
    label_to_id,
)
from canvas_copilot.settings import providers as pr

logger = logging.getLogger(__name__)

CAPABILITIES = (
    pg.CAPABILITY_BUILDER,
    pg.CAPABILITY_UPGRADE,
    pg.CAPABILITY_IMAGE,
    pg.CAPABILITY_EDIT,
)

FORMAT_LABELS = [fmt["label"] for fmt in PROMPT_FORMATS.values()]
SHORT_FORMAT = {
    "Format 1 · Narrative prose": "Narrative",
    "Format 2 · Labeled sections": "Labeled sections",
    "Format 3 · Structured JSON": "JSON",
}

FORMAT_CAPTION = {
    pg.CAPABILITY_BUILDER: "The brief is rewritten into this format.",
    pg.CAPABILITY_UPGRADE: "Your rough idea is rewritten into this format.",
    pg.CAPABILITY_IMAGE: "The image is reverse-engineered into this format.",
}

PERSON_PLACEHOLDER = (
    "A woman in her early 30s with a calm, self-assured presence and a direct gaze."
)
PROPS_PLACEHOLDER = "A ceramic cup, a folded newspaper, a brass ruler…"
DETAILS_PLACEHOLDER = "Glasses, freckles, tattoos, makeup, jewellery, distinguishing features…"
NOTES_PLACEHOLDER = "Anything else: specific props, on-image text, a reference vibe…"
NEGATIVE_PLACEHOLDER = "Any other element that must not appear…"


# ---------------------------------------------------------------------------
# Builder state helpers
# ---------------------------------------------------------------------------

def _facet_key(facet_id: str) -> str:
    return f"pb_{facet_id}"


def _default_value(facet: catalog.Facet) -> str | list[str]:
    if facet.kind == "multiselect":
        return []
    return ""


def _widget_default(facet: catalog.Facet) -> str | list[str]:
    if facet.kind == "text":
        return ""
    if facet.kind == "multiselect":
        return []
    return catalog.DEFAULT_LABEL


def _reset_facet_state() -> None:
    for facet in catalog.FACETS:
        st.session_state[_facet_key(facet.id)] = _widget_default(facet)


def _apply_preset() -> None:
    """Fill every builder widget from the selected preset, then rerun."""
    preset_id = st.session_state.get("pb_preset", "fresh")
    _reset_facet_state()
    preset = catalog.PRESETS.get(preset_id, catalog.PRESETS["fresh"])
    subject_type = str(preset.get("subject_type", "Person"))
    st.session_state["pb_subject_type"] = subject_type
    st.session_state["pb_subject"] = str(preset.get("subject", ""))
    for facet_id, value in catalog.preset_values(preset_id).items():
        st.session_state[_facet_key(facet_id)] = value


def _randomize() -> None:
    """Fill the visible facets with valid random options, leaving prose alone."""
    subject_type = st.session_state.get("pb_subject_type", "Person")
    values = catalog.randomize_selections(subject_type)
    for facet in catalog.facets_for(subject_type):
        if facet.kind == "text":
            continue
        value = values.get(facet.id)
        if value is not None:
            st.session_state[_facet_key(facet.id)] = value
    st.session_state["pb_preset"] = "fresh"


def _reset_builder() -> None:
    _reset_facet_state()
    st.session_state["pb_subject"] = ""
    st.session_state["pb_subject_type"] = "Person"
    st.session_state["pb_preset"] = "fresh"


def _selected_count(facets: tuple[catalog.Facet, ...]) -> int:
    count = 0
    for facet in facets:
        value = st.session_state.get(_facet_key(facet.id), _widget_default(facet))
        if isinstance(value, list):
            count += len(value)
        elif value and value != catalog.DEFAULT_LABEL:
            count += 1
    return count


def _current_selections(subject_type: str) -> dict[str, str | list[str]]:
    return {
        facet.id: st.session_state.get(_facet_key(facet.id), _widget_default(facet))
        for facet in catalog.facets_for(subject_type)
    }


# ---------------------------------------------------------------------------
# Builder rendering
# ---------------------------------------------------------------------------

def _render_facet(facet: catalog.Facet) -> None:
    key = _facet_key(facet.id)
    if facet.kind == "text":
        placeholder = {
            "character_details": DETAILS_PLACEHOLDER,
            "props_details": PROPS_PLACEHOLDER,
            "extra_notes": NOTES_PLACEHOLDER,
            "custom_negatives": NEGATIVE_PLACEHOLDER,
        }.get(facet.id, "")
        st.text_area(
            facet.label,
            key=key,
            height=78,
            placeholder=placeholder,
            help=facet.help,
        )
        return
    if facet.kind == "multiselect":
        st.multiselect(
            facet.label,
            [option.label for option in facet.options],
            key=key,
            help=facet.help,
        )
        return
    st.selectbox(
        facet.label,
        [catalog.DEFAULT_LABEL, *[option.label for option in facet.options]],
        key=key,
        help=facet.help,
    )


GROUP_HELP = {
    "character": "Adult subjects only. Pick only what matters; the rest is left to the model.",
    "scene": "Where it happens and how it is lit.",
    "camera": "Framing, lens, and composition.",
    "style": "Visual language, palette, and mood.",
    "finishing": "Extra direction, plus common failure modes to avoid.",
}


def _render_group(group_id: str, label: str, subject_type: str) -> None:
    facets = tuple(
        facet for facet in catalog.facets_for(subject_type) if facet.group == group_id
    )
    if not facets:
        return
    count = _selected_count(facets)
    expander_label = f"{label} · {count} selected" if count else label
    with st.expander(expander_label, expanded=False):
        if GROUP_HELP.get(group_id):
            st.caption(GROUP_HELP[group_id])
        columns = st.columns(2, gap="medium")
        selects = [facet for facet in facets if facet.kind != "text"]
        for index, facet in enumerate(selects):
            with columns[index % 2]:
                _render_facet(facet)
        for facet in facets:
            if facet.kind == "text":
                _render_facet(facet)


def _render_builder(fmt_id: str) -> tuple[str, str, str, bool]:
    """Render the builder; return (subject, subject_type, brief, submitted)."""
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        preset_columns = st.columns([1.7, 1, 1])
        with preset_columns[0]:
            st.selectbox(
                "Start from a preset",
                list(catalog.PRESETS),
                format_func=lambda preset_id: str(catalog.PRESETS[preset_id]["label"]),
                key="pb_preset",
                on_change=_apply_preset,
                help="Presets fill the facets with a working starting point. Everything stays editable.",
            )
        with preset_columns[1]:
            st.button(
                "Surprise me",
                icon=":material/casino:",
                on_click=_randomize,
                width="stretch",
            )
        with preset_columns[2]:
            st.button(
                "Reset",
                icon=":material/restart_alt:",
                on_click=_reset_builder,
                width="stretch",
            )

        with st.container(border=True):
            ui.section("Subject", "Who or what is the image about?")
            st.session_state.setdefault("pb_subject_type", "Person")
            subject_type = st.segmented_control(
                "Subject type",
                catalog.SUBJECT_TYPES,
                key="pb_subject_type",
                required=True,
            )
            subject_type = subject_type or "Person"
            subject = st.text_area(
                "Subject",
                key="pb_subject",
                height=96,
                placeholder=(
                    PERSON_PLACEHOLDER
                    if subject_type in ("Person", "Group")
                    else "A lone stone chapel on a ridge above a sea of low clouds…"
                ),
                help="A sentence or two. Concrete nouns beat adjectives.",
            )

        for group_id, label in catalog.GROUPS:
            if group_id == "subject":
                continue
            _render_group(group_id, label, subject_type)

    selections = _current_selections(subject_type)
    brief = catalog.build_brief(subject, subject_type, selections)
    preview_text = (
        brief
        if subject.strip()
        else "# Choose a subject and a few facets to see the brief."
    )

    submitted = False
    with right:
        with st.container(border=True):
            ui.section(
                "Brief preview",
                "This structured brief is what the model reads. The finished prompt below follows the exemplar format.",
            )
            st.code(
                preview_text,
                language="text",
                wrap_lines=True,
                height=380,
            )
            st.caption(f"{len(brief):,} characters · {_selected_count(catalog.facets_for(subject_type))} choices")
            submitted = st.button(
                "Generate Prompt",
                type="primary",
                icon=":material/auto_awesome:",
                width="stretch",
            )

    return subject, subject_type, brief, submitted


# ---------------------------------------------------------------------------
# Legacy capability forms
# ---------------------------------------------------------------------------

def _render_upgrade(fmt: dict[str, object]) -> bool:
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        with st.container(border=True):
            ui.section(
                "Your rough idea",
                "Rough is fine. The model expands it into the selected format.",
            )
            st.text_area(
                "Raw prompt",
                key="pg_raw_prompt",
                placeholder="A quiet harbor at dawn, fishing boats resting on glassy water…",
                height=220,
            )
    with right:
        with st.container(border=True):
            ui.section("Output", f"Format · {fmt['label']}")
            st.caption("The model keeps your intent and fills the gaps with concrete visual detail.")
            return st.button(
                "Upgrade Prompt",
                type="primary",
                icon=":material/auto_fix_high:",
                width="stretch",
            )
    return False


def _render_image_to_prompt(fmt: dict[str, object]) -> bool:
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        with st.container(border=True):
            ui.section(
                "Source image",
                "One image (PNG, JPEG, or WEBP, under 50MB). The model reverse-engineers a prompt from it.",
            )
            image_file = st.file_uploader(
                "Image",
                type=["png", "jpg", "jpeg", "webp"],
                key="pg_image",
                label_visibility="collapsed",
            )
            if image_file:
                st.image(image_file, caption=image_file.name, width="stretch")
    with right:
        with st.container(border=True):
            ui.section("Direction", f"Format · {fmt['label']}")
            st.text_input(
                "Extra direction (optional)",
                key="pg_extra_direction",
                placeholder="e.g. focus on the lighting",
                help="Steer the model toward what matters most about this image.",
            )
            return st.button(
                "Generate Prompt",
                type="primary",
                icon=":material/image_search:",
                width="stretch",
            )
    return False


def _render_edit() -> bool:
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        with st.container(border=True):
            ui.section(
                "Source image",
                "Upload the exact image you want to edit. Everything not named in the request is preserved.",
            )
            image_file = st.file_uploader(
                "Image to edit",
                type=["png", "jpg", "jpeg", "webp"],
                key="pg_image",
                label_visibility="collapsed",
            )
            if image_file:
                st.image(image_file, caption=image_file.name, width="stretch")
    with right:
        with st.container(border=True):
            ui.section("Edit request", "Image-edit prompts use their own dedicated structure.")
            st.text_area(
                "What should change?",
                key="pg_edit_request",
                height=160,
                placeholder=(
                    "e.g. remove the text overlay and QR code, keep everything else identical — "
                    "or: extend the canvas downward, keep the new city to the bottom 15%"
                ),
                help="Rough is fine. The model turns it into a structured edit prompt that preserves the rest.",
            )
            return st.button(
                "Generate Edit Prompt",
                type="primary",
                icon=":material/edit_note:",
                width="stretch",
            )
    return False


def _render_legacy(
    ctx: pr.ProviderContext,
    capability: str,
    fmt_id: str | None,
    fmt: dict[str, object],
) -> None:
    if capability == pg.CAPABILITY_UPGRADE:
        submitted = _render_upgrade(fmt)
    elif capability == pg.CAPABILITY_IMAGE:
        submitted = _render_image_to_prompt(fmt)
    else:
        submitted = _render_edit()
    if submitted:
        _handle_generate(ctx, capability, fmt_id)


# ---------------------------------------------------------------------------
# Generation and result
# ---------------------------------------------------------------------------

def _handle_generate(
    ctx: pr.ProviderContext,
    capability: str,
    fmt_id: str | None,
    *,
    guided_brief: str | None = None,
) -> None:
    if capability == pg.CAPABILITY_BUILDER:
        if not (guided_brief or "").strip():
            st.warning("Please describe the subject before generating a prompt.")
            return
    elif capability == pg.CAPABILITY_UPGRADE:
        raw_prompt = st.session_state.get("pg_raw_prompt") or ""
        if not raw_prompt.strip():
            st.warning("Please enter a raw prompt describing your idea.")
            return
    elif capability == pg.CAPABILITY_IMAGE:
        if st.session_state.get("pg_image") is None:
            st.warning("Please upload an image to reverse-engineer a prompt from.")
            return
    elif capability == pg.CAPABILITY_EDIT:
        if st.session_state.get("pg_image") is None:
            st.warning("Please upload the image you want to edit.")
            return
        if not (st.session_state.get("pg_edit_request") or "").strip():
            st.warning("Please describe what you want changed in the image.")
            return

    try:
        with st.spinner("Composing your prompt…"):
            image_file = st.session_state.get("pg_image")
            image_url = None
            if capability in (pg.CAPABILITY_IMAGE, pg.CAPABILITY_EDIT) and image_file is not None:
                image_url = pg.resolve_image_ref(
                    ctx,
                    image_file.getvalue(),
                    image_file.type or "image/png",
                    image_file.name,
                )
                if ctx.is_evolink:
                    logger.info("Source image uploaded to file service: %s", image_file.name)
                else:
                    logger.info("Source image embedded inline (data URL): %s", image_file.name)

            if capability == pg.CAPABILITY_BUILDER:
                generated = pg.generate_prompt(
                    ctx.client,
                    pg.PIPELINE_GUIDED,
                    fmt_id,
                    guided_brief=guided_brief,
                )
                source = f"Prompt Builder · {len(guided_brief or ''):,} character brief"
                st.session_state["pg_brief"] = guided_brief
            elif capability == pg.CAPABILITY_UPGRADE:
                raw_prompt = (st.session_state.get("pg_raw_prompt") or "").strip()
                generated = pg.generate_prompt(
                    ctx.client, pg.PIPELINE_TEXT, fmt_id, raw_prompt=raw_prompt
                )
                source = f"Raw prompt: {raw_prompt[:80]}…"
                st.session_state.pop("pg_brief", None)
            elif capability == pg.CAPABILITY_EDIT:
                generated = pg.generate_prompt(
                    ctx.client,
                    pg.PIPELINE_EDIT,
                    image_url=image_url,
                    edit_request=(st.session_state.get("pg_edit_request") or "").strip(),
                )
                source = f"Edit of image: {image_file.name}"
                st.session_state.pop("pg_brief", None)
            else:
                generated = pg.generate_prompt(
                    ctx.client,
                    pg.PIPELINE_IMAGE,
                    fmt_id,
                    image_url=image_url,
                    extra_direction=st.session_state.get("pg_extra_direction") or None,
                )
                source = f"Image: {image_file.name}"
                st.session_state.pop("pg_brief", None)

        st.session_state["pg_result"] = generated
        st.session_state["pg_result_capability"] = capability
        st.session_state["pg_result_format"] = fmt_id if fmt_id else "edit"
        st.session_state["pg_source"] = source
        st.toast("Prompt ready", icon=":material/check_circle:")
        if capability in (pg.CAPABILITY_IMAGE, pg.CAPABILITY_EDIT):
            st.session_state["pg_result_image"] = {
                "bytes": image_file.getvalue(),
                "mime": image_file.type or "image/png",
                "name": image_file.name,
            }
        else:
            st.session_state.pop("pg_result_image", None)
    except Exception as exc:  # noqa: BLE001 — surface any API/SDK failure in the UI
        logger.error("Prompt generation failed: %s", exc)
        st.error(f"Prompt generation failed: {exc}")


def _render_result(capability: str, fmt: dict[str, object]) -> None:
    result = st.session_state.get("pg_result")
    if not result or st.session_state.get("pg_result_capability") != capability:
        return

    # Render with the format the result was generated in, even if the user has
    # since switched the format control to something else.
    stored_format = st.session_state.get("pg_result_format")
    if stored_format == "edit":
        render_fmt = pg.EDIT_RESULT_META
    elif stored_format in PROMPT_FORMATS:
        render_fmt = get_format(stored_format)
    else:
        render_fmt = fmt

    st.divider()
    ui.section("Result", st.session_state.get("pg_source", ""))
    rendered_tab, raw_tab = st.tabs(["Rendered", "Raw (copy-ready)"])
    with rendered_tab:
        if render_fmt["is_json"]:
            parsed = pg.try_parse_json(result)
            if parsed is not None:
                st.json(parsed)
            else:
                st.error("The model returned text that is not valid JSON — showing the raw output instead.")
                st.code(result, language="text", wrap_lines=True)
        else:
            st.markdown(result)
    with raw_tab:
        st.code(
            result,
            language="json" if render_fmt["is_json"] else "markdown",
            wrap_lines=True,
        )

    if capability == pg.CAPABILITY_BUILDER and st.session_state.get("pg_brief"):
        with st.expander("Brief used"):
            st.code(st.session_state["pg_brief"], language="text", wrap_lines=True)

    download_column, handoff_column = st.columns(2)
    with download_column:
        st.download_button(
            "Download prompt",
            data=result,
            file_name=pg.result_filename(render_fmt),
            mime="application/json" if render_fmt["is_json"] else "text/markdown",
            icon=":material/download:",
            width="stretch",
        )
    with handoff_column:
        if st.button(
            "Use in Image Generation",
            icon=":material/arrow_forward:",
            width="stretch",
        ):
            st.session_state["pending_mode"] = "Image Generation"
            st.session_state["image_gen_prompt"] = result
            handoff_image = st.session_state.get("pg_result_image")
            if handoff_image:
                st.session_state["pending_handoff_image"] = handoff_image
            st.rerun()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def render_mode(ctx: pr.ProviderContext) -> None:
    """Render the Prompt Generation workspace for the active provider."""
    control_columns = st.columns([2.3, 1], gap="large")
    with control_columns[0]:
        if "pg_capability" not in st.session_state:
            st.session_state["pg_capability"] = pg.CAPABILITY_BUILDER
        capability = st.segmented_control(
            "Capability",
            CAPABILITIES,
            key="pg_capability",
            required=True,
            wrap=True,
        )
        capability = capability or pg.CAPABILITY_BUILDER
    with control_columns[1]:
        if capability == pg.CAPABILITY_EDIT:
            st.caption("Image-edit prompts use their own dedicated structure.")
            fmt_id = None
            fmt = pg.EDIT_RESULT_META
        else:
            format_label = st.selectbox(
                "Output format",
                FORMAT_LABELS,
                format_func=SHORT_FORMAT.get,
                key="pg_format",
            )
            fmt_id = label_to_id(format_label)
            fmt = get_format(fmt_id)
            st.caption(FORMAT_CAPTION.get(capability, ""))

    if capability == pg.CAPABILITY_BUILDER:
        subject, subject_type, brief, submitted = _render_builder(fmt_id)
        if submitted:
            if not subject.strip():
                st.warning("Please describe the subject before generating a prompt.")
            else:
                logger.info(
                    "Guided prompt generation started (subject_type=%s, facets=%d)",
                    subject_type,
                    _selected_count(catalog.facets_for(subject_type)),
                )
                _handle_generate(
                    ctx,
                    capability,
                    fmt_id,
                    guided_brief=brief,
                )
    else:
        _render_legacy(ctx, capability, fmt_id, fmt)

    _render_result(capability, fmt)
