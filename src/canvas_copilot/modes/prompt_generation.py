"""Prompt Generation mode — build, upgrade, or reverse-engineer image prompts.

Pipeline: the chosen format's exemplar (see `prompt_formats.py`) is injected
directly into the `gpt-6-luna` context together with the source material (a
structured builder brief, a raw text prompt, or an uploaded image), and the
model returns the finished prompt in exactly that format.
"""

from __future__ import annotations

import json
import logging
import time

from canvas_copilot.core import evolink_api as ev
from canvas_copilot.modes.prompt_formats import (
    EDIT_EXEMPLAR,
    PROMPT_FORMATS,
    get_format,
    label_to_id,
)
from canvas_copilot.settings import providers as pr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

MODEL = "gpt-6-luna"
REASONING_EFFORT = "medium"
MAX_OUTPUT_TOKENS = 20_000

CAPABILITY_UPGRADE = "Upgrade Prompt"
CAPABILITY_IMAGE = "Image-to-Prompt"
CAPABILITY_EDIT = "Image-Edit Prompt"
CAPABILITY_BUILDER = "Prompt Builder"

# Internal pipeline capability codes (what generate_prompt/build_user_content use).
PIPELINE_TEXT = "text"
PIPELINE_IMAGE = "image"
PIPELINE_EDIT = "edit"
PIPELINE_GUIDED = "guided"

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

GUIDED_REQUIREMENTS = """The user is supplying a structured creative brief. Treat every supplied facet as a hard requirement:
- If the brief contains a STARTING PROMPT, treat it as the user's authoritative draft: preserve its subject, intent, and specific details, then enrich it with the supplied facets. Do not replace or contradict it.
- Reflect every supplied facet explicitly in the finished prompt. Do not drop, merge away, or contradict any selection.
- Expand only in ways consistent with the brief; do not invent a different subject, setting, wardrobe, camera, or mood.
- When a facet is absent, choose details that fit the rest of the brief rather than inventing a conflicting direction.
- Keep every depicted subject clearly adult.
"""

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

GUIDED_USER_TEMPLATE = """Here is the structured creative brief to turn into a finished image-generation prompt:

{brief}

Follow the exemplar format above. Output only the finished prompt."""

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


def build_guided_system_prompt(format_id: str) -> str:
    """Format exemplar plus the builder's non-negotiable facet requirements."""
    return f"{build_system_prompt(format_id)}\n\n{GUIDED_REQUIREMENTS}"


def build_image_data_url(image_bytes: bytes, image_mime: str) -> str:
    """Encode uploaded image bytes as a data URL for the Responses API."""
    return ev.to_data_url(image_bytes, image_mime)


def resolve_image_ref(ctx: pr.ProviderContext, image_bytes: bytes, image_mime: str, file_name: str) -> str:
    """Return the `image_url` string the Responses `input_image` block needs.

    OpenAI accepts an inline base64 data URL; Evolink requires a downloadable
    public URL, so the bytes go through its file service first.
    """
    if ctx.is_evolink:
        return ev.upload_image(ctx.api_key, image_bytes, image_mime, file_name)
    return build_image_data_url(image_bytes, image_mime)


def build_user_content(
    capability: str,
    raw_prompt: str | None = None,
    image_bytes: bytes | None = None,
    image_mime: str | None = None,
    extra_direction: str | None = None,
    edit_request: str | None = None,
    image_url: str | None = None,
    guided_brief: str | None = None,
) -> str | list[dict]:
    """Build the user message: plain text for upgrades, text + image for image-to-prompt and image-edit.

    `image_url`, when given, is used verbatim (an already-uploaded Evolink file
    URL or any other downloadable reference); `image_bytes` is otherwise
    encoded as an inline base64 data URL.
    """
    if capability == PIPELINE_GUIDED:
        return GUIDED_USER_TEMPLATE.format(brief=(guided_brief or "").strip())
    if capability == PIPELINE_IMAGE:
        text = IMAGE_USER_TEXT
        if extra_direction:
            text += f"\n\nExtra direction from the user: {extra_direction.strip()}"
        return [
            {"type": "input_text", "text": text},
            {"type": "input_image", "image_url": image_url or build_image_data_url(image_bytes or b"", image_mime or "image/png")},
        ]
    if capability == PIPELINE_EDIT:
        return [
            {"type": "input_text", "text": EDIT_USER_TEMPLATE.format(edit_request=(edit_request or "").strip())},
            {"type": "input_image", "image_url": image_url or build_image_data_url(image_bytes or b"", image_mime or "image/png")},
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
    image_url: str | None = None,
    guided_brief: str | None = None,
) -> str:
    """One blocking call to `gpt-6-luna`; returns the finished prompt text."""
    if capability == PIPELINE_GUIDED and not (guided_brief or "").strip():
        raise ValueError("A guided brief is required before generating a prompt.")
    system_prompt = (
        build_guided_system_prompt(format_id)
        if capability == PIPELINE_GUIDED
        else build_edit_system_prompt()
        if capability == PIPELINE_EDIT
        else build_system_prompt(format_id)
    )
    logger.info(
        "Prompt generation started (capability=%s, format=%s)",
        capability,
        format_id or "-",
    )
    start_time = time.perf_counter()
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
                    image_url=image_url,
                    guided_brief=guided_brief,
                ),
            },
        ],
    )
    finished = strip_code_fences(extract_output_text(response))
    logger.info(
        "Prompt finished: %d characters in %.1fs",
        len(finished),
        time.perf_counter() - start_time,
    )
    return finished


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def render(ctx: pr.ProviderContext) -> None:
    """Render the Prompt Generation workspace (UI lives in prompt_views)."""
    from canvas_copilot.modes import prompt_views

    prompt_views.render_mode(ctx)
