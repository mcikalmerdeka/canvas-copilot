"""Registry of the prompt formats used by the Prompt Generation mode.

Each entry describes one target format the `gpt-6-luna` model can be asked to
emulate when upgrading raw prompts or reverse-engineering prompts from images.
The exemplar for each format is loaded once at import time so the files in
`prompts/` are the single source of truth for what the model receives. The
Image-Edit Prompt capability loads its own exemplar the same way.
"""

from __future__ import annotations

from canvas_copilot import PACKAGE_ROOT

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PROMPT_FORMATS: dict[str, dict[str, object]] = {
    "format_1": {
        "id": "format_1",
        "label": "Format 1 · Narrative prose",
        "exemplar_path": "prompts/format_1.md",
        "is_json": False,
    },
    "format_2": {
        "id": "format_2",
        "label": "Format 2 · Labeled sections",
        "exemplar_path": "prompts/format_2.md",
        "is_json": False,
    },
    "format_3": {
        "id": "format_3",
        "label": "Format 3 · Structured JSON",
        "exemplar_path": "prompts/format_3.json",
        "is_json": True,
    },
}

# Exemplar text is joined into the registry after the definitions above so the
# entries stay declarative (one reason: keeps the mapping readable at a glance).
# Paths resolve from the package root, where `prompts/` lives in the src layout.
for _fmt in PROMPT_FORMATS.values():
    _fmt["exemplar_text"] = (PACKAGE_ROOT / _fmt["exemplar_path"]).read_text(encoding="utf-8")
del _fmt


# The Image-Edit Prompt capability has its own exemplar (not one of the three
# generation formats above): a pair of finished edit prompts showing the removal and
# extension skeletons the model should emulate.
EDIT_EXEMPLAR_PATH = "prompts/edit_prompt.md"
EDIT_EXEMPLAR = (PACKAGE_ROOT / EDIT_EXEMPLAR_PATH).read_text(encoding="utf-8")


def get_format(format_id: str) -> dict[str, object]:
    """Return the registry entry for `format_id` (raises KeyError if unknown)."""
    return PROMPT_FORMATS[format_id]


def list_labels() -> list[str]:
    """Return the human-readable labels in registry order, for UI select boxes."""
    return [fmt["label"] for fmt in PROMPT_FORMATS.values()]


def label_to_id(label: str) -> str:
    """Inverse of `list_labels()`: map a UI label back to its registry key."""
    for fmt_id, fmt in PROMPT_FORMATS.items():
        if fmt["label"] == label:
            return fmt_id
    raise KeyError(f"Unknown prompt format label: {label!r}")
