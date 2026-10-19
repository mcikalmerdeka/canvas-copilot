"""Registry of the image models available on the Evolink provider.

One frozen `EvolinkModel` per model, transcribed from each model's official
OpenAPI spec (evolink.ai/docs). The Image Generation mode renders its option
menus and validates its request from this spec, so the UI automatically adapts
when a model is added or its schema changes:

- every model shares `POST /v1/images/generations` + task polling, but
- `quality` means resolution tiers (1K/2K/4K) on Nano Banana / Qwen and token
  tiers (low…max) on GPT Image 2.5, Grok adds a separate `resolution` param,
  Midjourney takes its aspect ratio inside the prompt (`--ar`) and its
  reference images as URLs prepended to the prompt text, and `n` is
  unsupported on Nano Banana and Midjourney.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

Option = tuple[str, str]  # (value, label) pairs for select boxes


@dataclass(frozen=True)
class EvolinkModel:
    """What one Evolink image model accepts (display labels included)."""

    model_id: str
    label: str
    size_options: tuple[Option, ...] | None
    quality_options: tuple[Option, ...] | None
    quality_default: str = "medium"
    resolution_options: tuple[Option, ...] | None = None
    resolution_default: str = "1K"
    speed_options: tuple[Option, ...] | None = None
    speed_default: str = "fast"
    max_variations: int = 1
    max_reference_images: int = 0
    supports_output_format: bool = False
    references_in_prompt: bool = False
    aspect_via_prompt: bool = False
    supports_negative_prompt: bool = False


def _pixel_options() -> tuple[Option, ...]:
    """The app's pixel-format sizes, shared by GPT Image 2.5 and Qwen."""
    return (
        ("1024x1024", "1024 × 1024 · Square"),
        ("1024x1536", "1024 × 1536 · Portrait"),
        ("1536x1024", "1536 × 1024 · Landscape"),
        ("2048x1152", "2048 × 1152 · Widescreen (16:9)"),
        ("1152x2048", "1152 × 2048 · Phone (9:16)"),
    )


def _ratio_options(ratios: Iterable[str]) -> tuple[Option, ...]:
    """`auto` + a ratio list, labeled with the ratio itself."""
    return tuple([("auto", "Auto"), *[(ratio, ratio) for ratio in ratios]])


ASPECT_RATIOS_MJ: tuple[str, ...] = (
    "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9",
)

_GPT_SIZES: tuple[Option, ...] = _ratio_options(
    ("1:1", "2:1", "1:2", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9", "9:21")
) + _pixel_options()

_NANO_SIZES: tuple[Option, ...] = _ratio_options(
    ("1:1", "1:4", "4:1", "1:8", "8:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9")
)

_GROK_SIZES: tuple[Option, ...] = _ratio_options(
    ("1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16", "2:1", "1:2", "19.5:9", "9:19.5", "20:9", "9:20")
)

_QWEN_SIZES: tuple[Option, ...] = _ratio_options(
    ("1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9", "9:21")
) + _pixel_options()

_GPT_QUALITY: tuple[Option, ...] = (
    ("low", "Low"),
    ("medium", "Medium"),
    ("high", "High"),
    ("xhigh", "X-High"),
    ("max", "Max"),
)
_GROK_QUALITY: tuple[Option, ...] = (("low", "Low"), ("medium", "Medium"))
_RESOLUTION_TIERS: tuple[Option, ...] = (("1K", "1K (~1 MP)"), ("2K", "2K (~4 MP)"))
_NANO_QUALITY: tuple[Option, ...] = (
    ("1K", "1K (~1 MP)"), ("2K", "2K (~4 MP)"), ("4K", "4K (~8.3 MP)"),
)
_QWEN_QUALITY: tuple[Option, ...] = (("1K", "1K (~1 MP)"), ("2K", "2K (~4 MP)"))
_MJ_QUALITY: tuple[Option, ...] = (("standard", "Standard"), ("hd", "HD (native 2K)"))
_MJ_SPEED: tuple[Option, ...] = (("fast", "Fast (4 images)"), ("draft", "Draft (24 sketches)"))


EVOLINK_MODELS: dict[str, EvolinkModel] = {
    "gpt-image-2.5-sunburst": EvolinkModel(
        model_id="gpt-image-2.5-sunburst",
        label="Sunburst (best editing precision)",
        size_options=_GPT_SIZES,
        quality_options=_GPT_QUALITY,
        max_variations=10,
        max_reference_images=16,
        supports_output_format=True,
    ),
    "gpt-image-2.5-flare": EvolinkModel(
        model_id="gpt-image-2.5-flare",
        label="Flare (faster, everyday quality)",
        size_options=_GPT_SIZES,
        quality_options=_GPT_QUALITY,
        max_variations=10,
        max_reference_images=16,
        supports_output_format=True,
    ),
    "gemini-nano-banana-2.1": EvolinkModel(
        model_id="gemini-nano-banana-2.1",
        label="Nano Banana 2.1",
        size_options=_NANO_SIZES,
        quality_options=_NANO_QUALITY,
        quality_default="2K",
        max_reference_images=14,
    ),
    "grok-imagine-image-2.0": EvolinkModel(
        model_id="grok-imagine-image-2.0",
        label="Grok Imagine 2.0",
        size_options=_GROK_SIZES,
        quality_options=_GROK_QUALITY,
        resolution_options=_RESOLUTION_TIERS,
        max_variations=10,
        max_reference_images=3,
    ),
    "mj-v8.2": EvolinkModel(
        model_id="mj-v8.2",
        label="Midjourney v8.2",
        size_options=None,
        quality_options=_MJ_QUALITY,
        quality_default="standard",
        speed_options=_MJ_SPEED,
        max_reference_images=3,
        references_in_prompt=True,
        aspect_via_prompt=True,
    ),
    "qwen-image-3.0": EvolinkModel(
        model_id="qwen-image-3.0",
        label="Qwen Image 3.0",
        size_options=_QWEN_SIZES,
        quality_options=_QWEN_QUALITY,
        quality_default="1K",
        max_variations=6,
        max_reference_images=3,
        supports_negative_prompt=True,
    ),
    "qwen-image-3.0-pro": EvolinkModel(
        model_id="qwen-image-3.0-pro",
        label="Qwen Image 3.0 Pro",
        size_options=_QWEN_SIZES,
        quality_options=_QWEN_QUALITY,
        quality_default="1K",
        max_variations=6,
        max_reference_images=3,
        supports_negative_prompt=True,
    ),
}


def build_model_prompt(spec: EvolinkModel, prompt: str, reference_urls: Iterable[str] = ()) -> str:
    """The prompt actually sent to the model.

    Midjourney takes reference images inside the prompt text (URLs at the
    start); every other model receives the prompt verbatim.
    """
    prompt = prompt.strip()
    if spec.references_in_prompt:
        urls = [url.strip() for url in reference_urls if url.strip()]
        return " ".join([*urls, prompt]) if urls else prompt
    return prompt


def aspect_prompt_flag(size: str | None) -> str:
    """`--ar` flag text for prompt-driven aspect ratios; empty for auto/None."""
    if not size or size == "auto":
        return ""
    return f"--ar {size}"
