# 🎨 Canvas Copilot

A Streamlit app with two modes, selectable from the left sidebar:

- **Image Generation** — turn prompts into images with OpenAI's GPT Image models, with reference images, aspect ratios, and quality controls.
- **Prompt Generation** — upgrade rough prompt ideas into richly structured prompts, or reverse-engineer a prompt from an uploaded photo, using OpenAI's `gpt-6-luna` reasoning model.

## Features

### Image Generation mode

- Text prompt → image with `gpt-image-2.5-sunburst` (best editing precision) or `gpt-image-2.5-flare` (faster, everyday quality).
- Optional **reference images** (up to 16): attach them and the app switches from the generations endpoint to the edits endpoint, using them as style/subject/composition references.
- Controls for **size** (square, portrait, landscape, 16:9, 9:16), **quality** (auto → max), **moderation** strictness, **variations** (1–10 per request), **output format** (PNG/JPEG/WEBP), and **compression** (JPEG/WEBP only).
- Multiple variations render in a tiled grid, each with its own download button.
- Content-moderation blocks show a targeted hint plus expandable moderation details (stage and categories), per the OpenAI API's `moderation_details` payload.

### Prompt Generation mode

Powered by `gpt-6-luna` with **medium reasoning effort**, called through the OpenAI **Responses API** (text and image input, ~1M context window). The selected format's exemplar file is injected directly into the model context as the template to emulate.

Two capabilities, selectable in the sidebar:

1. **Upgrade Prompt** — paste a rough prompt; the model expands it into the selected format.
2. **Image-to-Prompt** — upload one image (PNG/JPEG/WEBP); the model studies it and writes the prompt that would recreate a photo like it. An optional "extra direction" field steers attention (e.g. *focus on the lighting*).

Three target prompt formats (one at a time), each backed by a real exemplar in `prompts/`:

| Format | Style | Exemplar |
|---|---|---|
| Format 1 | Narrative prose ending with negative restrictions | `prompts/format_1.md` |
| Format 2 | Labeled sections (`Direction:`, `Mood:`, `Camera:`, …) | `prompts/format_2.md` |
| Format 3 | Structured JSON object | `prompts/format_3.json` |

Results render in **Rendered / Raw (copy-ready)** tabs — format 3 renders as a JSON tree, with a fallback to raw output if the model ever returns invalid JSON — plus a download button (`.md` or `.json` per format) and a **💡 Use in Image Generation** button that switches modes and pre-fills the Image Generation prompt box. Inputs and the last result survive Streamlit reruns.

## Getting started

Requirements: **Python 3.14+** and [uv](https://docs.astral.sh/uv/).

1. Install dependencies:

   ```pwsh
   uv sync
   ```

2. Create a `.env` file in the project root with your OpenAI API key:

   ```dotenv
   OPENAI_API_KEY=sk-...
   ```

3. Run the app:

   ```pwsh
   uv run streamlit run app.py
   ```

## Testing

The test suite covers the format registry, LLM message building, output handling (code-fence stripping, JSON parsing), and bare-mode rendering of both modes:

```pwsh
uv run pytest                # full suite (29 offline tests)
uv run pytest -m integration # live test: one real gpt-6-luna call (needs OPENAI_API_KEY)
uv run pytest -m "not integration"  # offline only (default behavior: live test auto-skips without a key)
```

## Project layout

```
app.py                  Entry point: sidebar mode selector + OpenAI client + dispatch
image_generation.py     Image Generation mode (GPT Image models)
prompt_generation.py    Prompt Generation mode (gpt-6-luna via the Responses API)
prompt_formats.py       Registry of the three prompt formats / exemplar loading
prompts/                Exemplar files used as the injected prompt templates
tests/                  Pytest suite (includes one live-API integration test)
PROMPTS.md              Prompt engineering notes + library for the image models
```

## Notes

- **Aspect ratios are set in the UI, not the prompt**, for image generation — describe composition instead of writing "16:9" into the text (see `PROMPTS.md` for a full guide to prompting the image models).
- Prompt Generation always depicts clearly adult subjects; the system prompt enforces this in both capabilities.
- Adding a fourth prompt format is one registry entry in `prompt_formats.py` plus its exemplar file in `prompts/`.
