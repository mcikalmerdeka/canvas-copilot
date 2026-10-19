# 🎨 Canvas Copilot

A Streamlit app with two modes and two providers, selectable from the left sidebar:

- **Provider** — *OpenAI* (official API, default) or *Evolink* (EvoLink.AI aggregator). Both modes run on whichever provider is selected.
- **Image Generation** — turn prompts into images with OpenAI's GPT Image models, with reference images, aspect ratios, and quality controls.
- **Prompt Generation** — upgrade rough prompt ideas into richly structured prompts, reverse-engineer a prompt from an uploaded photo, or turn an edit request into a structured image-editing prompt, using OpenAI's `gpt-6-luna` reasoning model.

## Providers

| | OpenAI (official) | Evolink (aggregator) |
|---|---|---|
| API key (in `.env`) | `OPENAI_API_KEY` | `EVOLINK_API_KEY` |
| Prompt Generation | OpenAI Responses API at `api.openai.com` | OpenAI-compatible Responses API at `direct.evolink.ai` — same models (`gpt-6-luna`), same parameters |
| Image Generation | Sync `/images` endpoints; reference images uploaded as multipart files; base64 responses | Async task API (`/v1/images/generations` → poll `/v1/tasks/{id}` with a live progress bar); reference images are uploaded to Evolink's file service and passed as URLs; finished images are downloaded from task result URLs (valid 24 h) |
| Quality tiers | auto, low … max | low … max (no `auto`; default `medium`) |
| Moderation control | yes | not offered by Evolink (control hidden) |
| Output compression (JPEG/WEBP) | yes | not in Evolink's schema (control hidden) |

Switching providers only requires the matching key in `.env`; the app stops with a clear message when the selected provider's key is missing. Prompt Generation's image capabilities under Evolink first upload the picture through `files-api.evolink.ai` (files expire after 72 h) because the Responses API there needs a downloadable URL rather than inline base64.

## Features

### Image Generation mode

- Text prompt → image with `gpt-image-2.5-sunburst` (best editing precision) or `gpt-image-2.5-flare` (faster, everyday quality).
- Optional **reference images** (up to 16): attach them and the app switches from the generations endpoint to the edits endpoint, using them as style/subject/composition references.
- Controls for **size** (square, portrait, landscape, 16:9, 9:16), **quality** (auto → max), **moderation** strictness, **variations** (1–10 per request), **output format** (PNG/JPEG/WEBP), and **compression** (JPEG/WEBP only).
- Multiple variations render in a tiled grid, each with its own download button.
- Content-moderation blocks show a targeted hint plus expandable moderation details (stage and categories), per the OpenAI API's `moderation_details` payload.

### Prompt Generation mode

Powered by `gpt-6-luna` with **medium reasoning effort**, called through the OpenAI **Responses API** (text and image input, ~1M context window). The selected format's exemplar file is injected directly into the model context as the template to emulate.

Three capabilities, selectable in the sidebar:

1. **Upgrade Prompt** — paste a rough prompt; the model expands it into the selected format.
2. **Image-to-Prompt** — upload one image (PNG/JPEG/WEBP); the model studies it and writes the prompt that would recreate a photo like it. An optional "extra direction" field steers attention (e.g. *focus on the lighting*).
3. **Image-Edit Prompt** — upload the image you want to edit plus a rough edit request (*"remove the text and QR code, keep everything else identical"*, *"extend downward, keep the new city to the bottom 15%"*); the model turns it into a richly structured **image-editing prompt** (task framing, element-by-element changes, restoration/construction guidance, strict preservation requirements, negative constraints, output spec). This capability uses its own dedicated exemplars in `prompts/edit_prompt.md` instead of the three generation formats, and the model studies the uploaded image so the prompt references its real content (named subjects, palette, composition).

Three target prompt formats (one at a time), each backed by a real exemplar in `prompts/`:

| Format | Style | Exemplar |
|---|---|---|
| Format 1 | Narrative prose ending with negative restrictions | `prompts/format_1.md` |
| Format 2 | Labeled sections (`Direction:`, `Mood:`, `Camera:`, …) | `prompts/format_2.md` |
| Format 3 | Structured JSON object | `prompts/format_3.json` |

Results render in **Rendered / Raw (copy-ready)** tabs — format 3 renders as a JSON tree, with a fallback to raw output if the model ever returns invalid JSON — plus a download button (`.md` or `.json` per format; edit prompts download as `prompt_edit.md`) and a **💡 Use in Image Generation** button that switches modes and pre-fills the Image Generation prompt box. For the two image capabilities, the uploaded image is carried along too: Image Generation shows it as an attached input (removable with one click) and routes the request through the edits endpoint, so the edit loop is upload → describe → generate without re-attaching. Inputs and the last result survive Streamlit reruns.

## Getting started

Requirements: **Python 3.14+** and [uv](https://docs.astral.sh/uv/).

1. Install dependencies:

   ```pwsh
   uv sync
   ```

2. Create a `.env` file in the project root with the key(s) of the provider(s) you want to use:

   ```dotenv
   OPENAI_API_KEY=sk-...
   EVOLINK_API_KEY=...   # optional — only needed for the Evolink provider
   ```

3. Run the app:

   ```pwsh
   uv run streamlit run src/canvas_copilot/app.py
   ```

## Testing

The test suite covers the provider registry, Evolink payload/poll logic, package imports, the format registry, LLM message building, output handling (code-fence stripping, JSON parsing), and bare-mode rendering of both modes:

```pwsh
uv run pytest                # full suite (104 offline tests + 1 live test)
uv run pytest -m integration # live test: one real gpt-6-luna call (needs OPENAI_API_KEY)
uv run pytest -m "not integration"  # offline only (default behavior: live test auto-skips without a key)
```

`tests/test_imports.py` is the package-wide import smoke test: it fails loudly if any module stops importing cleanly after a move or rename.

## Project layout

```
src/canvas_copilot/
  app.py                    Entry point: sidebar provider + mode selectors, provider context, dispatch
  settings/providers.py     Provider registry: env keys, OpenAI SDK client, ProviderContext
  core/evolink_api.py       Evolink image-task API client (upload refs, create/poll tasks, download)
  modes/image_generation.py Image Generation mode (GPT Image models, both providers)
  modes/prompt_generation.py Prompt Generation mode (gpt-6-luna via the Responses API, both providers)
  modes/prompt_formats.py   Registry of the three prompt formats + the edit exemplar loader
  prompts/                  Exemplar files used as the injected prompt templates (gitignored)
tests/                      Pytest suite (imports smoke test + one live-API integration test)
PROMPTS.md                  Prompt engineering notes + library for the image models
```

## Notes

- **Aspect ratios are set in the UI, not the prompt**, for image generation — describe composition instead of writing "16:9" into the text (see `PROMPTS.md` for a full guide to prompting the image models).
- Prompt Generation always depicts clearly adult subjects; the system prompt enforces this in both capabilities.
- Adding a fourth prompt format is one registry entry in `prompt_formats.py` plus its exemplar file in `prompts/`.
