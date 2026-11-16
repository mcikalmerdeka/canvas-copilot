# Prompt Builder & UI Revamp — Design

**Date:** 2026-11-17
**Status:** Approved by owner
**Scope:** Canvas Copilot Streamlit app (`src/canvas_copilot`)

## Goal

Replace Prompt Generation's single freeform text box as the primary entry point with a guided,
facet-based prompt builder, and apply the `reference_revamp` visual language across the whole app.
The builder converts curated selections into a structured creative brief which `gpt-6-luna` rewrites
into one of the three existing prompt formats. The user inspects, copies, downloads, or hands the
finished prompt to Image Generation exactly as before.

## Non-goals

- No new image model, provider, or network service.
- No change to image-generation payloads, model menus, moderation handling, or Evolink logic.
- No removal of the existing Upgrade Prompt, Image-to-Prompt, or Image-Edit Prompt capabilities.
- No image thumbnails or stock photography in the builder; options are curated text.
- No direct prompt assembly instead of the LLM: the exemplar formats remain the output contract.

## User flow

1. Open Prompt Generation. The default capability is **Prompt Builder**.
2. Pick a preset or fill the facets: subject type, subject, character, scene, camera, style, finishing.
3. Watch the compiled brief update live in the right-hand preview card.
4. Choose an output format (Narrative, Labeled sections, or JSON) and press **Generate Prompt**.
5. Inspect the result in Rendered / Raw tabs, copy or download it, or press
   **Use in Image Generation** to hand the final prompt to the image mode.
6. Switch to Upgrade Prompt, Image-to-Prompt, or Image-Edit Prompt at any time; builder state survives.

## Builder catalog

Facets are curated text, grouped in five UI sections. Every option maps a short label to rich prose
that the model receives. Character facets only appear for `Person` and `Group` subject types.

| Group | Facets |
|---|---|
| Subject | `subject` (free text, required), `subject_type` |
| Character | age band, gender presentation, hair style, hair colour, outfit, pose, expression, character details (free text) |
| Scene | setting, time of day, lighting, weather/atmosphere, props (free text) |
| Camera | framing, camera angle, lens, depth of field, composition |
| Style | medium/style, colour palette, mood |
| Finishing | extra notes (free text), negative constraints (multi-select + custom negatives) |

Option labels are plain-language; option text is a complete descriptive phrase. Adult-only subject
language is preserved in every character option. The catalog lives in one module so adding an option
is one entry.

### Presets

One-click starting points that fill facets and a sample subject, fully editable afterwards:

- Editorial portrait
- Candid street
- Golden-hour lifestyle
- Studio product
- Moody cinematic
- Travel landscape
- Illustration

A `Start fresh` preset clears the builder.

### Brief format

`build_brief()` emits labeled sections in a fixed order, omitting empty facets:

```text
SUBJECT TYPE
Person

SUBJECT
A woman in her early 30s with a calm, self-assured presence.

CHARACTER
Age: Adult in their early 30s
Hair: Sleek low bun, dark brown
Outfit: Tailored charcoal suit with a silk shirt
Pose: Standing with relaxed confidence, weight on one leg
Expression: Quiet confidence, direct gaze

SCENE
Setting: Seamless studio backdrop
Lighting: Single softbox from the left with a subtle rim light

CAMERA
Framing: Medium close-up
Angle: Eye level
Lens: 85mm portrait lens

STYLE
Medium: Editorial fashion photography
Palette: Warm neutrals
Mood: Quiet confidence

FINISHING TOUCHES
Avoid: No text or watermarks, No plastic skin
```

The brief is an intermediate representation only; it is never shown as the final result.

## Prompt pipeline

`prompt_generation.py` gains:

- `CAPABILITY_BUILDER = "Prompt Builder"` and `PIPELINE_GUIDED = "guided"`.
- `build_guided_system_prompt(format_id)`: the existing exemplar system prompt plus a hard requirement
  that every supplied facet survives into the finished prompt, that absent facets are filled
  consistently, and that nothing contradicts the brief.
- `build_user_content(..., guided_brief=...)` returning a plain-text guided user message.
- `generate_prompt(..., guided_brief=...)` forwarding the brief.
- The existing `text`, `image`, and `edit` pipelines remain unchanged in signature and behaviour.

The model, reasoning effort, max tokens, code-fence stripping, JSON parsing, filenames, and result
handling are unchanged. Builder results download as the selected format and hand off to Image
Generation through the existing `image_gen_prompt` + `pending_mode` mechanism.

## UI

### Global shell

- New `.streamlit/config.toml` using the `reference_revamp` palette: warm paper light theme, matching
  dark theme, teal primary, Fraunces headings, IBM Plex Sans body, IBM Plex Mono code, 10px radii,
  widget borders, compact heading scale.
- `app.py`: wide layout. Slim sidebar with the provider radio, key status, and a short
  "About" caption. Main area has a title, caption, badge row (provider, model, mode), and a
  segmented mode switch. The `pending_mode` handoff still selects Image Generation.
- New `ui.py` for shared presentation helpers: page header, badges, section label, empty state,
  result action row. No emoji headers; material icons on buttons.

### Prompt Generation

- Capability segmented control and output-format control in a compact row.
- Builder: two-column split on wide screens. Left = preset row, subject card, and five collapsible
  facet sections arranged in two-column grids. Right = bordered **Brief preview** card with a
  copyable `st.code` block, character count, primary Generate button, and Randomize / Reset actions.
- Presets and Randomize use widget callbacks that write the `pb_*` session keys before rerun.
- Upgrade Prompt, Image-to-Prompt, and Image-Edit Prompt keep their current fields and validation,
  rendered inside the new cards and columns.
- Result: full-width card with source caption, Rendered / Raw tabs, download, and Use in Image
  Generation. Existing JSON fallback and edit-prompt metadata are preserved.

### Image Generation

- Header and badge row.
- Two-column workspace: left = prompt, reference uploader, handoff image preview/removal;
  right = bordered output-settings card containing every existing control (model, size, quality,
  resolution/speed, variations, format, compression, negative prompt, moderation) and the Generate
  button.
- Results render below in a responsive grid with per-image download buttons; empty state explains the
  workflow.
- Generation code is split from view code (`image_views.py`) but behaviour is identical.

## Files

| File | Responsibility |
|---|---|
| `src/canvas_copilot/modes/prompt_catalog.py` | facet/option/preset registry, brief composition, randomization |
| `src/canvas_copilot/modes/prompt_views.py` | Prompt Generation UI (builder + three legacy capabilities + result) |
| `src/canvas_copilot/modes/image_views.py` | Image Generation UI |
| `src/canvas_copilot/ui.py` | shared shell/presentation helpers |
| `src/canvas_copilot/modes/prompt_generation.py` | guided pipeline + render delegation |
| `src/canvas_copilot/modes/image_generation.py` | generation logic + render delegation |
| `src/canvas_copilot/app.py` | shell, provider sidebar, mode switch, dispatch |
| `.streamlit/config.toml` | theme tokens |
| `tests/test_prompt_catalog.py` | catalog and brief tests |
| `tests/test_prompt_generation.py` | guided payload tests (extend) |
| `tests/test_app_smoke.py` | new-layout smoke tests (extend) |

## Verification

- `uv run pytest -q` green (existing 170 tests plus new ones).
- Catalog tests: unique facet/option ids, every option has non-empty prose, character facets only for
  people/group, brief includes selected prose and omits unselected, presets reference valid options,
  randomize returns valid values for the subject type.
- Pipeline tests: guided system prompt contains the exemplar and the hard requirement; guided user
  message contains every facet; existing pipelines unchanged.
- Smoke tests: both modes render in bare mode with no API calls; layout helpers import.
- Manual launch: `uv run streamlit run src/canvas_copilot/app.py` starts cleanly; builder preview,
  generate button, and image workspace render.

## Risks & mitigations

- **Session-state regressions** from moving widgets out of the sidebar and renaming keys:
  keep capability/result keys (`pg_*`), handoff keys (`pending_mode`, `image_gen_prompt`,
  `pending_handoff_image`), and generation keys (`last_images`, `last_prompt`, `last_format`).
  Smoke tests plus manual flow cover the handoff.
- **Theme config incompatibility**: keys were verified against Streamlit 1.63's config schema.
- **Builder becoming a wall of controls**: five collapsed sections, two-column grids, presets, and
  the live preview keep it scannable; only Subject is always open.
- **LLM omitting facets**: the guided system prompt makes each supplied facet a hard requirement and
  the brief repeats them as labeled sections.
- **File growth**: UI split into `prompt_views.py` / `image_views.py`; pipeline modules stay focused.

## Research references

- Midjourney prompt anatomy: subject → setting → style → lighting → camera → colour/mood → parameters.
- Leonardo AI image guidance: presets, character/style/pose references, adherence strength.
- Image Prompt Maker: visual controls compiling to a structured prompt with live preview.
- ComfyUI Photoreal Prompt Builder: pose, scene, lighting setup, photo style, shot type, camera angle,
  lens, depth of field, colour grading, hair facets.
