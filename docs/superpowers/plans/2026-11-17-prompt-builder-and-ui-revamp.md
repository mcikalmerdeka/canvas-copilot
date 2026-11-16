# Prompt Builder & UI Revamp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a guided prompt builder to Prompt Generation and restyle the app in the `reference_revamp` visual language without changing provider or image-generation behaviour.

**Architecture:** A pure `prompt_catalog` module owns facets, presets, and brief composition. `prompt_generation` gains a `guided` pipeline that sends the brief to `gpt-6-luna`. UI moves into `prompt_views` / `image_views` with shared helpers in `ui.py`; both mode modules keep thin `render()` entry points so existing tests and imports keep working.

**Tech Stack:** Python 3.14, Streamlit 1.63, OpenAI SDK 3.13, pytest 9.1.

**Spec:** `docs/superpowers/specs/2026-11-17-prompt-builder-and-ui-revamp-design.md`

## Global Constraints

- No new runtime dependencies.
- Keep `MODES == ("Image Generation", "Prompt Generation")` and both `render(ctx)` entry points.
- Keep `generate_prompt(client, capability, format_id=None, ...)` backward compatible; new params are keyword-only additions at the end.
- Keep session keys `app_provider`, `app_mode`, `pg_capability`, `pg_format`, `pg_result`, `pg_result_capability`, `pg_source`, `pg_result_image`, `pending_mode`, `image_gen_prompt`, `pending_handoff_image`, `handoff_image`, `last_images`, `last_prompt`, `last_format`.
- All builder character options describe clearly adult subjects only.
- No emoji in page titles or section headers; material icons on buttons.
- Tests must pass offline (`uv run pytest -q`).

## Review Focus

- Empty subject with facets selected must show a validation warning and never call the API.
- Switching capability, format, preset, or mode must not lose builder state or crash on a facet widget that is not currently rendered.
- The Image Generation handoff must carry the final prompt text (and image for image capabilities) exactly as before.
- Evolink vs OpenAI branches in Image Generation must keep their existing menus, payload branches, and hidden controls.
- Result rendering must still fall back to raw text when format 3 returns invalid JSON.

---

### Task 1: Prompt catalog and brief composition

**Files:**
- Create: `src/canvas_copilot/modes/prompt_catalog.py`
- Test: `tests/test_prompt_catalog.py`

**Interfaces:**
- Produces:
  - `SUBJECT_TYPES: tuple[str, ...]`
  - `GROUPS: tuple[tuple[str, str], ...]`
  - `DEFAULT_LABEL: str`
  - `@dataclass(frozen=True) FacetOption(label: str, text: str)`
  - `@dataclass(frozen=True) Facet(id: str, label: str, group: str, help: str, options: tuple[FacetOption, ...] = (), kind: str = "select", applies_to: tuple[str, ...] = ())`
  - `FACETS: tuple[Facet, ...]`, `FACET_BY_ID: dict[str, Facet]`
  - `PRESETS: dict[str, dict[str, object]]` with keys `label`, `subject_type`, `subject`, `values`
  - `facets_for(subject_type: str) -> tuple[Facet, ...]`
  - `option_text(facet_id: str, label: str) -> str` (returns `""` for `DEFAULT_LABEL` / unknown)
  - `build_brief(subject: str, subject_type: str, selections: Mapping[str, str | Sequence[str]], notes: str = "") -> str`
  - `randomize_selections(subject_type: str, rng: random.Random | None = None) -> dict[str, str | list[str]]`
  - `preset_values(preset_id: str) -> dict[str, str | list[str]]`

- [ ] Write catalog tests: unique ids; every option has prose; character facets only for people/group; brief contains mapped prose and omits unspecified; multiselect joins prose; presets reference valid facet/option labels; randomize is deterministic with a seeded `random.Random` and valid.
- [ ] Implement the catalog with all facets and six presets.
- [ ] Run `uv run pytest tests/test_prompt_catalog.py -q`.
- [ ] Run full suite to confirm no regressions.

### Task 2: Guided pipeline

**Files:**
- Modify: `src/canvas_copilot/modes/prompt_generation.py`
- Test: `tests/test_prompt_generation.py`

**Interfaces:**
- Produces:
  - `CAPABILITY_BUILDER = "Prompt Builder"`
  - `PIPELINE_GUIDED = "guided"`
  - `GUIDED_USER_TEMPLATE: str`
  - `build_guided_system_prompt(format_id: str) -> str`
  - `build_user_content(..., guided_brief: str | None = None) -> str | list[dict]`
  - `generate_prompt(..., guided_brief: str | None = None) -> str`

- [ ] Add failing tests: guided system prompt includes exemplar + hard-requirement language; guided user payload includes the whole brief; `generate_prompt` forwards it; legacy pipelines unchanged.
- [ ] Implement the guided branch without touching existing payload branches.
- [ ] Run targeted and full tests.

### Task 3: Theme and shared UI shell

**Files:**
- Create: `.streamlit/config.toml`
- Create: `src/canvas_copilot/ui.py`
- Modify: `src/canvas_copilot/app.py`
- Test: `tests/test_app_smoke.py`

**Interfaces:**
- Produces:
  - `ui.page_header(title: str, caption: str, badges: Sequence[tuple[str, str | None, str]] = ()) -> None`
  - `ui.section(title: str, caption: str = "") -> None`
  - `ui.empty_state(title: str, body: str, icon: str) -> None`
  - `ui.result_actions(...)` is kept local to prompt views to avoid premature abstraction.

- [ ] Add theme config (warm paper, teal, Fraunces/IBM Plex, dark variant).
- [ ] Add `ui.py` helpers using `st.badge`, material icons and bounded layout.
- [ ] Rework `app.py`: sidebar provider radio with key status; main header and badges; segmented mode switch; preserve `pending_mode` logic and dispatch; default Image Generation.
- [ ] Update smoke tests for the new shell; assert both modes still render bare and dispatch default is image.
- [ ] Run tests.

### Task 4: Prompt Generation views

**Files:**
- Create: `src/canvas_copilot/modes/prompt_views.py`
- Modify: `src/canvas_copilot/modes/prompt_generation.py` (delegate `render`)
- Test: `tests/test_prompt_catalog.py` (builder state helpers), `tests/test_app_smoke.py`

**Interfaces:**
- Consumes: catalog functions, pipeline functions, `ui` helpers.
- Produces: `render_mode(ctx) -> None`; internal `_builder_values()` and `_apply_preset()` / `_randomize()` / `_reset()` callbacks.

- [ ] Builder UI: capability/format row; preset+randomize+reset; starting-prompt card; subject card; persistent section selector (no collapsing expanders); live brief preview; generate button; validation warning when both starting prompt and subject are empty.
- [ ] Legacy capability forms in cards (Upgrade, Image-to-Prompt, Image-Edit), keeping current fields, help text, and validation.
- [ ] Result card with Rendered/Raw tabs, JSON fallback, download, Use in Image Generation.
- [ ] `prompt_generation.render` delegates to `prompt_views.render_mode`.
- [ ] Smoke and AppTest tests for section persistence and starting-prompt preservation.

### Task 5: Image Generation views

**Files:**
- Create: `src/canvas_copilot/modes/image_views.py`
- Modify: `src/canvas_copilot/modes/image_generation.py` (delegate `render`)
- Test: `tests/test_app_smoke.py`

**Interfaces:**
- Consumes: image_generation helpers/constants and Evolink registry.
- Produces: `render_mode(ctx) -> None`.

- [ ] Move the existing render body into `image_views`, preserving every control/key/help text and branch.
- [ ] Reorganize into two columns with a settings card and empty/results states.
- [ ] `image_generation.render` delegates; existing pure-function tests untouched.
- [ ] Smoke tests for OpenAI and Evolink.

### Task 6: Verification and docs

**Files:**
- Modify: `README.md`
- Test: full suite

- [ ] Add README section for the Prompt Builder and the new UI.
- [ ] Run `uv run pytest -q` and confirm all tests pass.
- [ ] Launch `uv run streamlit run src/canvas_copilot/app.py --server.headless true` and confirm clean startup and HTTP 200.
- [ ] Review diff for accidental behavior changes; stage changes for the owner to commit.
