"""Canvas Copilot — a Streamlit app with two modes and two providers.

- Image Generation: text-to-image with OpenAI's GPT Image models (modes/image_generation.py).
- Prompt Generation: guided prompt building, raw-prompt upgrades, or prompt
  reverse-engineering from images with `gpt-6-luna` (modes/prompt_generation.py).

A sidebar provider selector picks the backend for both modes
(settings/providers.py): **OpenAI** (official API) or **Evolink** (aggregator,
EVOLINK_API_KEY).
"""

import logging

import streamlit as st
from dotenv import load_dotenv

from canvas_copilot import ui
from canvas_copilot.modes import image_generation, prompt_generation
from canvas_copilot.settings import providers

MODES = ("Image Generation", "Prompt Generation")

# Terminal progress logging for generation milestones (source image uploads,
# task progress, prompt/image finished, failures). Root stays at WARNING so
# third-party chatter (openai/httpx DEBUG|INFO) doesn't flood the console;
# only this package's loggers emit at INFO.
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("canvas_copilot").setLevel(logging.INFO)

load_dotenv()

st.set_page_config(
    page_title="Canvas Copilot",
    page_icon=":material/palette:",
    layout="wide",
)

# --- Sidebar: provider --------------------------------------------------------------

provider_id = st.sidebar.radio(
    "Provider",
    providers.PROVIDERS,
    format_func=lambda provider: providers.PROVIDER_INFO[provider]["label"],
    key="app_provider",
    horizontal=True,
    help="Which backend runs the models. Each has its own API key in .env.",
)

api_key = providers.get_api_key(provider_id)
if api_key:
    st.sidebar.badge("Key connected", icon=":material/key:", color="green")
else:
    st.sidebar.badge("Key missing", icon=":material/key_off:", color="red")

st.sidebar.caption(
    "Prompt Generation runs on `gpt-6-luna`. Image Generation uses the selected "
    "provider's GPT Image catalog."
)
st.sidebar.divider()
st.sidebar.caption("Set keys in `.env`, then restart the app.")

# --- Mode switch --------------------------------------------------------------------

# A handoff (e.g. "Use in Image Generation") stages the new mode here, because
# the segmented control has already been instantiated by the time the button
# inside a mode is clicked.
pending_mode = st.session_state.pop("pending_mode", None)
if pending_mode in MODES:
    st.session_state["app_mode"] = pending_mode

preview_mode = st.session_state.get("app_mode", MODES[0])
if preview_mode not in MODES:
    preview_mode = MODES[0]
if "app_mode" not in st.session_state:
    st.session_state["app_mode"] = preview_mode

ui.page_header(
    "Canvas Copilot",
    (
        "Build a richly structured prompt, inspect it, then hand it straight to "
        "the image workspace."
        if preview_mode == "Prompt Generation"
        else "Describe anything. Attach references when you want control. Paint it."
    ),
    badges=(
        (provider_id, ":material/cloud:", "blue"),
        (
            "gpt-6-luna" if preview_mode == "Prompt Generation" else "GPT Image",
            ":material/model_training:",
            "violet",
        ),
        (preview_mode, ":material/tune:", "gray"),
    ),
)

mode = st.segmented_control(
    "Mode",
    MODES,
    key="app_mode",
    required=True,
    label_visibility="collapsed",
    help="Pick what Canvas Copilot should help you do.",
)

# --- Provider context ---------------------------------------------------------------

if not api_key:
    st.error(providers.missing_key_message(provider_id))
    # st.stop() halts a real session right here; in bare-mode imports it is a
    # no-op, so the context stays unset and dispatch below is skipped.
    st.stop()
    ctx = None
else:
    ctx = providers.build_context(provider_id, api_key)

# --- Dispatch -----------------------------------------------------------------------

if ctx:
    if mode == "Prompt Generation":
        prompt_generation.render(ctx)
    else:
        image_generation.render(ctx)
