"""Canvas Copilot — a Streamlit app with two modes and two providers.

- Image Generation: text-to-image with OpenAI's GPT Image models (modes/image_generation.py).
- Prompt Generation: upgrade raw prompts or reverse-engineer prompts from images
  with `gpt-6-luna` (modes/prompt_generation.py).

A sidebar "Provider" radio picks the backend for both modes (settings/providers.py):
**OpenAI** (official API) or **Evolink** (aggregator, EVOLINK_API_KEY).
"""

import logging

import streamlit as st
from dotenv import load_dotenv

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

st.set_page_config(page_title="Canvas Copilot", page_icon="🎨", layout="centered")

# --- Sidebar: provider + app mode -------------------------------------------------

st.sidebar.markdown("### 🎨 Canvas Copilot")

provider_id = st.sidebar.radio(
    "Provider",
    providers.PROVIDERS,
    format_func=lambda provider: providers.PROVIDER_INFO[provider]["label"],
    key="app_provider",
    help="Which backend runs the models. Each has its own API key in .env.",
)
st.sidebar.divider()

# A handoff (e.g. "Use in Image Generation") stages the new mode here, because
# the radio widget has already been instantiated by the time the button is clicked.
pending_mode = st.session_state.pop("pending_mode", None)
if pending_mode in MODES:
    st.session_state["app_mode"] = pending_mode

mode = st.sidebar.radio("App mode", MODES, key="app_mode", help="Pick what Canvas Copilot should help you do.")
st.sidebar.caption("Canvas Copilot helps you craft prompts and paint images.")

# --- Provider context ---------------------------------------------------------------

api_key = providers.get_api_key(provider_id)
if not api_key:
    st.error(providers.missing_key_message(provider_id))
    # st.stop() halts a real session right here; in bare-mode imports it is a
    # no-op, so the context stays unset and dispatch below is skipped.
    st.stop()
    ctx = None
else:
    ctx = providers.build_context(provider_id, api_key)

# --- Dispatch ------------------------------------------------------------------------

if ctx:
    if mode == "Image Generation":
        image_generation.render(ctx)
    else:
        prompt_generation.render(ctx)
