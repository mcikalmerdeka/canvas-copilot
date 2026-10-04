"""Canvas Copilot — a Streamlit app with two modes.

- Image Generation: text-to-image with OpenAI's GPT Image models (image_generation.py).
- Prompt Generation: upgrade raw prompts or reverse-engineer prompts from images
  with `gpt-6-luna` (prompt_generation.py).
"""

import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

import image_generation
import prompt_generation

MODES = ("Image Generation", "Prompt Generation")

load_dotenv()

st.set_page_config(page_title="Canvas Copilot", page_icon="🎨", layout="centered")

# --- Sidebar: app mode --------------------------------------------------------

st.sidebar.markdown("### 🎨 Canvas Copilot")

# A handoff (e.g. "Use in Image Generation") stages the new mode here, because
# the radio widget has already been instantiated by the time the button is clicked.
pending_mode = st.session_state.pop("pending_mode", None)
if pending_mode in MODES:
    st.session_state["app_mode"] = pending_mode

mode = st.sidebar.radio("App mode", MODES, key="app_mode", help="Pick what Canvas Copilot should help you do.")
st.sidebar.divider()
st.sidebar.caption("Canvas Copilot helps you craft prompts and paint images.")

# --- Shared OpenAI client -------------------------------------------------------

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("Missing OPENAI_API_KEY — add it to your .env file and restart the app.")
    st.stop()
client = OpenAI(api_key=api_key)

# --- Dispatch --------------------------------------------------------------------

if mode == "Image Generation":
    image_generation.render(client)
else:
    prompt_generation.render(client)
