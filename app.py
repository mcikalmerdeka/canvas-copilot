"""Canvas Copilot — text-to-image generation with OpenAI's GPT Image model."""

import base64
import io
import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Load the environment variables and initialize the OpenAI client
# (same pattern as reference/generate_image.py).
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-image-2.5-sunburst"

SIZE_OPTIONS = {
    "auto": "Auto",
    "1024x1024": "1024 × 1024 · Square",
    "1024x1536": "1024 × 1536 · Portrait",
    "1536x1024": "1536 × 1024 · Landscape",
}
QUALITY_OPTIONS = {
    "auto": "Auto",
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "xhigh": "X-High",
    "max": "Max",
}

st.set_page_config(page_title="Canvas Copilot", page_icon="🎨", layout="centered")

st.title("🎨 Canvas Copilot")
st.caption("Describe anything — GPT Image paints it.")

# --- Input section ----------------------------------------------------------

prompt = st.text_area(
    "Prompt",
    placeholder="A quiet harbor at dawn, fishing boats resting on glassy water…",
    height=120,
)

# Optional reference images. With references attached, the app switches from
# the generations endpoint to the edits endpoint (client.images.edit).
reference_files = st.file_uploader(
    "Reference images (optional)",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
    help=(
        "Attach up to 16 reference images (PNG, JPEG, or WEBP, <50MB each). "
        "The model will use them as style, subject, or composition references."
    ),
)

if reference_files:
    ref_cols = st.columns(len(reference_files))
    for col, file in zip(ref_cols, reference_files):
        with col:
            st.image(file, caption=file.name, width="stretch")

size_col, quality_col = st.columns(2)
with size_col:
    size = st.selectbox("Size", list(SIZE_OPTIONS), format_func=SIZE_OPTIONS.get)
with quality_col:
    quality = st.selectbox("Quality", list(QUALITY_OPTIONS), format_func=QUALITY_OPTIONS.get)

generate = st.button("Generate", type="primary", width="stretch")

if generate:
    cleaned_prompt = prompt.strip()
    if not cleaned_prompt:
        st.warning("Please enter a prompt describing the image you want to create.")
    else:
        try:
            with st.spinner("Painting your image…"):
                if reference_files:
                    # The edits endpoint expects file-like objects, not raw bytes.
                    input_images = []
                    for file in reference_files:
                        buffer = io.BytesIO(file.getvalue())
                        buffer.name = file.name
                        input_images.append(buffer)
                    result = client.images.edit(
                        model=MODEL,
                        image=input_images,
                        prompt=cleaned_prompt,
                        size=size,
                        quality=quality,
                    )
                else:
                    result = client.images.generate(
                        model=MODEL,
                        prompt=cleaned_prompt,
                        size=size,
                        quality=quality,
                    )
                image_bytes = base64.b64decode(result.data[0].b64_json)
            # Keep the result so it survives reruns (e.g. download-button clicks).
            st.session_state["last_image"] = image_bytes
            st.session_state["last_prompt"] = cleaned_prompt
        except Exception as exc:
            st.error(f"Image generation failed: {exc}")

# --- Result section ---------------------------------------------------------

if st.session_state.get("last_image"):
    st.divider()
    st.image(
        st.session_state["last_image"],
        caption=st.session_state.get("last_prompt"),
        width="stretch",
    )
    st.download_button(
        "Download image",
        data=st.session_state["last_image"],
        file_name="generated_image.png",
        mime="image/png",
        width="stretch",
    )
