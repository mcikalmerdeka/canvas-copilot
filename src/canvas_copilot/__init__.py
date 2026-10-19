"""Canvas Copilot — a Streamlit app with two modes (Image / Prompt Generation)
and two selectable providers (OpenAI, Evolink), laid out as:

    canvas_copilot.app                    Streamlit entry point
    canvas_copilot.settings.providers     provider registry + context
    canvas_copilot.core.evolink_api       Evolink image-task REST client
    canvas_copilot.modes.*                the two modes + prompt format registry
    canvas_copilot.prompts/               exemplar files used by prompt_formats
"""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
