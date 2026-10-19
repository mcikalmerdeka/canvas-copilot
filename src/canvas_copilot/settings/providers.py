"""Provider registry — which backend Canvas Copilot talks to.

Either of two providers is selected in the sidebar, and the choice flows to
both modes through `ProviderContext`:

- **OpenAI** — the official OpenAI API (`OPENAI_API_KEY`), default endpoints.
- **Evolink** — the EvoLink.AI aggregator (`EVOLINK_API_KEY`). Its GPT series
  exposes an OpenAI-compatible Responses API at `direct.evolink.ai`, so text
  calls (Prompt Generation) work through the regular OpenAI SDK client pointed
  at that base URL. Its image series runs on an async task API instead, which
  lives in `evolink_api.py`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from openai import OpenAI

PROVIDER_OPENAI = "OpenAI"
PROVIDER_EVOLINK = "Evolink"

PROVIDERS = (PROVIDER_OPENAI, PROVIDER_EVOLINK)

PROVIDER_INFO = {
    PROVIDER_OPENAI: {
        "label": "OpenAI (official API)",
        "env_var": "OPENAI_API_KEY",
        "responses_base_url": None,
    },
    PROVIDER_EVOLINK: {
        "label": "Evolink (aggregator)",
        "env_var": "EVOLINK_API_KEY",
        "responses_base_url": "https://direct.evolink.ai/v1",
    },
}


@dataclass(frozen=True)
class ProviderContext:
    """Everything the modes need to talk to the selected provider."""

    provider_id: str
    api_key: str
    client: OpenAI

    @property
    def is_evolink(self) -> bool:
        return self.provider_id == PROVIDER_EVOLINK


def get_api_key(provider_id: str) -> str | None:
    """Read the selected provider's API key from the environment."""
    env_var = PROVIDER_INFO[provider_id]["env_var"]
    return os.getenv(env_var) or None


def missing_key_message(provider_id: str) -> str:
    """Error text shown when the selected provider has no API key configured."""
    env_var = PROVIDER_INFO[provider_id]["env_var"]
    return (
        f"Missing {env_var} — add it to your .env file (or switch the provider "
        "in the sidebar) and restart the app."
    )


def build_context(provider_id: str, api_key: str) -> ProviderContext:
    """Build the provider context with an OpenAI SDK client for text calls."""
    base_url = PROVIDER_INFO[provider_id]["responses_base_url"]
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    return ProviderContext(provider_id=provider_id, api_key=api_key, client=client)
