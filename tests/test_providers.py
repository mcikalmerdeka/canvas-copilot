"""Tests for the provider registry (key lookup, text client construction)."""

from __future__ import annotations

import canvas_copilot.settings.providers as pr

OPENAI_ENV_VAR = pr.PROVIDER_INFO[pr.PROVIDER_OPENAI]["env_var"]
EVOLINK_ENV_VAR = pr.PROVIDER_INFO[pr.PROVIDER_EVOLINK]["env_var"]


class TestProviderRegistry:
    def test_lists_openai_first_then_evolink(self) -> None:
        assert pr.PROVIDERS == ("OpenAI", "Evolink")

    def test_openai_reads_the_openai_api_key_env_var(self) -> None:
        assert OPENAI_ENV_VAR == "OPENAI_API_KEY"

    def test_evolink_reads_the_evolink_api_key_env_var(self) -> None:
        assert EVOLINK_ENV_VAR == "EVOLINK_API_KEY"

    def test_openai_text_calls_go_to_the_default_openai_endpoint(self) -> None:
        assert pr.PROVIDER_INFO[pr.PROVIDER_OPENAI]["responses_base_url"] is None

    def test_evolink_text_calls_go_to_the_direct_gateway(self) -> None:
        assert (
            pr.PROVIDER_INFO[pr.PROVIDER_EVOLINK]["responses_base_url"]
            == "https://direct.evolink.ai/v1"
        )

    def test_every_provider_has_a_display_label(self) -> None:
        for provider_id in pr.PROVIDERS:
            assert pr.PROVIDER_INFO[provider_id]["label"]


class TestApiKeyLookup:
    def test_returns_the_value_of_the_selected_env_var(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
        monkeypatch.setenv("EVOLINK_API_KEY", "ek-evolink-test")
        assert pr.get_api_key(pr.PROVIDER_OPENAI) == "sk-openai-test"
        assert pr.get_api_key(pr.PROVIDER_EVOLINK) == "ek-evolink-test"

    def test_missing_key_returns_none(self, monkeypatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("EVOLINK_API_KEY", raising=False)
        assert pr.get_api_key(pr.PROVIDER_OPENAI) is None
        assert pr.get_api_key(pr.PROVIDER_EVOLINK) is None


class TestMissingKeyMessage:
    def test_message_names_the_env_var_to_add(self) -> None:
        message = pr.missing_key_message(pr.PROVIDER_EVOLINK)
        assert "EVOLINK_API_KEY" in message


class TestBuildContext:
    def test_openai_context_uses_the_default_openai_client(self) -> None:
        ctx = pr.build_context(pr.PROVIDER_OPENAI, "sk-test")
        assert ctx.provider_id == pr.PROVIDER_OPENAI
        assert ctx.api_key == "sk-test"
        assert str(ctx.client.base_url) == "https://api.openai.com/v1/"

    def test_evolink_context_points_the_openai_sdk_at_the_gateway(self) -> None:
        ctx = pr.build_context(pr.PROVIDER_EVOLINK, "ek-test")
        assert str(ctx.client.base_url) == "https://direct.evolink.ai/v1/"
        assert ctx.client.api_key == "ek-test"

    def test_evolink_context_flags_itself_for_provider_branches(self) -> None:
        assert pr.build_context(pr.PROVIDER_EVOLINK, "k").is_evolink is True
        assert pr.build_context(pr.PROVIDER_OPENAI, "k").is_evolink is False
