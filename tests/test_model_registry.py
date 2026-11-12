"""Tests for the Evolink model registry (catalog, per-model option shapes).

The registry is pure data + helpers; these tests pin it against the facts in
the official OpenAPI specs for each model so a future edit that drifts from a
model's real schema fails here.
"""

from __future__ import annotations

import canvas_copilot.core.model_registry as mr


class TestRegistryCatalog:
    def test_eleven_models_in_the_documented_order(self) -> None:
        ids = tuple(spec.model_id for spec in mr.EVOLINK_MODELS.values())
        assert ids == (
            "gpt-image-2.6-sunburst",
            "gpt-image-2.6-flare",
            "gpt-image-2.5-sunburst",
            "gpt-image-2.5-flare",
            "gemini-nano-banana-2.1",
            "grok-imagine-image-2.0",
            "mj-v8.2",
            "qwen-image-3.0",
            "qwen-image-3.0-pro",
            "doubao-seedream-5.5",
            "minimax-h3-image",
        )

    def test_labels_are_unique_and_human_readable(self) -> None:
        labels = [spec.label for spec in mr.EVOLINK_MODELS.values()]
        assert len(set(labels)) == len(labels)
        assert all(label.strip() for label in labels)


class TestDisplayLabels:
    def test_no_model_label_carries_a_parenthetical_description(self) -> None:
        labels = [spec.label for spec in mr.EVOLINK_MODELS.values()]
        assert all("(" not in label and ")" not in label for label in labels)

    def test_gpt_image_25_labels_name_the_family_and_version(self) -> None:
        assert mr.EVOLINK_MODELS["gpt-image-2.5-sunburst"].label == "GPT Sunburst 2.5"
        assert mr.EVOLINK_MODELS["gpt-image-2.5-flare"].label == "GPT Flare 2.5"

    def test_gpt_image_26_labels_name_the_family_and_version(self) -> None:
        assert mr.EVOLINK_MODELS["gpt-image-2.6-sunburst"].label == "GPT Sunburst 2.6"
        assert mr.EVOLINK_MODELS["gpt-image-2.6-flare"].label == "GPT Flare 2.6"

    def test_every_gpt_model_label_starts_with_the_family_name(self) -> None:
        for model_id in (
            "gpt-image-2.6-sunburst", "gpt-image-2.6-flare",
            "gpt-image-2.5-sunburst", "gpt-image-2.5-flare",
        ):
            assert mr.EVOLINK_MODELS[model_id].label.startswith("GPT ")


class TestGptImage26Models:
    def test_sizes_cover_auto_the_fifteen_ratios_and_pixels(self) -> None:
        for model_id in ("gpt-image-2.6-sunburst", "gpt-image-2.6-flare"):
            values = [value for value, _ in mr.EVOLINK_MODELS[model_id].size_options]
            assert "auto" in values
            assert {"1:1", "16:9", "21:9", "1:3", "3:1"} <= set(values)
            assert len([value for value in values if ":" in value]) == 15
            assert "1024x1024" in values

    def test_quality_tiers_are_the_token_scale_without_auto(self) -> None:
        for model_id in ("gpt-image-2.6-sunburst", "gpt-image-2.6-flare"):
            spec = mr.EVOLINK_MODELS[model_id]
            assert [value for value, _ in spec.quality_options] == [
                "low", "medium", "high", "xhigh", "max",
            ]
            assert spec.quality_default == "medium"

    def test_resolution_is_a_separate_1k_2k_4k_tier(self) -> None:
        for model_id in ("gpt-image-2.6-sunburst", "gpt-image-2.6-flare"):
            spec = mr.EVOLINK_MODELS[model_id]
            assert [value for value, _ in spec.resolution_options] == ["1K", "2K", "4K"]
            assert spec.resolution_default == "1K"

    def test_format_variations_and_reference_support(self) -> None:
        for model_id in ("gpt-image-2.6-sunburst", "gpt-image-2.6-flare"):
            spec = mr.EVOLINK_MODELS[model_id]
            assert spec.supports_output_format is True
            assert spec.max_reference_images == 16
            assert spec.max_variations == 10

    def test_both_variants_have_distinct_human_readable_labels(self) -> None:
        sunburst = mr.EVOLINK_MODELS["gpt-image-2.6-sunburst"].label
        flare = mr.EVOLINK_MODELS["gpt-image-2.6-flare"].label
        assert sunburst != flare
        assert "2.6" in sunburst and "2.6" in flare


class TestGptImageModels:
    def test_sizes_cover_auto_and_the_pixel_formats(self) -> None:
        for model_id in ("gpt-image-2.5-sunburst", "gpt-image-2.5-flare"):
            spec = mr.EVOLINK_MODELS[model_id]
            values = [value for value, _ in spec.size_options]
            assert "auto" in values
            assert "1024x1024" in values
            assert "1536x1024" in values

    def test_quality_tiers_are_the_token_scale_without_auto(self) -> None:
        spec = mr.EVOLINK_MODELS["gpt-image-2.5-sunburst"]
        assert [value for value, _ in spec.quality_options] == ["low", "medium", "high", "xhigh", "max"]
        assert spec.quality_default == "medium"

    def test_format_and_wide_reference_support(self) -> None:
        spec = mr.EVOLINK_MODELS["gpt-image-2.5-flare"]
        assert spec.supports_output_format is True
        assert spec.max_reference_images == 16
        assert spec.max_variations == 10


class TestNanoBanana:
    def test_quality_field_holds_resolution_tiers(self) -> None:
        spec = mr.EVOLINK_MODELS["gemini-nano-banana-2.1"]
        assert [value for value, _ in spec.quality_options] == ["1K", "2K", "4K"]
        assert spec.quality_default == "2K"

    def test_sizes_are_ratio_only(self) -> None:
        spec = mr.EVOLINK_MODELS["gemini-nano-banana-2.1"]
        values = [value for value, _ in spec.size_options]
        assert "auto" in values
        assert "16:9" in values
        assert "21:9" in values
        assert all("x" not in value for value in values)  # no pixel sizes

    def test_one_image_per_request_with_fourteen_references(self) -> None:
        spec = mr.EVOLINK_MODELS["gemini-nano-banana-2.1"]
        assert spec.max_variations == 1
        assert spec.max_reference_images == 14
        assert spec.supports_output_format is False


class TestGrokImagine:
    def test_quality_is_two_tiers_with_a_separate_resolution(self) -> None:
        spec = mr.EVOLINK_MODELS["grok-imagine-image-2.0"]
        assert [value for value, _ in spec.quality_options] == ["low", "medium"]
        assert spec.quality_default == "medium"
        assert [value for value, _ in spec.resolution_options] == ["1K", "2K"]
        assert spec.resolution_default == "1K"

    def test_sizes_are_ratio_only(self) -> None:
        spec = mr.EVOLINK_MODELS["grok-imagine-image-2.0"]
        values = [value for value, _ in spec.size_options]
        assert "auto" in values and "16:9" in values and "20:9" in values
        assert all("x" not in value for value in values)

    def test_variations_and_references(self) -> None:
        spec = mr.EVOLINK_MODELS["grok-imagine-image-2.0"]
        assert spec.max_variations == 10
        assert spec.max_reference_images == 3


class TestMidjourney:
    def test_aspect_ratio_comes_from_the_prompt_not_a_size_param(self) -> None:
        spec = mr.EVOLINK_MODELS["mj-v8.2"]
        assert spec.size_options is None
        assert spec.aspect_via_prompt is True
        assert "16:9" in mr.ASPECT_RATIOS_MJ

    def test_references_travel_inside_the_prompt(self) -> None:
        spec = mr.EVOLINK_MODELS["mj-v8.2"]
        assert spec.references_in_prompt is True
        assert spec.max_reference_images == 3

    def test_quality_is_standard_or_hd_with_a_speed_choice(self) -> None:
        spec = mr.EVOLINK_MODELS["mj-v8.2"]
        assert [value for value, _ in spec.quality_options] == ["standard", "hd"]
        assert spec.quality_default == "standard"
        # Default first so the plain selectbox (index 0) pre-selects it.
        assert [value for value, _ in spec.speed_options] == ["fast", "draft"]
        assert spec.speed_default == "fast"

    def test_fixed_image_count_per_generation(self) -> None:
        assert mr.EVOLINK_MODELS["mj-v8.2"].max_variations == 1


class TestQwenModels:
    def test_both_editions_are_registered_with_negative_prompt_support(self) -> None:
        for model_id in ("qwen-image-3.0", "qwen-image-3.0-pro"):
            spec = mr.EVOLINK_MODELS[model_id]
            assert spec.supports_negative_prompt is True
            assert spec.max_variations == 6
            assert spec.max_reference_images == 3

    def test_quality_is_a_resolution_tier_defaulting_to_1k(self) -> None:
        spec = mr.EVOLINK_MODELS["qwen-image-3.0-pro"]
        assert [value for value, _ in spec.quality_options] == ["1K", "2K"]
        assert spec.quality_default == "1K"

    def test_sizes_allow_ratios_and_pixels(self) -> None:
        spec = mr.EVOLINK_MODELS["qwen-image-3.0"]
        values = [value for value, _ in spec.size_options]
        assert "auto" in values
        assert "16:9" in values
        assert "1024x1024" in values


class TestSeedream55:
    def test_model_id_and_label(self) -> None:
        spec = mr.EVOLINK_MODELS["doubao-seedream-5.5"]
        assert spec.model_id == "doubao-seedream-5.5"
        assert spec.label == "Seedream 5.5"

    def test_sizes_are_the_documented_eleven_ratios_and_pixels(self) -> None:
        spec = mr.EVOLINK_MODELS["doubao-seedream-5.5"]
        values = [value for value, _ in spec.size_options]
        assert "auto" in values
        assert {
            "1:1", "2:3", "3:2", "3:4", "4:3", "4:5",
            "5:4", "9:16", "16:9", "21:9", "9:21",
        } <= set(values)
        assert len([value for value in values if ":" in value]) == 11
        assert "1024x1024" in values

    def test_quality_is_resolution_tiers_defaulting_to_1k(self) -> None:
        spec = mr.EVOLINK_MODELS["doubao-seedream-5.5"]
        assert [value for value, _ in spec.quality_options] == ["1K", "1.5K", "2K"]
        assert spec.quality_default == "1K"
        assert spec.resolution_options is None

    def test_one_image_ten_references_and_png_jpeg_only(self) -> None:
        spec = mr.EVOLINK_MODELS["doubao-seedream-5.5"]
        assert spec.max_variations == 1
        assert spec.max_reference_images == 10
        assert spec.supports_output_format is True
        assert spec.output_format_options == (("png", "PNG"), ("jpeg", "JPEG"))


class TestMiniMaxH3Image:
    def test_model_id_and_label(self) -> None:
        spec = mr.EVOLINK_MODELS["minimax-h3-image"]
        assert spec.model_id == "minimax-h3-image"
        assert spec.label == "MiniMax H3 Image"

    def test_sizes_cover_the_fifteen_ratios_and_pixels(self) -> None:
        spec = mr.EVOLINK_MODELS["minimax-h3-image"]
        values = [value for value, _ in spec.size_options]
        assert "auto" in values
        assert {"1:1", "16:9", "21:9", "1:3", "3:1"} <= set(values)
        assert len([value for value in values if ":" in value]) == 15
        assert "1024x1024" in values

    def test_has_no_quality_parameter(self) -> None:
        assert mr.EVOLINK_MODELS["minimax-h3-image"].quality_options is None

    def test_resolution_is_the_separate_1k_2k_4k_tier_defaulting_to_2k(self) -> None:
        spec = mr.EVOLINK_MODELS["minimax-h3-image"]
        assert [value for value, _ in spec.resolution_options] == ["1K", "2K", "4K"]
        assert spec.resolution_default == "2K"

    def test_variations_references_and_all_three_formats(self) -> None:
        spec = mr.EVOLINK_MODELS["minimax-h3-image"]
        assert spec.max_variations == 10
        assert spec.max_reference_images == 16
        assert spec.supports_output_format is True
        assert spec.output_format_options == (
            ("png", "PNG"), ("jpeg", "JPEG"), ("webp", "WEBP"),
        )


class TestBuildModelPrompt:
    def test_plain_models_get_the_prompt_verbatim(self) -> None:
        spec = mr.EVOLINK_MODELS["gemini-nano-banana-2.1"]
        assert mr.build_model_prompt(spec, "a quiet harbor", ["https://x/1.png"]) == "a quiet harbor"

    def test_midjourney_prepends_reference_urls(self) -> None:
        spec = mr.EVOLINK_MODELS["mj-v8.2"]
        prompt = mr.build_model_prompt(spec, "a sunset in watercolor --iw 1.5", ["https://x/1.png"])
        assert prompt == "https://x/1.png a sunset in watercolor --iw 1.5"

    def test_midjourney_without_references_keeps_the_prompt(self) -> None:
        spec = mr.EVOLINK_MODELS["mj-v8.2"]
        assert mr.build_model_prompt(spec, "a sunset", ()) == "a sunset"


class TestAspectPromptFlag:
    def test_auto_yields_no_flag(self) -> None:
        assert mr.aspect_prompt_flag("auto") == ""
        assert mr.aspect_prompt_flag(None) == ""

    def test_ratio_yields_the_ar_flag(self) -> None:
        assert mr.aspect_prompt_flag("16:9") == "--ar 16:9"
        assert mr.aspect_prompt_flag("2:3") == "--ar 2:3"
