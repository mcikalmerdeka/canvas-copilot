"""Tests for the guided prompt-builder catalog and brief composition.

The catalog is pure data plus one deterministic composition function, so these
tests pin the UI's option vocabulary without touching Streamlit or the network.
"""

from __future__ import annotations

import random

import pytest

from canvas_copilot.modes import prompt_catalog as pc


class TestCatalogIntegrity:
    def test_subject_types_are_non_empty_and_unique(self) -> None:
        assert pc.SUBJECT_TYPES
        assert len(set(pc.SUBJECT_TYPES)) == len(pc.SUBJECT_TYPES)

    def test_groups_are_non_empty_and_unique(self) -> None:
        group_ids = [group_id for group_id, _ in pc.GROUPS]
        assert group_ids
        assert len(set(group_ids)) == len(group_ids)

    def test_facet_ids_are_unique(self) -> None:
        ids = [facet.id for facet in pc.FACETS]
        assert len(set(ids)) == len(ids)
        assert pc.FACET_BY_ID.keys() == set(ids)

    def test_every_facet_belongs_to_a_known_group(self) -> None:
        groups = {group_id for group_id, _ in pc.GROUPS}
        assert {facet.group for facet in pc.FACETS} <= groups

    def test_every_select_facet_has_options_with_prose(self) -> None:
        for facet in pc.FACETS:
            if facet.kind == "text":
                assert not facet.options
                continue
            assert facet.options, f"{facet.id} has no options"
            labels = [option.label for option in facet.options]
            assert len(set(labels)) == len(labels)
            for option in facet.options:
                assert option.label.strip()
                assert option.text.strip()
                assert len(option.text) > len(option.label), (
                    f"{facet.id}/{option.label} should expand into a richer description"
                )

    def test_character_facets_are_limited_to_people(self) -> None:
        character = [facet for facet in pc.FACETS if facet.group == "character"]
        assert character
        for facet in character:
            assert set(facet.applies_to) == {"Person", "Group"}

    def test_facets_for_hides_character_for_non_people(self) -> None:
        person_ids = {facet.id for facet in pc.facets_for("Person")}
        product_ids = {facet.id for facet in pc.facets_for("Product")}
        assert "character_outfit" in person_ids
        assert "character_outfit" not in product_ids
        assert "setting" in product_ids


class TestOptionText:
    def test_maps_a_label_to_its_prose(self) -> None:
        prose = pc.option_text("character_outfit", "Streetwear")
        assert "hoodie" in prose.lower()

    def test_returns_empty_for_default_and_unknown_labels(self) -> None:
        assert pc.option_text("character_outfit", pc.DEFAULT_LABEL) == ""
        assert pc.option_text("character_outfit", "No such outfit") == ""


class TestBuildBrief:
    def test_contains_subject_and_selected_prose(self) -> None:
        brief = pc.build_brief(
            "A woman in her early 30s.",
            "Person",
            {
                "character_age": "30s",
                "character_outfit": "Tailored editorial",
                "setting": "Photo studio",
                "lighting": "Single softbox",
                "camera_angle": "Eye level",
                "medium_style": "Editorial photo",
            },
        )
        assert "SUBJECT" in brief
        assert "A woman in her early 30s." in brief
        assert pc.option_text("character_outfit", "Tailored editorial") in brief
        assert pc.option_text("setting", "Photo studio") in brief
        assert pc.option_text("lighting", "Single softbox") in brief
        assert "CHARACTER" in brief
        assert "SCENE" in brief
        assert "CAMERA" in brief
        assert "STYLE" in brief

    def test_omits_unselected_facets_and_empty_groups(self) -> None:
        brief = pc.build_brief("A lone chapel on a ridge.", "Place", {})
        assert "A lone chapel on a ridge." in brief
        assert "CHARACTER" not in brief
        assert "SCENE" not in brief
        assert "Avoid:" not in brief

    def test_joins_multiselect_prose(self) -> None:
        brief = pc.build_brief(
            "A studio portrait.",
            "Person",
            {
                "negative_constraints": [
                    "No text or watermarks",
                    "No plastic skin",
                ]
            },
        )
        assert pc.option_text("negative_constraints", "No text or watermarks") in brief
        assert pc.option_text("negative_constraints", "No plastic skin") in brief
        assert ";" in brief.split("Avoid:")[1]

    def test_includes_free_text_facets_and_notes(self) -> None:
        brief = pc.build_brief(
            "A workshop bench.",
            "Product",
            {
                "props_details": "Wood shavings and a brass ruler.",
                "extra_notes": "Keep the framing tight.",
            },
            notes="Owner preference: warm light.",
        )
        assert "Wood shavings and a brass ruler." in brief
        assert "Keep the framing tight." in brief
        assert "Owner preference: warm light." in brief

    def test_group_subject_keeps_character_facets(self) -> None:
        brief = pc.build_brief(
            "Three friends on a rooftop.",
            "Group",
            {"character_outfit": "Casual denim"},
        )
        assert "CHARACTER" in brief
        assert pc.option_text("character_outfit", "Casual denim") in brief

    def test_bald_head_drops_a_contradictory_hair_colour(self) -> None:
        brief = pc.build_brief(
            "A bald man in a workshop.",
            "Person",
            {"character_hair_style": "Bald", "character_hair_color": "Dark brown"},
        )
        assert "A shaved or bald head" in brief
        assert "Dark brown hair" not in brief


class TestPresets:
    def test_every_preset_uses_valid_facets_and_labels(self) -> None:
        for preset_id, preset in pc.PRESETS.items():
            subject_type = preset["subject_type"]
            assert subject_type in pc.SUBJECT_TYPES, preset_id
            if preset_id != "fresh":
                assert str(preset["subject"]).strip(), preset_id
            values = preset["values"]
            assert isinstance(values, dict)
            available = {facet.id: facet for facet in pc.facets_for(subject_type)}
            for facet_id, value in values.items():
                assert facet_id in available, f"{preset_id}: unknown facet {facet_id}"
                facet = available[facet_id]
                labels = {option.label for option in facet.options}
                if facet.kind == "text":
                    assert isinstance(value, str) and value.strip(), (
                        f"{preset_id}: empty text facet {facet_id}"
                    )
                elif isinstance(value, str):
                    assert value in labels, f"{preset_id}: {facet_id}={value!r}"
                else:
                    assert value, f"{preset_id}: empty multiselect {facet_id}"
                    assert set(value) <= labels, f"{preset_id}: bad {facet_id} values"

    def test_preset_values_returns_a_copy(self) -> None:
        first = pc.preset_values("editorial_portrait")
        first["setting"] = "tampered"
        second = pc.preset_values("editorial_portrait")
        assert second["setting"] != "tampered"

    def test_unknown_preset_raises(self) -> None:
        with pytest.raises(KeyError):
            pc.preset_values("does-not-exist")


class TestRandomize:
    def test_is_deterministic_with_a_seeded_rng(self) -> None:
        first = pc.randomize_selections("Person", random.Random(7))
        second = pc.randomize_selections("Person", random.Random(7))
        assert first == second

    def test_returns_valid_and_non_empty_values(self) -> None:
        values = pc.randomize_selections("Product", random.Random(3))
        facets = {facet.id: facet for facet in pc.facets_for("Product")}
        assert values
        for facet_id, value in values.items():
            assert facet_id in facets
            facet = facets[facet_id]
            labels = {option.label for option in facet.options}
            if isinstance(value, str):
                assert value in labels
                assert value != pc.DEFAULT_LABEL
            else:
                assert value and set(value) <= labels

    def test_person_randomization_includes_character_facets(self) -> None:
        values = pc.randomize_selections("Person", random.Random(1))
        assert any(key.startswith("character_") for key in values)
