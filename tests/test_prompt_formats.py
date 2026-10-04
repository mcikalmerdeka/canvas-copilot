"""Tests for the prompt format registry (prompt_formats.py)."""

from __future__ import annotations

import json
from pathlib import Path

from prompt_formats import (
    PROMPT_FORMATS,
    get_format,
    list_labels,
)

EXPECTED_FORMATS = ("format_1", "format_2", "format_3")


class TestRegistryContents:
    def test_contains_exactly_the_three_documented_formats(self) -> None:
        assert tuple(PROMPT_FORMATS) == EXPECTED_FORMATS

    def test_labels_are_unique_and_human_readable(self) -> None:
        labels = [fmt["label"] for fmt in PROMPT_FORMATS.values()]
        assert len(set(labels)) == len(labels)
        assert all(label.strip() for label in labels)

    def test_every_format_has_exemplar_text_loaded_in_memory(self) -> None:
        for fmt in PROMPT_FORMATS.values():
            assert isinstance(fmt["exemplar_text"], str)
            assert fmt["exemplar_text"].strip(), f"{fmt['id']} exemplar must not be empty"

    def test_every_format_has_exemplar_path_pointing_to_real_file(self, project_root: Path) -> None:
        for fmt in PROMPT_FORMATS.values():
            path = project_root / fmt["exemplar_path"]
            assert path.is_file(), f"missing exemplar file: {fmt['exemplar_path']}"

    def test_in_memory_exemplar_matches_file_content(self, project_root: Path) -> None:
        for fmt in PROMPT_FORMATS.values():
            path = project_root / fmt["exemplar_path"]
            assert path.read_text(encoding="utf-8") == fmt["exemplar_text"]

    def test_is_json_flag_is_true_only_for_format_3(self) -> None:
        assert PROMPT_FORMATS["format_1"]["is_json"] is False
        assert PROMPT_FORMATS["format_2"]["is_json"] is False
        assert PROMPT_FORMATS["format_3"]["is_json"] is True

    def test_format_3_exemplar_is_valid_json_object(self) -> None:
        exemplar = PROMPT_FORMATS["format_3"]["exemplar_text"]
        parsed = json.loads(exemplar)
        assert isinstance(parsed, dict)
        assert "task" in parsed  # top-level key visible in the source file


class TestAccessors:
    def test_get_format_returns_entry_for_known_key(self) -> None:
        assert get_format("format_2")["id"] == "format_2"

    def test_list_labels_returns_all_labels_in_order(self) -> None:
        assert list_labels() == [PROMPT_FORMATS[key]["label"] for key in EXPECTED_FORMATS]
