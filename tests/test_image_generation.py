"""Tests for Image Generation logic (handoff image handling, edits-endpoint inputs)."""

from __future__ import annotations

from types import SimpleNamespace

import image_generation as ig


class TestCollectInputImages:
    def test_no_inputs_yields_empty_list(self) -> None:
        assert ig.collect_input_images(None, None) == []

    def test_handoff_image_becomes_first_named_buffer(self) -> None:
        handoff = {"bytes": b"\x89PNG handoff", "mime": "image/png", "name": "source.png"}
        buffers = ig.collect_input_images(None, handoff)
        assert len(buffers) == 1
        assert buffers[0].getvalue() == b"\x89PNG handoff"
        assert buffers[0].name == "source.png"

    def test_uploaded_references_follow_the_handoff_image(self) -> None:
        upload = SimpleNamespace(getvalue=lambda: b"upload-bytes", name="upload.jpg")
        handoff = {"bytes": b"handoff-bytes", "mime": "image/png", "name": "handoff.png"}
        buffers = ig.collect_input_images([upload], handoff)
        assert [b.name for b in buffers] == ["handoff.png", "upload.jpg"]
        assert [b.getvalue() for b in buffers] == [b"handoff-bytes", b"upload-bytes"]

    def test_multiple_uploads_keep_their_order(self) -> None:
        first = SimpleNamespace(getvalue=lambda: b"one", name="one.png")
        second = SimpleNamespace(getvalue=lambda: b"two", name="two.png")
        buffers = ig.collect_input_images([first, second], None)
        assert [b.name for b in buffers] == ["one.png", "two.png"]
