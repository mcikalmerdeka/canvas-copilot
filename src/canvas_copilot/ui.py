"""Shared presentation helpers for the Canvas Copilot shell.

The app intentionally keeps its visual vocabulary in one small module: a page
header, a badge row, section labels, and an empty state. Mode-specific views use
these helpers so the two workspaces feel like one product as the UI evolves.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import streamlit as st

BadgeColor = Literal[
    "red", "orange", "yellow", "blue", "green", "violet", "gray", "grey", "primary"
]
Badge = tuple[str, str | None, BadgeColor]


def page_header(
    title: str,
    caption: str,
    badges: Sequence[Badge] = (),
) -> None:
    """Render the standard page title, one-line caption, and compact badges."""
    st.title(title)
    st.caption(caption)
    if badges:
        with st.container(horizontal=True):
            for label, icon, color in badges:
                st.badge(label, icon=icon, color=color)


def section(title: str, caption: str = "") -> None:
    """A small section label that sits inside or above a card."""
    st.markdown(f"**{title}**")
    if caption:
        st.caption(caption)


def empty_state(title: str, body: str) -> None:
    """A bordered placeholder for a workspace that has no content yet."""
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(body)
