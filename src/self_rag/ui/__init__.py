"""Gradio web application interface for Self-Correcting RAG."""

from __future__ import annotations

from self_rag.ui.app import create_app
from self_rag.ui.css import CUSTOM_CSS

__all__ = ["CUSTOM_CSS", "create_app"]
