"""Top-level entry point to launch the Self-Correcting RAG Gradio web application."""

from __future__ import annotations

import os

from dotenv import load_dotenv

from self_rag.config import get_settings
from self_rag.ui.app import create_app
from self_rag.ui.css import CUSTOM_CSS


def main() -> None:
    """Load configuration and launch the Gradio web interface."""
    load_dotenv()
    settings = get_settings()

    host = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    share = os.environ.get("GRADIO_SHARE", "false").lower() in ("true", "1", "yes")

    app = create_app(settings=settings)
    app.launch(
        server_name=host,
        server_port=port,
        share=share,
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
