"""Observability, Langfuse integration, and execution logging."""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Callable
from typing import Any

from self_rag.config import Settings, get_settings

logger = logging.getLogger("self_rag.execution")


class ObservabilityManager:
    """Manages Langfuse tracing callbacks and client lifecycle."""

    def __init__(self, settings: Settings | None = None) -> None:
        active_settings = settings or get_settings()
        self.enabled = active_settings.langfuse_enabled
        self._client: Any = None
        self._handler: Any = None

        if not self.enabled:
            return

        pk = active_settings.langfuse_public_key
        sk = active_settings.langfuse_secret_key
        if (
            not pk
            or not sk
            or not pk.get_secret_value().strip()
            or not sk.get_secret_value().strip()
        ):
            self.enabled = False
            return

        try:
            from langfuse import Langfuse
            from langfuse.langchain import CallbackHandler

            self._client = Langfuse(
                public_key=pk.get_secret_value(),
                secret_key=sk.get_secret_value(),
                host=active_settings.langfuse_base_url,
            )
            if self._client.auth_check():
                self._handler = CallbackHandler()
            else:
                self.enabled = False
        except Exception as e:
            logger.warning("Failed to initialize Langfuse: %s", e)
            self.enabled = False

    def get_callbacks(self) -> list[Any]:
        """Return callback handlers for LangChain/LangGraph runnables."""
        return [self._handler] if self.enabled and self._handler is not None else []

    def flush(self) -> None:
        """Flush pending events to the Langfuse server."""
        if self._client is not None:
            with contextlib.suppress(Exception):
                self._client.flush()


class ExecutionLogger:
    """Provides structured execution logging for graph node events."""

    def __init__(
        self,
        enabled: bool = False,
        max_chars: int = 1200,
    ) -> None:
        self.enabled = enabled
        self.max_chars = max_chars

    def log_node_start(self, node_name: str, input_preview: str = "") -> None:
        if not self.enabled:
            return
        preview = f" | {input_preview[: self.max_chars]}" if input_preview else ""
        logger.info("[NODE START] %s%s", node_name, preview)

    def log_node_end(self, node_name: str, duration_sec: float, output_preview: str = "") -> None:
        if not self.enabled:
            return
        preview = f" -> {output_preview[: self.max_chars]}" if output_preview else ""
        logger.info("[NODE END] %s (%.3fs)%s", node_name, duration_sec, preview)

    def wrap_node(self, node_name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap a node callable with timing and before/after logging."""
        if not self.enabled:
            return fn

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            self.log_node_start(node_name)
            result = fn(*args, **kwargs)
            duration = time.perf_counter() - start_time
            self.log_node_end(node_name, duration, output_preview=str(result))
            return result

        return wrapped
