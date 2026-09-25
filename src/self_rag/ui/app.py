"""Gradio application assembling Chat, Documents, and System Status tabs."""

from __future__ import annotations

import logging
from collections.abc import Generator
from pathlib import Path
from typing import Any, cast

import gradio as gr

from self_rag.config import Settings, get_settings
from self_rag.ingestion.pipeline import IngestionPipeline, IngestionResult
from self_rag.system import RAGSystem

logger = logging.getLogger(__name__)


def format_sources_markdown(pipeline: IngestionPipeline) -> str:
    """Format indexed sources into a clean markdown table."""
    try:
        sources = pipeline.list_documents()
    except Exception as err:
        logger.warning("Could not read indexed sources: %s", err)
        return f"⚠️ Error reading indexed sources: {err}"

    if not sources:
        return "*No documents indexed yet. Upload documents above to begin.*"

    lines = [
        "| Source Document | Parent Chunks | Status |",
        "| :--- | :---: | :---: |",
    ]
    for s in sources:
        src = s.get("source", "unknown")
        parents = s.get("parents", 0)
        lines.append(f"| `{src}` | **{parents}** | ✅ Indexed |")

    return "\n".join(lines)


def format_system_status_markdown(system: RAGSystem) -> str:
    """Format system runtime status and storage statistics."""
    cfg = system.settings
    try:
        if system.vector_store_manager.collection_exists():
            vec_count = system.vector_store_manager.count()
        else:
            vec_count = 0
    except Exception:
        vec_count = 0

    try:
        parent_count = len(system.parent_store.list_parent_ids())
    except Exception:
        parent_count = 0

    correction = (
        f"✅ Enabled (max {cfg.max_correction_retries} retries)"
        if cfg.self_correction_enabled
        else "⚪ Disabled"
    )
    observability = "✅ Langfuse Enabled" if cfg.langfuse_enabled else "⚪ Local Execution Logging"

    lines = [
        "### 🛡️ Runtime Configuration & Storage Metrics",
        "",
        "| Component | Active Setting |",
        "| :--- | :--- |",
        f"| **LLM Provider** | `{cfg.llm_provider}` |",
        f"| **Model** | `{cfg.llm_model}` |",
        f"| **Dense Embeddings** | `{cfg.dense_model}` ({cfg.dense_dimension}d) |",
        f"| **Sparse Embeddings** | `{cfg.sparse_model}` (BM25) |",
        f"| **Child Vectors** | `{cfg.child_collection}` ({vec_count} vectors) |",
        f"| **Parent Chunks** | `{parent_count}` parent documents |",
        f"| **Self-Correction** | {correction} |",
        f"| **Observability** | {observability} |",
        f"| **Storage Root** | `{cfg.data_dir}` |",
    ]
    return "\n".join(lines)


def chat_stream_handler(
    message: str,
    history: list[dict[str, Any]],
    thread_id: str | None,
    system: RAGSystem,
) -> Generator[str, None, None]:
    """Handle chat interaction by streaming responses from RAGSystem."""
    tid = thread_id or system.create_thread_id()
    yield from system.chat_stream(message, tid)


def create_app(
    rag_system: RAGSystem | None = None,
    pipeline: IngestionPipeline | None = None,
    settings: Settings | None = None,
) -> gr.Blocks:
    """Build and return the Gradio interface for Self-Correcting RAG.

    Provides:
    - **Chat Tab**: Multi-turn conversation with memory, clarification, and per-session isolation.
    - **Documents Tab**: Multi-file document upload, ingestion status, source list, and clearing.
    - **Status Tab**: Runtime configuration, storage metrics, and active model providers.
    """
    active_settings = settings or get_settings()
    system = rag_system or RAGSystem(settings=active_settings)
    ingest_pipeline = pipeline or IngestionPipeline(
        settings=active_settings,
        parent_store=system.parent_store,
        vector_store_manager=system.vector_store_manager,
    )

    with gr.Blocks(title="Self-Correcting RAG") as demo:
        gr.Markdown(
            """
<div class="app-header">
    <h1 class="app-title">🛡️ Self-Correcting RAG</h1>
    <p class="app-subtitle">
        Hierarchical Parent/Child Chunking • Hybrid Vector Search •
        Autonomous Research Loop &amp; Self-Correction
    </p>
</div>
"""
        )

        with gr.Tabs(elem_classes=["tabs"]):
            # ======================== CHAT TAB ========================
            with gr.Tab("💬 Conversational Chat"):
                session_thread_id = gr.State()

                with gr.Row():
                    thread_display = gr.Markdown(
                        value="*Session thread initializing...*",
                        elem_classes=["thread-badge"],
                    )

                chatbot = gr.Chatbot(
                    height=580,
                    elem_classes=["chatbot-container"],
                )

                def chat_handler(
                    message: str,
                    history: list[dict[str, Any]],
                    thread_id: str | None,
                ) -> Generator[str, None, None]:
                    yield from chat_stream_handler(message, history, thread_id, system)

                def handle_clear_chat() -> tuple[str, str]:
                    new_id = system.create_thread_id()
                    badge = f"Thread: `{new_id[:8]}...`"
                    return new_id, badge

                gr.ChatInterface(
                    fn=chat_handler,
                    chatbot=chatbot,
                    additional_inputs=[session_thread_id],
                )

                chatbot.clear(
                    fn=handle_clear_chat,
                    outputs=[session_thread_id, thread_display],
                )

            # ===================== DOCUMENTS TAB =====================
            with gr.Tab("📄 Document Ingestion"):
                gr.Markdown("### Manage Ingested Knowledge Base")
                gr.Markdown(
                    "Upload PDF or Markdown documents. Files are automatically parsed, converted, "
                    "split into hierarchical parent/child chunks, and indexed into Qdrant."
                )

                file_upload = gr.File(
                    file_count="multiple",
                    file_types=[".pdf", ".md"],
                    label="Select PDF or Markdown Files",
                )

                with gr.Row():
                    add_btn = gr.Button(
                        "📥 Ingest Uploaded Documents",
                        variant="primary",
                        elem_classes=["primary-btn"],
                    )
                    refresh_btn = gr.Button("🔄 Refresh Sources", variant="secondary")
                    clear_btn = gr.Button("🗑️ Clear All Indexed Data", variant="stop")

                ingest_status = gr.Markdown(value="*Status: Ready.*")

                gr.Markdown("### Currently Indexed Documents")
                sources_box = gr.Markdown(
                    value=format_sources_markdown(ingest_pipeline),
                    elem_classes=["sources-box"],
                )

                def handle_add_documents(
                    files: list[Any] | None,
                ) -> tuple[str, str]:
                    if not files:
                        return "⚠️ No files selected. Please select files first.", (
                            format_sources_markdown(ingest_pipeline)
                        )

                    results: list[IngestionResult] = []
                    for f in files:
                        raw_path = getattr(f, "name", str(f))
                        res = ingest_pipeline.ingest_file(Path(raw_path))
                        results.append(res)

                    added = sum(1 for r in results if r.status == "added")
                    skipped = sum(1 for r in results if r.status == "skipped")
                    failed = sum(1 for r in results if r.status == "failed")
                    parents = sum(r.parent_count for r in results)
                    children = sum(r.child_count for r in results)

                    msg_parts = [
                        f"✅ **Ingestion completed:** {added} added ({parents} parents, "
                        f"{children} child vectors), {skipped} unchanged, {failed} failed."
                    ]
                    for r in results:
                        if r.status == "failed":
                            msg_parts.append(f"- ❌ `{r.source_path.name}`: {r.error}")

                    return "\n".join(msg_parts), format_sources_markdown(ingest_pipeline)

                def handle_refresh_sources() -> str:
                    return format_sources_markdown(ingest_pipeline)

                def handle_clear_all() -> tuple[str, str]:
                    ingest_pipeline.clear_all()
                    return (
                        "🗑️ All document vectors, parent stores, and cached markdown cleared.",
                        format_sources_markdown(ingest_pipeline),
                    )

                add_btn.click(
                    fn=handle_add_documents,
                    inputs=[file_upload],
                    outputs=[ingest_status, sources_box],
                )
                refresh_btn.click(
                    fn=handle_refresh_sources,
                    outputs=[sources_box],
                )
                clear_btn.click(
                    fn=handle_clear_all,
                    outputs=[ingest_status, sources_box],
                )

            # ====================== STATUS TAB =======================
            with gr.Tab("⚙️ System Status"):
                status_box = gr.Markdown(value=format_system_status_markdown(system))
                refresh_status_btn = gr.Button("🔄 Refresh Status", variant="secondary")

                def handle_refresh_status() -> str:
                    return format_system_status_markdown(system)

                refresh_status_btn.click(
                    fn=handle_refresh_status,
                    outputs=[status_box],
                )

        # Initialize session thread on page load
        def init_session() -> tuple[str, str]:
            tid = system.create_thread_id()
            return tid, f"Thread: `{tid[:8]}...`"

        demo.load(fn=init_session, outputs=[session_thread_id, thread_display])

    return cast(gr.Blocks, demo)
