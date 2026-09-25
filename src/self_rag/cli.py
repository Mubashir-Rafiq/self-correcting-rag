"""Command-line entry point.

Each build stage adds its own subcommand here, so there is always one place to exercise whatever
has been built so far.
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Sequence
from pathlib import Path

from self_rag.config import Settings, get_settings


def _print_config(settings: Settings) -> None:
    fields = settings.model_dump()
    width = max(len(name) for name in fields)
    print("configuration")
    for name in sorted(fields):
        print(f"  {name:<{width}}  {fields[name]!r}")

    derived = {
        "markdown_dir": settings.markdown_dir,
        "parent_store_dir": settings.parent_store_dir,
        "qdrant_path": settings.qdrant_path,
        "checkpoint_db_path": settings.checkpoint_db_path,
    }
    print("\nderived paths")
    for name, value in derived.items():
        print(f"  {name:<{width}}  {value}")

    key = settings.api_key_for_provider()
    status = "present" if key and key.get_secret_value().strip() else "MISSING"
    print(f"\ncredentials\n  api key for {settings.llm_provider}: {status}")


def _chunk_file(file_path: Path, settings: Settings) -> None:
    """Convert *file_path* if needed, then chunk it and print a summary."""
    from self_rag.ingestion.chunker import DocumentChunker
    from self_rag.ingestion.converters import convert_file, file_sha256

    print(f"source       {file_path}")
    print(f"sha-256      {file_sha256(file_path)}")

    md_path = convert_file(file_path, settings.markdown_dir)
    print(f"markdown     {md_path}")

    chunker = DocumentChunker(settings)
    parents, children = chunker.chunk_file(md_path)

    print(f"\nparents      {len(parents)}")
    print(f"children     {len(children)}")

    # Parent size distribution
    parent_sizes = [len(doc.page_content) for _, doc in parents]
    child_sizes = [len(doc.page_content) for doc in children]

    for label, sizes in [("parent", parent_sizes), ("child", child_sizes)]:
        if not sizes:
            continue
        print(f"\n{label} size distribution (chars)")
        print(f"  min        {min(sizes)}")
        print(f"  max        {max(sizes)}")
        print(f"  mean       {statistics.mean(sizes):.0f}")
        if len(sizes) >= 2:
            print(f"  median     {statistics.median(sizes):.0f}")
            print(f"  stdev      {statistics.stdev(sizes):.0f}")

    # Quick histogram — 5 buckets between min and max
    if parent_sizes and len(parent_sizes) > 1:
        _print_histogram("parent", parent_sizes)
    if child_sizes and len(child_sizes) > 1:
        _print_histogram("child", child_sizes)


def _print_histogram(label: str, sizes: list[int], buckets: int = 5) -> None:
    """Print a simple text histogram of *sizes*."""
    lo, hi = min(sizes), max(sizes)
    if lo == hi:
        print(f"\n{label} histogram: all chunks are {lo} chars")
        return

    step = (hi - lo) / buckets
    counts = [0] * buckets
    for s in sizes:
        idx = min(int((s - lo) / step), buckets - 1)
        counts[idx] += 1

    print(f"\n{label} histogram")
    max_count = max(counts)
    bar_width = 30
    for i in range(buckets):
        lower = lo + i * step
        upper = lo + (i + 1) * step
        bar_len = int(counts[i] / max_count * bar_width) if max_count else 0
        bar = "█" * bar_len
        print(f"  {lower:>6.0f}-{upper:<6.0f}  {bar}  {counts[i]}")


def _search_query(query: str, limit: int | None, settings: Settings) -> None:
    """Run a hybrid search across indexed child chunks and print ranked excerpts."""
    from self_rag.storage.embeddings import get_dense_embeddings, get_sparse_embeddings
    from self_rag.storage.vector_store import VectorStoreManager

    with VectorStoreManager(settings=settings) as vm:
        if not vm.collection_exists() or vm.count() == 0:
            print("error: collection is empty or does not exist. Ingest documents first.")
            return

        k = limit or settings.retrieval_k
        dense = get_dense_embeddings(settings)
        sparse = get_sparse_embeddings(settings)
        store = vm.get_vector_store(embedding=dense, sparse_embedding=sparse)

        results = store.similarity_search_with_score(query, k=k)
        if not results:
            print(f"no results found for query: {query!r}")
            return

        print(f"query: {query!r}")
        print(f"found {len(results)} results (limit: {k})\n")

        for rank, (doc, score) in enumerate(results, start=1):
            source = doc.metadata.get("source", "unknown")
            parent_id = doc.metadata.get("parent_id", "-")
            headers = [
                f"{h}={doc.metadata[h]}" for _, h in settings.markdown_headers if h in doc.metadata
            ]
            header_str = f" [{', '.join(headers)}]" if headers else ""

            print(f"[{rank}] score={score:.4f}  source={source}  parent={parent_id}{header_str}")
            content = doc.page_content.strip().replace("\n", " ")
            if len(content) > 160:
                content = content[:157] + "..."
            print(f"    {content}\n")


def _ingest_file(file_path: Path, force: bool, settings: Settings) -> int:
    """Ingest a file using IngestionPipeline."""
    from self_rag.ingestion.pipeline import IngestionPipeline

    if not file_path.exists():
        print(f"error: file not found: {file_path}")
        return 1

    pipeline = IngestionPipeline(settings=settings)
    try:
        result = pipeline.ingest_file(file_path, force=force)
    finally:
        pipeline.vector_store_manager.close()

    if result.status == "skipped":
        print(f"skipped: {file_path.name} is unchanged (use --force to re-index)")
        return 0
    elif result.status == "added":
        print(f"ingested: {file_path.name}")
        print(f"  parents:  {result.parent_count}")
        print(f"  children: {result.child_count}")
        return 0
    else:
        print(f"error: failed to ingest {file_path.name}: {result.error}")
        return 1


def _list_documents(settings: Settings) -> None:
    """List indexed documents."""
    from self_rag.ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline(settings=settings)
    try:
        docs = pipeline.list_documents()
    finally:
        pipeline.vector_store_manager.close()

    if not docs:
        print("No documents indexed. Ingest documents with 'self-rag ingest <file>'.")
        return

    print(f"Indexed documents ({len(docs)} total):\n")
    print(f"  {'SOURCE':<30}  {'PARENTS':<8}  {'SHA-256':<12}")
    print(f"  {'-' * 30}  {'-' * 8}  {'-' * 12}")
    for d in docs:
        sha_short = (d["sha256"][:10] + "..") if d["sha256"] else "-"
        print(f"  {d['source']:<30}  {d['parents']:<8}  {sha_short:<12}")


def _check_llm(settings: Settings) -> int:
    """Verify LLM connectivity and structured output capabilities."""
    from self_rag.agent.schemas import QueryAnalysis
    from self_rag.llm.factory import MissingApiKeyError, build_llm

    print("checking LLM configuration...")
    print(f"  provider:    {settings.llm_provider}")
    print(f"  model:       {settings.llm_model}")
    print(f"  temperature: {settings.llm_temperature}")
    print(f"  retries:     {settings.llm_max_retries}")
    print(f"  rate limit:  {settings.llm_requests_per_second} req/s\n")

    try:
        llm = build_llm(settings)
    except MissingApiKeyError as err:
        print(f"error: {err}")
        return 1
    except Exception as err:
        print(f"error initializing model: {err}")
        return 1

    print("sending test prompt...")
    try:
        response = llm.invoke("Respond with 'LLM connection successful' and nothing else.")
        content = response.content
        if isinstance(content, list):
            parts = [
                str(item.get("text", item)) if isinstance(item, dict) else str(item)
                for item in content
            ]
            content = " ".join(parts)
        print(f"  response:    {str(content).strip()}")
    except Exception as err:
        print(f"error during basic generation: {err}")
        return 1

    import time

    time.sleep(1.0)
    print("\ntesting structured output (QueryAnalysis)...")
    try:
        structured_llm = llm.with_structured_output(QueryAnalysis)
        analysis = structured_llm.invoke("What is self-correcting RAG?")
        if not isinstance(analysis, QueryAnalysis):
            print(f"error: expected QueryAnalysis instance, got {type(analysis)}")
            return 1
        print(f"  is_clear:    {analysis.is_clear}")
        print(f"  questions:   {analysis.questions}")
        print(f"  clarify:     {analysis.clarification_needed!r}")
    except Exception as err:
        print(f"error during structured output check: {err}")
        return 1

    print("\nstatus: OK — LLM layer is functional.")
    return 0


def _ask_question(question: str, settings: Settings) -> int:
    """Execute the agent research loop on *question* and print the response."""
    from self_rag.agent.graph import ask_question
    from self_rag.llm.factory import MissingApiKeyError

    clean_question = question.strip()
    if not clean_question:
        print("error: question cannot be empty")
        return 1

    try:
        result = ask_question(clean_question, settings=settings)
        answer = result.get("final_answer", "")
        if not answer:
            print("Unable to generate an answer.")
            return 1
        print(answer)
        return 0
    except MissingApiKeyError as err:
        print(f"error: {err}")
        return 1
    except ValueError as err:
        print(f"error: {err}")
        return 1
    except Exception as err:
        print(f"error executing research loop: {err}")
        return 1


def _run_chat(thread_id: str | None, settings: Settings) -> int:
    """Run an interactive terminal chat session with conversation memory."""
    from self_rag.llm.factory import MissingApiKeyError
    from self_rag.system import RAGSystem

    try:
        with RAGSystem(settings=settings) as system:
            session_id = thread_id or system.create_thread_id()
            print("=" * 60)
            print("  Interactive Multi-Turn RAG Chat")
            print(f"  Thread ID: {session_id}")
            print("  Type '/exit' to quit, '/new' for a fresh session.")
            print("=" * 60)

            while True:
                try:
                    user_input = input("\nYou: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nExiting chat.")
                    break

                if not user_input:
                    continue
                if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
                    print("Goodbye!")
                    break
                if user_input.lower() == "/new":
                    session_id = system.create_thread_id()
                    print(f"Started new session: {session_id}")
                    continue

                res = system.chat(user_input, thread_id=session_id)
                if res["is_clarification"]:
                    print(f"\nAssistant (Clarification): {res['answer']}")
                else:
                    print(f"\nAssistant: {res['answer']}")
        return 0
    except MissingApiKeyError as err:
        print(f"error: {err}")
        return 1
    except ValueError as err:
        print(f"error: {err}")
        return 1
    except Exception as err:
        print(f"error in chat session: {err}")
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="self-rag",
        description="An agentic, self-correcting RAG system.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("config", help="Print the resolved configuration and exit.")

    chunk_parser = subcommands.add_parser(
        "chunk",
        help="Convert and chunk a document, showing size distribution.",
    )
    chunk_parser.add_argument(
        "file",
        type=Path,
        help="Path to a PDF or Markdown file to chunk.",
    )

    ingest_parser = subcommands.add_parser(
        "ingest",
        help="Convert, chunk, and index a document into storage.",
    )
    ingest_parser.add_argument(
        "file",
        type=Path,
        help="Path to a PDF or Markdown file to ingest.",
    )
    ingest_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force re-indexing even if file is unchanged.",
    )

    subcommands.add_parser("list", help="List all indexed documents.")

    search_parser = subcommands.add_parser(
        "search",
        help="Run hybrid search for a query and print ranked excerpts.",
    )
    search_parser.add_argument(
        "query",
        type=str,
        help="The search query text.",
    )
    search_parser.add_argument(
        "-k",
        "--limit",
        type=int,
        default=None,
        help="Number of results to retrieve (defaults to settings.retrieval_k).",
    )

    subcommands.add_parser(
        "llm-check",
        help="Verify connection and structured output for the configured LLM.",
    )

    ask_parser = subcommands.add_parser(
        "ask",
        help="Ask a question and receive a grounded, cited answer.",
    )
    ask_parser.add_argument(
        "question",
        type=str,
        help="The question to answer using indexed documents.",
    )

    chat_parser = subcommands.add_parser(
        "chat",
        help="Start an interactive multi-turn chat session.",
    )
    chat_parser.add_argument(
        "--thread-id",
        type=str,
        default=None,
        help="Continue an existing conversation thread ID.",
    )

    ui_parser = subcommands.add_parser(
        "ui",
        help="Launch the Gradio web application.",
    )
    ui_parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind the web server (default: 127.0.0.1).",
    )
    ui_parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to bind the web server (default: 7860).",
    )
    ui_parser.add_argument(
        "--share",
        action="store_true",
        help="Generate a publicly shareable Gradio link.",
    )

    eval_parser = subcommands.add_parser(
        "eval",
        help="Run offline evaluation metrics on a dataset file (JSON or JSONL).",
    )
    eval_parser.add_argument(
        "dataset",
        type=Path,
        help="Path to evaluation dataset file (.json or .jsonl).",
    )
    eval_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Optional path to write full JSON evaluation report.",
    )
    return parser


def _run_ui(host: str, port: int, share: bool, settings: Settings) -> int:
    """Launch the Gradio web application."""
    from self_rag.ui.app import create_app
    from self_rag.ui.css import CUSTOM_CSS

    print(f"Starting Self-Correcting RAG UI at http://{host}:{port}...")
    app = create_app(settings=settings)
    app.launch(server_name=host, server_port=port, share=share, css=CUSTOM_CSS)
    return 0


def _run_eval(dataset_path: Path, output_path: Path | None, settings: Settings) -> int:
    """Run offline evaluation against an evaluation dataset."""
    from self_rag.evaluation import RAGEvaluator

    if not dataset_path.exists():
        print(f"error: evaluation dataset not found: {dataset_path}")
        return 1

    print(f"Loading evaluation dataset from {dataset_path}...")
    samples = RAGEvaluator.load_dataset_from_file(dataset_path)
    print(f"Loaded {len(samples)} evaluation sample(s). Evaluating...")

    evaluator = RAGEvaluator(settings=settings)
    report = evaluator.evaluate_dataset(samples)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total Samples Evaluated: {len(report.results)}")
    print(f"Mean Faithfulness:       {report.mean_faithfulness:.4f}")
    print(f"Mean Answer Relevance:   {report.mean_answer_relevance:.4f}")
    print(f"Mean Context Precision:  {report.mean_context_precision:.4f}")
    print("=" * 60)

    if output_path is not None:
        output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        print(f"Detailed evaluation report written to {output_path}")

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()

    if args.command == "config":
        _print_config(settings)
    elif args.command == "chunk":
        file_path: Path = args.file
        if not file_path.exists():
            print(f"error: file not found: {file_path}")
            return 1
        _chunk_file(file_path, settings)
    elif args.command == "ingest":
        return _ingest_file(args.file, args.force, settings)
    elif args.command == "list":
        _list_documents(settings)
    elif args.command == "search":
        _search_query(args.query, args.limit, settings)
    elif args.command == "llm-check":
        return _check_llm(settings)
    elif args.command == "ask":
        return _ask_question(args.question, settings)
    elif args.command == "chat":
        return _run_chat(args.thread_id, settings)
    elif args.command == "ui":
        return _run_ui(args.host, args.port, args.share, settings)
    elif args.command == "eval":
        return _run_eval(args.dataset, args.output, settings)

    return 0
