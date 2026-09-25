# Build Plan & Progress

This is the living record of how `self-correcting-rag` is being built. It is updated at the end of
every stage, so it always answers two questions: **what works today**, and **what is next**.

Every stage ends the same way: a written explanation in [`docs/stages/`](docs/stages/), an update to
this file, then a commit pushed to GitHub. Nothing moves to the next stage until the current one is
understood.

**Legend:** ✅ done · ▶️ in progress · ⬜ not started

---

## Progress at a glance

| # | Stage | Status | What it gives you |
|---|---|---|---|
| 0 | Skeleton, tooling & typed configuration | ✅ | A project that lints, type-checks and tests itself |
| 1 | Document conversion & chunking | ✅ | Turn a PDF into searchable parent/child chunks |
| 2 | Parent store & Qdrant collection | ✅ | Somewhere to put those chunks |
| 3 | Embeddings & hybrid search | ✅ | Actually find the right chunk for a question |
| 4 | Ingestion pipeline | ✅ | One safe, repeatable "add this document" operation |
| 5 | LLM layer, prompts & structured output | ✅ | The first real call to a language model |
| 6 | Agent state & retrieval tools | ✅ | Tools the model can decide to call |
| 7 | Minimal agent loop | ✅ | **The first real answer from your own documents** |
| 8 | Research budget, compression & fallback | ✅ | An agent that can't loop forever or blow its context |
| 9 | Main graph | ✅ | Memory, clarifying questions, parallel sub-questions |
| 10 | Composition root, observability & CLI | ✅ | **A complete app you can chat with** |
| 11 | Self-correction | ✅ | **The namesake:** it grades its own evidence and retries |
| 12 | Gradio UI | ✅ | A browser interface |
| 13 | Production polish | ✅ | Qdrant server, reranking, citations, evaluation |

---

## Where we are now

**All 14 stages (0 through 13) are complete.** The system is production-polished:
1. **Qdrant Server Support**: Production `docker-compose.yml` deployment and configuration-only switch via `SELF_RAG_QDRANT_URL` and `SELF_RAG_QDRANT_API_KEY`.
2. **Cross-Encoder Reranking**: FastEmbed ONNX-based cross-encoder (`FastEmbedReranker`) reranking candidate child chunks prior to agent synthesis without requiring PyTorch.
3. **Page-Level Citations**: Chunking detects and tracks page boundaries and metadata (`page`), propagating through retrieval tools to format citations like `Sources: manual.pdf, p. 14`.
4. **Offline Evaluation & RAGAS Compatibility**: Standalone evaluation engine (`RAGEvaluator`) and CLI command (`self-rag eval`) scoring faithfulness, answer relevance, and context precision, with export to RAGAS dataset format.
5. **Quality Gates**: 244 unit tests passing, strict Mypy type-checking across 56 files, Ruff linting and formatting cleanly enforced.

Run it yourself:

```bash
uv sync --all-extras
uv run self-rag config          # print the resolved configuration
uv run self-rag chunk <file>    # inspect chunking for a PDF or Markdown file
uv run self-rag ingest <file>   # convert, chunk, and index a document
uv run self-rag list            # list all indexed documents
uv run self-rag search "<query>"# hybrid search across indexed chunks
uv run self-rag llm-check       # test LLM connectivity and structured output
uv run self-rag ask "<question>"# grounded, cited answer from documents
uv run self-rag chat            # interactive multi-turn chat in your terminal
uv run pytest                   # 180 tests
```

---

## Stages in detail

### ✅ Stage 0 — Skeleton, tooling & typed configuration

📖 [docs/stages/stage-00-skeleton-and-configuration.md](docs/stages/stage-00-skeleton-and-configuration.md)

- [x] `pyproject.toml` + `uv.lock` — reproducible install, Python pinned to 3.12
- [x] Ruff (lint + format), mypy (`strict`, with the pydantic plugin), pytest
- [x] `pre-commit` hooks and a GitHub Actions CI workflow running the same checks
- [x] `src/self_rag/config.py` — a validated `pydantic-settings` configuration
- [x] `src/self_rag/cli.py` — the `self-rag` entry point, starting with `self-rag config`
- [x] 18 unit tests covering every validation rule and the credential handling

### ✅ Stage 1 — Document conversion & chunking

Pure logic, no network and no LLM, so all of it is unit-testable.

- [x] `ingestion/converters.py` — PDF → Markdown, Markdown passthrough, SHA-256 file hashing
- [x] `ingestion/chunker.py` — the parent/child algorithm (header split → merge small → split
      large → rebalance → child split)
- [x] `text/tokens.py` — token estimation used later by the compression trigger
- [x] `self-rag chunk <file>` showing chunk counts and size distribution
- [x] 46 unit tests covering converters, parent/child chunking, tokens, and CLI

### ✅ Stage 2 — Parent store & Qdrant collection

- [x] `storage/parent_store.py` — parent chunks as JSON, returning `None` on a miss rather than raising
- [x] `storage/vector_store.py` — collection lifecycle, dimension guard, `delete_by_source`
- [x] Tests driven by a fake embedding stub, so they need no model download and run in CI
- [x] 27 unit tests for parent storage and vector database lifecycle (73 tests total)

### ✅ Stage 3 — Embeddings & hybrid search

- [x] `storage/embeddings.py` — a small LangChain `Embeddings` adapter over FastEmbed's ONNX runtime
- [x] BM25 sparse vectors wired up with the IDF modifier
- [x] `self-rag search "<query>"` returning ranked excerpts
- [x] 6 unit tests covering embeddings, factories, and CLI search (79 tests total)

### ✅ Stage 4 — Ingestion pipeline

- [x] `ingestion/pipeline.py` — convert → chunk → store parents → index children
- [x] Content-hash change detection (re-adding an unchanged file is a no-op)
- [x] Stale-chunk purge before re-indexing, and rollback that leaves no orphans behind
- [x] `self-rag ingest <file>` and `self-rag list`
- [x] 11 unit tests covering pipeline operations and CLI integration (90 tests total)

### ✅ Stage 5 — LLM layer, prompts & structured output

- [x] `llm/factory.py` — provider-agnostic construction, rate limiting, retries
- [x] `agent/prompts.py` — the system prompts, with snapshot tests
- [x] `agent/schemas.py` — `QueryAnalysis` and friends
- [x] `self-rag llm-check` — the first real model call
- [x] 17 unit tests covering prompts, schemas, factory, and CLI (107 tests total)

### ✅ Stage 6 — Agent state & retrieval tools

- [x] `agent/state.py` — graph state and its reducers (`accumulate_or_reset`, `set_union`, `append_unique`)
- [x] `agent/tools.py` — the two retrieval tools with JSON contracts, centralized sentinels, and ToolFactory
- [x] 28 unit tests covering reducers, state graphs, tools, and context formatters (135 tests total)

### ✅ Stage 7 — Minimal agent loop

- [x] `orchestrator` → `tools` → answer, plus `collect_answer`
- [x] `self-rag ask "<question>"` — the first grounded, cited answer
- [x] 14 unit tests covering orchestrator, routing, answer collection, agent loop integration, and CLI (149 tests total)

### ✅ Stage 8 — Research budget, compression & fallback

- [x] Tool-call and iteration budgets with routing to fallback
- [x] `should_compress_context` / `compress_context` with dynamic token growth factor
- [x] `fallback_response` when the budget runs out
- [x] 7 unit tests covering token estimation, threshold gating, context compression, message removals, and fallback response (156 tests total)

### ✅ Stage 9 — Main graph

- [x] Rolling conversation summary, query rewriting, clarification interrupt
- [x] Parallel sub-question fan-out and answer aggregation
- [x] SQLite checkpointer, so conversations survive a restart
- [x] 13 unit tests covering summarization, rewriting, clarification interrupt/resume, parallel fan-out, answer aggregation, and SQLite checkpointing (169 tests total)

### ✅ Stage 10 — Composition root, observability & CLI

- [x] `system.py` wiring everything together with dependency injection
- [x] Execution logging and optional Langfuse tracing
- [x] `self-rag chat` — a full multi-turn conversation
- [x] 11 unit tests covering observability, logging, RAGSystem, thread ID, config, and CLI chat (180 tests total)

### ✅ Stage 11 — Self-correction

- [x] A grader that scores retrieved evidence against the question
- [x] Corrective re-retrieval when the evidence does not support an answer, with its own budget
- [x] 27 unit tests covering document grading, answer groundedness, query refinement, routing, and self-correcting retrieval loop (207 tests total)

### ✅ Stage 12 — Gradio UI

- [x] Documents tab and streaming Chat tab
- [x] Per-browser-session conversation isolation
- [x] 13 unit tests covering CSS styling, formatters, create_app, document management, session isolation, and CLI launch (220 tests total)

### ✅ Stage 13 — Production polish

- [x] Qdrant as a server via `docker-compose` (a configuration-only switch)
- [x] Reranking via FastEmbed ONNX cross-encoder
- [x] Page-level citations in chunk metadata and formatted sources
- [x] Offline evaluation module and RAGAS-compatible dataset scoring with CLI `self-rag eval`
- [x] 24 unit tests covering reranking, page-level citations, Qdrant server mode, evaluation, and CLI integration (244 tests total)

---

## Decisions on record

| Decision | Choice | Reason |
|---|---|---|
| Language model | Google Gemini / Groq free tiers | No GPU and ~2 GiB free RAM, so a local 8B model is not usable here |
| Embeddings | FastEmbed ONNX, dense + sparse | Avoids installing PyTorch entirely |
| Layout | `src/` package, injected dependencies | Keeps the layering one-directional and testable |
| Vector store | Embedded Qdrant now, server in Stage 13 | No infrastructure while learning; the later swap proves the abstraction |
| Python | 3.12 | 3.14 breaks several pinned packages |

## Findings that changed the design

Things verified against the reference implementation rather than assumed. Each is explained in the
stage document where it becomes relevant.

1. **A hybrid search `score_threshold` is not a similarity threshold.** Qdrant ranks hybrid results
   by Reciprocal Rank Fusion, so the score is `1/(rank+2)` summed across the dense and sparse
   branches — it maxes out at 1.0 and carries no information about semantic closeness. Measured: a
   threshold of `0.4` cut a 7-result search down to 3. Disabled by default here.
2. **TypedDict fields cannot have default values.** `class State(TypedDict): flag: bool = False` is
   accepted by Python but the key is simply absent at runtime, so reading it raises `KeyError`.
   State fields will use `NotRequired` instead.
3. **Re-indexing a changed file must purge the old vectors first**, because each insert gets fresh
   random point IDs — otherwise the old chunks stay in the index forever.
4. **Passing a file path to a glob-based converter silently does nothing** when the name contains
   `[`, `]`, `*` or `?`. Single files will be converted directly.
