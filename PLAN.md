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
| 1 | Document conversion & chunking | ⬜ | Turn a PDF into searchable parent/child chunks |
| 2 | Parent store & Qdrant collection | ⬜ | Somewhere to put those chunks |
| 3 | Embeddings & hybrid search | ⬜ | Actually find the right chunk for a question |
| 4 | Ingestion pipeline | ⬜ | One safe, repeatable "add this document" operation |
| 5 | LLM layer, prompts & structured output | ⬜ | The first real call to a language model |
| 6 | Agent state & retrieval tools | ⬜ | Tools the model can decide to call |
| 7 | Minimal agent loop | ⬜ | **The first real answer from your own documents** |
| 8 | Research budget, compression & fallback | ⬜ | An agent that can't loop forever or blow its context |
| 9 | Main graph | ⬜ | Memory, clarifying questions, parallel sub-questions |
| 10 | Composition root, observability & CLI | ⬜ | **A complete app you can chat with** |
| 11 | Self-correction | ⬜ | **The namesake:** it grades its own evidence and retries |
| 12 | Gradio UI | ⬜ | A browser interface |
| 13 | Production polish | ⬜ | Qdrant server, reranking, citations, evaluation |

---

## Where we are now

**Stage 0 is complete.** The repository is a working, self-checking Python project: dependencies
resolve and install reproducibly, and lint, formatting, strict type checking and tests all pass
locally and in CI. The configuration layer that every later stage reads from is in place and
validated.

There is not yet any RAG behaviour — that begins in Stage 1.

Run it yourself:

```bash
uv sync --all-extras
uv run self-rag config     # print the resolved configuration
uv run pytest              # 18 tests
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

### ⬜ Stage 1 — Document conversion & chunking

Pure logic, no network and no LLM, so all of it is unit-testable.

- [ ] `ingestion/converters.py` — PDF → Markdown, Markdown passthrough, SHA-256 file hashing
- [ ] `ingestion/chunker.py` — the parent/child algorithm (header split → merge small → split
      large → rebalance → child split)
- [ ] `text/tokens.py` — token estimation used later by the compression trigger
- [ ] `self-rag chunk <file>` showing chunk counts and size distribution

### ⬜ Stage 2 — Parent store & Qdrant collection

- [ ] `storage/parent_store.py` — parent chunks as JSON, returning `None` on a miss rather than raising
- [ ] `storage/vector_store.py` — collection lifecycle, dimension guard, `delete_by_source`
- [ ] Tests driven by a fake embedding stub, so they need no model download and run in CI

### ⬜ Stage 3 — Embeddings & hybrid search

- [ ] `storage/embeddings.py` — a small LangChain `Embeddings` adapter over FastEmbed's ONNX runtime
- [ ] BM25 sparse vectors wired up with the IDF modifier
- [ ] `self-rag search "<query>"` returning ranked excerpts

### ⬜ Stage 4 — Ingestion pipeline

- [ ] `ingestion/pipeline.py` — convert → chunk → store parents → index children
- [ ] Content-hash change detection (re-adding an unchanged file is a no-op)
- [ ] Stale-chunk purge before re-indexing, and rollback that leaves no orphans behind
- [ ] `self-rag ingest <file>` and `self-rag list`

### ⬜ Stage 5 — LLM layer, prompts & structured output

- [ ] `llm/factory.py` — provider-agnostic construction, rate limiting, retries
- [ ] `agent/prompts.py` — the system prompts, with snapshot tests
- [ ] `agent/schemas.py` — `QueryAnalysis` and friends
- [ ] `self-rag llm-check` — the first real model call

### ⬜ Stage 6 — Agent state & retrieval tools

- [ ] `agent/state.py` — graph state and its reducers
- [ ] `agent/tools.py` — the two retrieval tools with JSON contracts and centralised sentinels

### ⬜ Stage 7 — Minimal agent loop

- [ ] `orchestrator` → `tools` → answer, plus `collect_answer`
- [ ] `self-rag ask "<question>"` — the first grounded, cited answer

### ⬜ Stage 8 — Research budget, compression & fallback

- [ ] Tool-call and iteration budgets
- [ ] `should_compress_context` / `compress_context`
- [ ] `fallback_response` when the budget runs out

### ⬜ Stage 9 — Main graph

- [ ] Rolling conversation summary, query rewriting, clarification interrupt
- [ ] Parallel sub-question fan-out and answer aggregation
- [ ] SQLite checkpointer, so conversations survive a restart

### ⬜ Stage 10 — Composition root, observability & CLI

- [ ] `system.py` wiring everything together with dependency injection
- [ ] Execution logging and optional Langfuse tracing
- [ ] `self-rag chat` — a full multi-turn conversation

### ⬜ Stage 11 — Self-correction

- [ ] A grader that scores retrieved evidence against the question
- [ ] Corrective re-retrieval when the evidence does not support an answer, with its own budget

### ⬜ Stage 12 — Gradio UI

- [ ] Documents tab and streaming Chat tab
- [ ] Per-browser-session conversation isolation

### ⬜ Stage 13 — Production polish

- [ ] Qdrant as a server via `docker-compose` (a configuration-only switch)
- [ ] Reranking, page-level citations, RAGAS evaluation

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
