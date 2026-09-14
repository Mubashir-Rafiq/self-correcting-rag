# Improvement Ideas — A Learning Roadmap

This is a punch list for rebuilding/improvising this project from scratch. Everything here was found by actually reading the code in `project/`, not guessed — file/line references are included so you can go verify each claim yourself before changing it (that verification is itself a good way to learn the codebase).

Items are grouped by theme, each tagged with rough **effort** (S/M/L) and **why it matters**. Suggested build order is at the bottom. Items marked **✅ DONE** are already implemented in this codebase and reflected as baseline behavior in `BLUEPRINT.md` — use that file alongside this one for a from-scratch rebuild: `BLUEPRINT.md` is the target baseline, this file is what to layer on top of it.

---

## 1. Real Bugs / Limitations to Fix First — ✅ DONE

**All five items below are already implemented in this codebase** and are documented as the baseline behavior in `BLUEPRINT.md`. Left in place (rather than deleted) as a record of what was wrong and why — useful if you're rebuilding from scratch and want to understand what each fix actually changes. Nothing left to do in this section.

These weren't nitpicks — they would have bitten as soon as you deviated from "one person, one browser tab, never restart the server."

### 1.1 All chat users currently share ONE conversation (S, high impact)
`ui/gradio_app.py:10-11` creates a single `RAGSystem()` at app startup, and `RAGSystem.__init__` (`core/rag_system.py:20`) generates **one** `thread_id` for the whole process. Every browser tab that connects talks to the same LangGraph thread — messages interleave across users.

**Fix:** Use Gradio's per-session state (`gr.State`) to hold a `RAGSystem`/thread id per browser session, or at minimum generate a new thread id per Gradio session hash. This is explicitly called out as a known limitation in `project/README.md` ("assign a separate LangGraph thread ID per session") — it's flagged but not fixed. Good first PR.

**Status:** ✅ Fixed — see `BLUEPRINT.md` §12 (thread id now minted per-session via `gr.State` + `demo.load`, threaded through `chat()`/`clear_session()`).

### 1.2 Conversation memory dies on every restart (S, high impact)
`rag_agent/graph.py:25` uses `InMemorySaver()`. Restart the process (or redeploy) and every conversation vanishes — no error, just silent memory loss.

**Fix:** Swap in `langgraph-checkpoint-sqlite` (or `-postgres` for multi-instance deployments) for a two-line change:
```python
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
```
This alone makes the app survive restarts — a real, visible feature for very little effort.

**Status:** ✅ Fixed — see `BLUEPRINT.md` §9.2/§12 (uses a long-lived `sqlite3.Connection` + `SqliteSaver(conn)` rather than the `from_conn_string` context-manager form, since a long-running app has no natural `with`-block lifetime).

### 1.3 LLM provider is hardcoded, `config.py` lies about being the single control point (S)
`config.py` documents `LLM_MODEL = "granite4.1:8b"` as *the* setting, but `core/rag_system.py:27-31` hardcodes `ChatOllama(...)` directly. Switching providers today means editing `rag_system.py`, not `config.py` — contradicts the "central configuration hub" claim in `project/README.md`.

**Fix:** Build the small provider factory the README already sketches under "Optional Multi-Provider Configuration" — turn it into actual code instead of copy-paste documentation. See §3.1 below for a cleaner version of this using LangChain's `init_chat_model`.

**Status:** ✅ Fixed — see `BLUEPRINT.md` §14 (`init_chat_model(config.LLM_MODEL, model_provider=config.LLM_PROVIDER, ...)`; switching providers is now config-only, per §3.1's original suggestion).

### 1.4 Silent re-indexing staleness (S)
`core/document_manager.py:34-36` skips ingestion whenever a markdown file with the same **name** already exists — it never checks whether the *source PDF changed*. Re-uploading an edited PDF with the same filename silently keeps the old content indexed.

**Fix:** Hash the source file (e.g. `hashlib.sha256`) and store the hash in the parent metadata; skip only if the hash matches, not just the filename.

**Status:** ✅ Fixed — see `BLUEPRINT.md` §7.3. Implementing this surfaced a second bug the original description didn't anticipate: naively reprocessing a changed file left the *old* chunks as permanent stale duplicates in Qdrant (since `add_documents` assigns fresh random point IDs). The fix also purges the old vector points (`VectorDbManager.delete_by_source`) and orphaned parent JSON files before reprocessing.

### 1.5 Fragile string-based tool outputs (M)
`rag_agent/tools.py:36-41` and `:69-73` build retrieval results as hand-formatted strings (`f"Parent ID: {...}\nFile Name: {...}\nContent: {...}"`), and other code (`nodes.py`'s context builders, the aggregation prompts in `prompts.py`) has to parse or embed that format by convention. It's brittle — a stray colon or newline in a filename breaks downstream parsing.

**Fix:** Return structured JSON from tools (`json.dumps({"parent_id":..., "source":..., "content":...})`) and have the LLM-facing prompt tell the model it's reading JSON. Structured data is easier to log, test, and evolve than string templates.

**Status:** ✅ Fixed — see `BLUEPRINT.md` §11. `CHILD_CHUNK_SEPARATOR` was removed entirely; `nodes.py::_retrieval_contexts` now parses JSON (falling back to raw text for sentinel/error strings) and reconstructs the same downstream string shape, so nothing else needed to change.

---

## 2. Modernize the Project Scaffolding

The current repo is a great *teaching* codebase but skips almost all of the tooling a "real" 2026 Python project has. Adding this is low-effort and high learning value.

| What's missing | Add | Effort |
|---|---|---|
| No `pyproject.toml` — only `requirements.txt` | Migrate to `pyproject.toml` managed with `uv` (you're already using `uv` locally). Gives you a lockfile (`uv.lock`), reproducible installs, and `uv run` | S |
| No linting | [`ruff`](https://docs.astral.sh/ruff/) (format + lint in one tool, extremely fast) | S |
| No type checking | `mypy` or `pyright`, start with `--strict` only on `rag_agent/` | M |
| No pre-commit hooks | `pre-commit` running ruff + mypy on commit | S |
| No CI | GitHub Actions workflow: install deps, ruff check, run tests (see §5) on every push/PR | S |
| Config is a bare `.py` module with `os.environ` sprinkled in only for Langfuse | Replace `config.py` with a `pydantic-settings` `BaseSettings` class — validates types, reads `.env` automatically and consistently for *all* settings, not just Langfuse | M |
| No `docker-compose.yml` | Compose file running the app + a real **Qdrant server container** (see §3.4) + optionally Ollama, instead of one monolithic Dockerfile | M |

This category alone — pyproject + ruff + basic CI — takes an afternoon and immediately makes the repo look and behave like a modern OSS project.

---

## 3. RAG-Quality Upgrades (the actual ML/IR part)

These are the "did you actually learn how RAG works" upgrades — the highest-value ones for a portfolio.

### 3.1 Provider-agnostic LLM init (S) — ✅ DONE
LangChain now ships [`init_chat_model`](https://python.langchain.com/docs/how_to/chat_models_universal_init/), which does exactly the dispatch that `project/README.md` used to describe doing by hand. This replaces the entire hand-written `if/elif` provider factory the README used to suggest, with zero custom branching code — a genuinely "better way to build it" versus the original.

**Status:** ✅ Fixed — see `BLUEPRINT.md` §14. Implemented as `init_chat_model(config.LLM_MODEL, model_provider=config.LLM_PROVIDER, **llm_kwargs)`, using a separate `LLM_PROVIDER` config field rather than a `"provider:model"`-prefixed string, to keep `LLM_MODEL` itself unprefixed and consistent with the rest of `config.py`'s naming.

### 3.2 Add a reranker between search and answer (M, big quality win)
Right now `search_child_chunks` (`rag_agent/tools.py:12`) returns raw vector-similarity hits straight to the LLM. Adding a **cross-encoder reranker** (e.g. `sentence-transformers` `CrossEncoder`, or a lightweight one like `mixedbread-ai/mxbai-rerank-base-v1` or FlashRank) after retrieval, before returning to the LLM, is one of the best-known, lowest-effort RAG quality upgrades:
```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("mixedbread-ai/mxbai-rerank-base-v1")
scores = reranker.predict([(query, doc.page_content) for doc in results])
results = [doc for _, doc in sorted(zip(scores, results), reverse=True)][:top_n]
```
Since `sentence-transformers` is already a dependency, this needs no new install.

### 3.3 Page-level citations (S–M)
`utils.py:28` calls `pymupdf4llm.to_markdown(..., page_separators=True, ignore_images=True)` — page breaks exist in the text but aren't captured as structured metadata per chunk, so answers can only cite a filename, never a page number. Switch to `pymupdf4llm`'s `page_chunks=True` mode (or post-process the `page_separators` markers) to attach `page_number` to each parent/child chunk's metadata, then surface it in the "Sources" section the prompts already build (`prompts.py`). Turns "Sources: manual.pdf" into "Sources: manual.pdf, p. 14" — much more useful and barely more code.

### 3.4 Real vector DB server instead of embedded Qdrant (M)
`db/vector_db_manager.py:12` uses `QdrantClient(path=config.QDRANT_DB_PATH)` — an embedded, single-process, file-locked database. It can't be shared across multiple app instances and doesn't support concurrent writers. Running Qdrant as its own Docker container (`docker run qdrant/qdrant`) and connecting via `QdrantClient(url="http://localhost:6333")` is a one-line config change that unlocks horizontal scaling and a web dashboard for free (`http://localhost:6333/dashboard`).

### 3.5 Multi-format ingestion (S–M)
`core/document_manager.py:18` only accepts `.pdf` and `.md`. Adding `.docx`, `.txt`, `.html`, and `.csv` via `langchain_community.document_loaders` (or `unstructured`) is mostly a matter of adding a converter function per type and routing by extension before the existing chunker takes over — the chunking/indexing pipeline downstream doesn't care where the Markdown came from.

### 3.6 Adaptive / query-classification routing (M, "cool factor")
Every query today goes through the full pipeline: summarize → rewrite → parallel agent(s) with up to `MAX_TOOL_CALLS` tool calls each → aggregate. For a query like "hi" or "thanks", that's wasteful. Adding a cheap classification step ("does this need retrieval at all?") before `rewrite_query`, and short-circuiting straight to a direct LLM response for greetings/meta-questions, is the well-known **Adaptive RAG** pattern and a nice addition to `rag_agent/edges.py`.

### 3.7 Self-grading / groundedness check (L, stretch)
`config.JUDGE_MODEL` is already defined (`config.py:18`) but is currently used **only** by `notebooks/evaluation.ipynb` for offline RAGAS scoring — it's dead weight in the live app. A genuinely educational stretch goal: add a small "grader" node (Self-RAG / CRAG style) that uses `JUDGE_MODEL` to score whether the retrieved chunks actually support the drafted answer, and triggers one more retrieval pass if not, *before* returning to the user — turning an offline-only metric into an online self-correction signal.

---

## 4. New Features (low effort, high visible value)

- **Live settings panel in the UI** — expose `RETRIEVAL_SCORE_THRESHOLD`, `DEFAULT_RETRIEVAL_K`, and provider/model as Gradio dropdowns/sliders (`ui/gradio_app.py`) instead of requiring a `config.py` edit + restart.
- **Visualize the graph** — LangGraph can render its own structure: `agent_graph.get_graph().draw_mermaid_png()`. Add a button or a startup script that saves this, so newcomers can *see* the state machine instead of reading `graph.py`. (`assets/agentic_rag_workflow.png` currently looks hand-drawn — auto-generating it keeps it accurate as you modify the graph.)
- **Show retrieval scores in the chat UI** — `_handle_tool_result` already pretty-prints the JSON tool output (see §1.5) in the collapsible blocks; the remaining piece is exposing each result's similarity score, which `similarity_search` doesn't currently return (would need `similarity_search_with_score` in `tools.py` plus a `"score"` field in the JSON payload).
- **Token/cost tracking per conversation** — `utils.py:53` already has `estimate_context_tokens`; wire a running total into the UI so you can see the running estimated cost of a session, a genuinely useful learning exercise in understanding LLM cost structure.
- **CLI mode** — add a `project/cli.py` that drives `RAGSystem`/`ChatInterface` from the terminal without Gradio, useful for scripting, quick tests, and understanding that the Gradio UI is just one consumer of the graph.
- **Conversation export** — dump `agent_graph.get_state(config).values["messages"]` to Markdown/JSON for saving a session transcript.

---

## 5. Testing (currently: zero tests in the repo)

There are no automated tests anywhere in `project/` today — everything is manually verified by running the Gradio app. This is the single highest-leverage area to add for a "portfolio" version of this project, and it doesn't require calling a real LLM:

- **Unit-test the pure functions** in `document_chunker.py` (merge/split/rebalance logic) and `utils.py` — these have no LLM dependency and are prime `pytest` targets.
- **Test graph routing logic in isolation** using LangChain's `FakeListChatModel` / `GenericFakeChatModel` to stub LLM responses, so you can assert `route_after_rewrite` and `route_after_orchestrator_call` (`rag_agent/edges.py`) send state down the correct branch without hitting Ollama at all.
- **Snapshot-test the prompts** (`prompts.py`) so an accidental edit doesn't silently change agent behavior.
- Wire the above into the CI workflow from §2.

---

## 6. Suggested Build Order

§1's bug fixes are already done (see `BLUEPRINT.md`). Picking up from there:

1. **Scaffolding first** (§2): `pyproject.toml` + `uv`, `ruff`, a `tests/` folder with even one trivial test, GitHub Actions CI. Gives you a safety net before you start changing behavior.
2. **Add tests around the graph edges** (§5) before you touch `nodes.py`/`edges.py` further — you'll want the safety net for the next steps.
3. **Reranker** (§3.2) — the single highest ROI retrieval-quality change, and a great excuse to learn what a cross-encoder is and why it differs from the bi-encoder you already use for embeddings.
4. **Page-level citations** (§3.3) — touches the same ingestion/metadata pipeline as the reranker work, natural to pair together.
5. Pick your own stretch goal from §3.4–§3.7/§4 depending on whether you're more interested in agent behavior, retrieval quality, or UX polish.

Each of these is small enough to be its own PR/commit, which is also good practice for a portfolio repo — a clean commit history showing incremental, well-tested improvements reads a lot better to anyone reviewing it than one giant rewrite commit.
