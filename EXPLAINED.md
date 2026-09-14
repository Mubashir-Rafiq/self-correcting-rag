# Agentic RAG for Dummies — Explained for Students

This document explains the project in plain language, assuming you know general CS concepts (functions, graphs, APIs) but have **not** necessarily worked with LLMs, RAG, or LangGraph before. Read it top to bottom once, then use it as a reference while you read the code.

---

## 1. The Problem This Project Solves

A plain LLM chatbot (like base ChatGPT) only knows what it was trained on. It can't answer questions about *your* private PDFs, and it can't tell you which document a fact came from.

**RAG (Retrieval-Augmented Generation)** fixes this with a simple idea:

1. Chop your documents into small pieces ("chunks").
2. When the user asks a question, search those chunks for the most relevant ones.
3. Paste the relevant chunks into the LLM's prompt, along with the question.
4. The LLM answers using that pasted context instead of guessing from memory.

**Agentic RAG** goes one step further: instead of "search once, answer once," an LLM-driven **agent** decides *for itself*, in a loop, whether to search again, search differently, fetch more context, or stop and answer. It behaves more like a researcher than a lookup table.

This project builds exactly that, using a framework called **LangGraph**.

---

## 2. Key Vocabulary (read this before anything else)

| Term | Meaning |
|---|---|
| **Chunk** | A small piece of a document (a few sentences to a paragraph) stored so it can be searched. |
| **Embedding** | A list of numbers (a vector) that represents the *meaning* of a piece of text. Similar meanings → similar vectors. |
| **Vector database** | A database (here: **Qdrant**) specialized in storing embeddings and finding the "nearest" ones to a query — i.e., semantic search. |
| **Dense retrieval** | Search by meaning, using embeddings (handles synonyms, paraphrasing). |
| **Sparse retrieval (BM25)** | Classic keyword search (handles exact terms, codes, acronyms that embeddings sometimes blur). |
| **Hybrid search** | Using both dense + sparse together and combining the results — this project's default. |
| **LLM tool calling / function calling** | The ability of an LLM to say "call this function with these arguments" instead of just returning text. This is what lets the LLM *decide* to search. |
| **LangGraph** | A library for building LLM applications as an explicit **graph of steps (nodes)** connected by **edges**, with support for loops, conditional branches, and parallel branches — instead of one long linear prompt chain. |
| **State** | A shared Python dict/object that flows through the graph. Every node reads from it and returns updates to it. |
| **Agent** | Here, an LLM wrapped in a loop: it looks at the state, optionally calls a tool (like "search"), looks at the result, and repeats until it's ready to answer. |
| **Node** | One step in the LangGraph graph — a Python function that takes the state and returns updates. |
| **Edge** | A connection between nodes. A **conditional edge** picks the next node based on the state (like an `if` statement for the graph). |
| **Checkpointer** | LangGraph's mechanism for saving state between turns of a conversation (so the agent "remembers" earlier messages). |

---

## 3. The Big Idea: Parent/Child Chunking

Small chunks are great for *finding* the right spot in a document (precise search), but bad for *answering*, because they lack surrounding context. Large chunks are great for context, but bad for search precision (too much irrelevant text dilutes the match).

This project uses **both**, linked together:

```
Document
   └── Parent chunk (large, ~2000-4000 chars, split on Markdown headers)
          └── Child chunks (small, ~500 chars, split from the parent)
```

- **Child chunks** go into the vector database and are what gets searched.
- **Parent chunks** are stored separately as plain JSON files, each tagged with a `parent_id`.
- When a search hits a child chunk, the agent can optionally say "give me the *whole* parent this came from" to get more surrounding context before answering.

This is why the project has two tools (see §5): `search_child_chunks` (search) and `retrieve_parent_chunks` (expand context).

---

## 4. Where the Files Live and What They Do

```
project/
├── app.py                 # Entry point: run this to launch the app
├── config.py               # ALL tunable settings live here (models, chunk sizes, limits)
├── document_chunker.py     # Splits documents into parent/child chunks
├── utils.py                 # PDF → Markdown conversion, token counting helpers
├── .env.example             # Template for optional Langfuse observability keys
│
├── core/
│   ├── rag_system.py        # "Wires everything together": creates DB, embeddings, LLM, compiles the graph
│   ├── document_manager.py  # Ingestion pipeline: PDF → Markdown → chunks → vector DB
│   ├── chat_interface.py    # Talks to the compiled LangGraph graph, streams the answer to the UI
│   └── observability.py     # Optional Langfuse tracing hookup
│
├── db/
│   ├── vector_db_manager.py     # Qdrant client + embedding models wrapper
│   └── parent_store_manager.py  # Reads/writes parent chunk JSON files on disk
│
├── rag_agent/                # <-- THE CORE LOGIC: the LangGraph agent itself
│   ├── graph.py               # Builds and compiles the graph (nodes + edges wired together)
│   ├── graph_state.py         # Defines the State/AgentState data structures (see §7 of the tutorial README)
│   ├── nodes.py                # The actual step functions (summarize, rewrite, orchestrate, aggregate...)
│   ├── edges.py                # The conditional routing logic ("if X, go to node A, else node B")
│   ├── tools.py                 # The two retrieval tools the LLM can call
│   ├── prompts.py               # All the system prompts (the "personality"/instructions for each LLM call)
│   └── schemas.py               # Pydantic models for structured LLM outputs (e.g., "is this query clear?")
│
└── ui/
    ├── gradio_app.py         # The chat UI (built with Gradio, a Python web-UI library)
    └── css.py                 # Styling for the UI
```

**Reading order if you want to actually understand the code**, from simplest to most complex:
1. `config.py` — see all the knobs that exist
2. `rag_agent/tools.py` — the two functions the LLM can call
3. `rag_agent/graph_state.py` — the shape of the data flowing through the graph
4. `rag_agent/prompts.py` — what each LLM call is told to do
5. `rag_agent/nodes.py` + `rag_agent/edges.py` — the actual logic
6. `rag_agent/graph.py` — how it's all assembled into a graph
7. `core/rag_system.py` and `app.py` — how the graph gets built and launched

---

## 5. The Two Tools the Agent Can Call

```python
search_child_chunks(query: str, limit: int = 7) -> str
```
Searches the vector DB for the `limit` most relevant child chunks (hybrid dense + sparse). Returns their text plus their `parent_id` and source filename, or `"NO_RELEVANT_CHUNKS"` if nothing scores above the similarity threshold.

```python
retrieve_parent_chunks(parent_id: str) -> str
```
Loads the full parent chunk (bigger context) for a `parent_id` returned by the search above.

The LLM decides *when* to call these — that's the "agentic" part. It's not a fixed pipeline of "always search once, always fetch parent"; the LLM reasons about what it needs.

---

## 6. The Full Request Lifecycle (step by step)

Say the user asks: *"What is JavaScript? What is Python?"*

```
1. summarize_history   → Compress old chat turns into a rolling summary (keeps memory small)
2. rewrite_query       → LLM checks: is this clear? Can it be split into sub-questions?
                          → Here: splits into ["What is JavaScript?", "What is Python?"]
3. route_after_rewrite → If unclear → pause and ask user (see §8).
                          If clear   → fan out: spawn one parallel "agent" subgraph per sub-question
4. agent subgraph (×2, running in parallel — one per sub-question):
      a. orchestrator        → LLM decides: call search_child_chunks? call retrieve_parent_chunks? or answer?
      b. tools (if called)   → actually executes the tool, returns result to the LLM
      c. should_compress_context → if the conversation got too long (token-wise), summarize it down
      d. (loop back to orchestrator until the LLM stops calling tools, or a limit is hit)
      e. fallback_response   → if limits (MAX_TOOL_CALLS / MAX_ITERATIONS) are hit before an answer, force
                                a best-effort answer from whatever was retrieved so far
      f. collect_answer      → package the final text answer for this sub-question
5. aggregate_answers   → Combine both sub-question answers into ONE final response for the user
```

This is a **map-reduce pattern**: the query is "mapped" into independent sub-questions handled in parallel, then "reduced" (aggregated) into one answer.

---

## 7. Why a Graph Instead of "Just Prompt the LLM"?

A single big prompt can't:
- Loop ("search again if the first result wasn't good enough")
- Branch ("if the question is ambiguous, stop and ask the user; otherwise proceed")
- Run things in parallel ("handle these 3 sub-questions simultaneously")
- Persist memory across turns cleanly

LangGraph gives you an explicit state machine for all of this, instead of hiding it inside one enormous prompt. Each node is small, testable, and independently swappable — that's also why the codebase is organized as multiple `.py` files instead of one script.

---

## 8. Human-in-the-Loop Clarification

If `rewrite_query` decides the question is genuinely ambiguous (e.g., "How do I update it?" with no prior context for what "it" is), the graph **pauses** at `request_clarification` (LangGraph's `interrupt_before` feature) instead of guessing. The next user message is treated as the clarification, gets merged with the original ambiguous query, and `rewrite_query` runs again. This avoids wasting retrieval calls on a question the system can't actually answer yet.

---

## 9. Context Compression (why it exists)

Each agent subgraph might call tools multiple times, accumulating a big pile of tool outputs in its conversation. If left unchecked, that grows the prompt sent to the LLM every iteration, which is slow and expensive. So after every tool call, `should_compress_context` estimates the token count (`tiktoken`); once it crosses a threshold, `compress_context` asks the LLM to summarize everything found *so far* into a compact Markdown summary, then deletes the raw messages. The loop continues with a small, information-dense summary instead of the full history.

---

## 10. Configuration Cheat Sheet (`project/config.py`)

| Setting | What it controls | When to change it |
|---|---|---|
| `LLM_MODEL` | Which chat model answers questions | Switching provider/model |
| `DENSE_MODEL` / `SPARSE_MODEL` | Embedding models for search | Better quality/speed trade-off, or after language changes |
| `RETRIEVAL_SCORE_THRESHOLD` | Minimum similarity score to keep a search result | Too few results → lower it; too much noise → raise it |
| `DEFAULT_RETRIEVAL_K` | How many child chunks to fetch per search | More context per search vs. more noise |
| `CHILD_CHUNK_SIZE` / `MIN_PARENT_SIZE` / `MAX_PARENT_SIZE` | Chunking granularity | Technical docs = smaller; legal/narrative docs = larger |
| `MAX_TOOL_CALLS` / `MAX_ITERATIONS` | Agent loop budget (prevents infinite loops) | Complex multi-hop questions need more; simple Q&A needs less |
| `BASE_TOKEN_THRESHOLD` / `TOKEN_GROWTH_FACTOR` | When to trigger context compression | Losing important details → increase threshold |
| `LANGFUSE_ENABLED` and friends | Optional tracing/observability | Turn on to debug what the agent is actually doing under the hood |

---

## 11. How to Run It (after setup)

```bash
source .venv/bin/activate          # activate the virtual environment (already created for you)
python project/app.py              # launch the app
```

Open the printed local URL (usually `http://127.0.0.1:7860`) in a browser. Upload PDFs through the UI, wait for indexing, then ask questions in the chat box.

By default the app expects a local LLM served by **Ollama** (`LLM_MODEL = "granite4.1:8b"` in `config.py`). If you don't have Ollama installed, either:
- Install it from https://ollama.com and run `ollama pull granite4.1:8b`, **or**
- Swap in a cloud provider (OpenAI/Anthropic/Google) by following the "Switching LLM Provider" section of `project/README.md` — you'll need an API key and a one-line edit to `core/rag_system.py` plus `config.py`.

---

## 12. Where to Learn More

- `README.md` (repo root) — the full step-by-step tutorial that this project is generated from; great for understanding *why* each piece of code exists.
- `project/README.md` — reference documentation for the modular app: configuration, customization recipes, Docker deployment, troubleshooting table.
- `notebooks/agentic_rag.ipynb` — the same system built inline in a single notebook, useful for experimenting cell-by-cell.
- `notebooks/evaluation.ipynb` — scoring retrieval/answer quality with RAGAS metrics.
- [LangGraph docs](https://docs.langchain.com/oss/python/langgraph) — the underlying framework.

---

## 13. Common Points of Confusion (FAQ)

**Q: Why two chunk sizes instead of one?**
Small chunks make search precise (a query only needs to match a small piece), but answering needs surrounding context that a small chunk doesn't have. Parent/child chunking gets both.

**Q: Why does the "agent" sometimes call the search tool more than once?**
Because it's not a script — the LLM decides after seeing each result whether it has enough evidence. If the first search wasn't good enough, it can rephrase and search again, up to `MAX_TOOL_CALLS`.

**Q: What happens if the agent never finds a good answer?**
After `MAX_TOOL_CALLS` or `MAX_ITERATIONS` is hit, `fallback_response` forces the LLM to answer with whatever partial evidence it *did* find, instead of looping forever.

**Q: Why is there a separate summarization step for conversation history?**
So a long chat doesn't blow up the prompt size / cost on every single turn. Old turns get compressed into a short rolling summary; only the most recent few messages stay verbatim.

**Q: Is this project tied to Ollama?**
No — the *runnable app* defaults to Ollama for zero-cost local testing, but the LLM is just a variable (`llm = ChatOllama(...)`) that can be swapped for `ChatOpenAI`, `ChatAnthropic`, or `ChatGoogleGenerativeAI` from LangChain, as documented in `project/README.md`.
