# Rebuild Blueprint — Agentic RAG System

This is a complete technical specification of this project as it currently stands (i.e., **including** the five fixes already applied under `IMPROVEMENTS.md` §1 — config-driven provider, persistent checkpointer, per-session threads, content-hash re-indexing, structured JSON tool outputs). Use it together with `IMPROVEMENTS.md` to rebuild an improved version from scratch:

- **This file** = "here is exactly what the current system does and how it's wired" (the baseline to reproduce).
- **`IMPROVEMENTS.md`** = "here is what to change/add on top of that baseline" (§1 is already reflected here; §2–§6 are still open).
- **`EXPLAINED.md`** = a beginner-friendly conceptual tour, if you want the "why" in plain language alongside this spec's "what/how."

Where logic is intricate or easy to get subtly wrong (the chunking algorithm, the graph routing conditions, the prompts, the JSON tool contracts), this document reproduces it **verbatim** rather than paraphrasing — paraphrasing that kind of logic is how rebuilds drift from working behavior.

---

## 1. System Summary

An agentic RAG (Retrieval-Augmented Generation) chat app: users upload PDFs/Markdown, the system indexes them with **hierarchical parent/child chunking** into **Qdrant** (hybrid dense+sparse search), and a **LangGraph**-orchestrated agent answers questions by iteratively searching, optionally fetching larger parent context, self-correcting, and falling back gracefully when a research budget is exhausted. Conversation memory is rolling-summarized and persisted across restarts. The default LLM runtime is local **Ollama**, swappable to any LangChain chat-model provider via config only.

---

## 2. Tech Stack & Dependencies

Python **3.11+** (this repo's `.venv` was built with 3.11 via `uv` since the host had 3.14, which breaks some pinned packages).

```
fastembed==0.8.0                    # BM25 sparse embedding backend for FastEmbedSparse
gradio==6.26.0                      # Web UI
ipykernel==7.3.0                    # Notebook support (notebooks/ only)
langchain==1.4.0                    # init_chat_model (provider-agnostic LLM factory)
langchain-huggingface==1.2.2        # HuggingFaceEmbeddings (dense vectors)
langchain-ollama==1.1.0             # ChatOllama backend
langchain-qdrant==1.1.0             # QdrantVectorStore (hybrid retrieval)
langchain-text-splitters==1.1.2     # MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
langfuse==4.15.1                    # Optional LLM/graph observability
langgraph==1.2.11                   # Agent graph orchestration
langgraph-checkpoint-sqlite==3.1.1  # Persistent conversation checkpointer
matplotlib==3.11.1                  # notebooks/evaluation.ipynb plots
pymupdf4llm==1.28.2                 # PDF -> Markdown conversion
python-dotenv==1.2.3                # .env loading
ragas==0.4.3                        # notebooks/evaluation.ipynb RAG metrics
seaborn==0.13.2                     # notebooks/evaluation.ipynb plots
sentence-transformers==6.0.0        # transitive dep of langchain-huggingface
tiktoken==0.14.0                    # token counting for context-compression triggers
```

Also install **PyTorch CPU-only** first if there's no GPU, to avoid pulling multi-GB CUDA packages:
```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r requirements.txt
```

**Runtime dependency (not pip-installed):** [Ollama](https://ollama.com) running locally with a tool-calling-capable model pulled (default `granite4.1:8b`) — **or** an API key for OpenAI/Anthropic/Google if using a cloud provider instead (see §14).

---

## 3. Repository Layout

```
project/
├── app.py                     # Entry point — loads .env, launches Gradio
├── config.py                  # Every tunable setting (see §5)
├── document_chunker.py        # Parent/child chunking algorithm (see §7.2)
├── utils.py                   # PDF->MD conversion, file hashing, token estimation
├── .env.example                # Langfuse env var template
├── Dockerfile                  # Bundles Ollama + app in one container
│
├── core/
│   ├── rag_system.py           # Bootstraps LLM, DB, tools, compiles the graph
│   ├── document_manager.py     # Ingestion pipeline (convert -> chunk -> index)
│   ├── chat_interface.py       # Streams graph output into Gradio chat messages
│   ├── observability.py        # Optional Langfuse callback handler
│   └── execution_logger.py     # Optional verbose terminal logging of graph steps
│
├── db/
│   ├── vector_db_manager.py    # Qdrant client + embeddings + collection lifecycle
│   └── parent_store_manager.py # Parent chunks as JSON files on disk
│
├── rag_agent/
│   ├── graph.py                 # Builds + compiles both graphs (see §9)
│   ├── graph_state.py           # State / AgentState schemas (see §8)
│   ├── nodes.py                  # All node function implementations
│   ├── edges.py                  # Conditional routing functions
│   ├── tools.py                  # search_child_chunks, retrieve_parent_chunks
│   ├── prompts.py                # All 6 system prompts (see §10)
│   └── schemas.py                # QueryAnalysis (structured LLM output)
│
└── ui/
    ├── gradio_app.py            # Blocks layout: Documents tab + Chat tab
    └── css.py                   # Custom CSS string passed to demo.launch(css=...)
```

Top-level (outside `project/`): `requirements.txt`, `.gitignore`, `README.md` (step-by-step notebook-style tutorial), `notebooks/` (standalone notebook reimplementation + PDF conversion + evaluation), `assets/` (logo, demo gif, architecture diagram).

Runtime-generated, git-ignored: `markdown_docs/`, `parent_store/`, `qdrant_db/`, `checkpoints.db`, `.env`.

---

## 4. Environment / .env

```bash
# project/.env (copied from .env.example)
LANGFUSE_ENABLED=false
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=http://localhost:3000
```

Loaded via `python-dotenv` at the very top of `app.py`. All other settings live in `config.py`, not environment variables (only Langfuse reads `os.environ`, as a deliberate override point).

---

## 5. Configuration Reference (`config.py`, full file)

```python
import os

# --- Directory Configuration ---
_BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MARKDOWN_DIR = os.path.join(_BASE_DIR, "markdown_docs")
PARENT_STORE_PATH = os.path.join(_BASE_DIR, "parent_store")
QDRANT_DB_PATH = os.path.join(_BASE_DIR, "qdrant_db")
CHECKPOINT_DB_PATH = os.path.join(_BASE_DIR, "checkpoints.db")

# --- Qdrant Configuration ---
CHILD_COLLECTION = "document_child_chunks"
SPARSE_VECTOR_NAME = "sparse"

# --- Model Configuration ---
DENSE_MODEL = "Qwen/Qwen3-Embedding-0.6B"
SPARSE_MODEL = "Qdrant/bm25"
LLM_PROVIDER = "ollama"  # e.g. "ollama", "openai", "anthropic", "google_genai" (requires the matching langchain-<provider> package)
LLM_MODEL = "granite4.1:8b"
JUDGE_MODEL = "ministral-3:3b-instruct-2512-q8_0"  # used only by notebooks/evaluation.ipynb (RAGAS), not the live app
LLM_TEMPERATURE = 0
LLM_SEED = 42  # only applied when LLM_PROVIDER == "ollama"; not all providers accept a seed

# --- Retrieval Configuration ---
RETRIEVAL_SCORE_THRESHOLD = 0.4
DEFAULT_RETRIEVAL_K = 7

# --- Agent Configuration ---
MAX_TOOL_CALLS = 8
MAX_ITERATIONS = 10
GRAPH_RECURSION_LIMIT = 50
MAIN_HISTORY_MESSAGES_TO_KEEP = 4
BASE_TOKEN_THRESHOLD = 2000
TOKEN_GROWTH_FACTOR = 0.9

# --- Terminal Execution Logging ---
EXECUTION_LOGGING_ENABLED = False
EXECUTION_LOG_MAX_CHARS = 1200
EXECUTION_LOG_USE_COLOR = True

# --- Text Splitter Configuration ---
CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 100
MIN_PARENT_SIZE = 2000
MAX_PARENT_SIZE = 4000
HEADERS_TO_SPLIT_ON = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3")
]

# --- Langfuse Observability ---
LANGFUSE_ENABLED = os.environ.get("LANGFUSE_ENABLED", "false").lower() == "true"
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "")
LANGFUSE_BASE_URL = os.environ.get("LANGFUSE_BASE_URL", "http://localhost:3000")
```

| Setting | Meaning |
|---|---|
| `LLM_PROVIDER` / `LLM_MODEL` | Fed directly into `init_chat_model(LLM_MODEL, model_provider=LLM_PROVIDER, ...)` — see §14 |
| `RETRIEVAL_SCORE_THRESHOLD` | Minimum similarity score `search_child_chunks` accepts; below this, results are dropped |
| `DEFAULT_RETRIEVAL_K` | Default `limit` for `search_child_chunks` |
| `MAX_TOOL_CALLS` / `MAX_ITERATIONS` | Hard budget per agent sub-question before forcing `fallback_response` |
| `BASE_TOKEN_THRESHOLD` / `TOKEN_GROWTH_FACTOR` | Drive the context-compression trigger formula — see §9.3 |
| `MAIN_HISTORY_MESSAGES_TO_KEEP` | Raw (uncompressed) messages kept in the main graph's history after each turn; must be ≥ 2 |
| `CHILD_CHUNK_SIZE`/`OVERLAP`, `MIN`/`MAX_PARENT_SIZE` | Feed the chunking algorithm — see §7.2 |

---

## 6. Provider Reference Table (LLM)

| Provider | `LLM_PROVIDER` value | Env var | Package |
|---|---|---|---|
| Ollama (local, default) | `ollama` | none | `langchain-ollama` |
| OpenAI | `openai` | `OPENAI_API_KEY` | `langchain-openai` |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` | `langchain-anthropic` |
| Google | `google_genai` | `GOOGLE_API_KEY` | `langchain-google-genai` |

---

## 7. Data Pipeline

### 7.1 PDF → Markdown (`utils.py`)

```python
def pdf_to_markdown(pdf_path, output_dir):
    doc = pymupdf.open(pdf_path)
    md = pymupdf4llm.to_markdown(doc, header=False, footer=False, page_separators=True, ignore_images=True, write_images=False, image_path=None)
    md_cleaned = md.encode('utf-8', errors='surrogatepass').decode('utf-8', errors='ignore')
    output_path = Path(output_dir) / Path(doc.name).stem
    Path(output_path).with_suffix(".md").write_bytes(md_cleaned.encode('utf-8'))

def pdfs_to_markdowns(path_pattern, overwrite: bool = False):
    output_dir = Path(config.MARKDOWN_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    for pdf_path in map(Path, glob.glob(path_pattern)):
        md_path = (output_dir / pdf_path.stem).with_suffix(".md")
        if overwrite or not md_path.exists():
            pdf_to_markdown(pdf_path, output_dir)
```

Notes: images are dropped entirely (`ignore_images=True`); page breaks exist as separator text but are not captured as structured per-chunk metadata (no page-number citations — see `IMPROVEMENTS.md` §3.3 for that gap).

### 7.2 Parent/Child Chunking Algorithm (`document_chunker.py`, full file)

This is the most intricate piece of custom logic in the project — reproduce it exactly:

```python
import os
import glob
import config
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

class DocumentChunker:
    def __init__(self):
        if config.MIN_PARENT_SIZE <= 0 or config.MAX_PARENT_SIZE < config.MIN_PARENT_SIZE:
            raise ValueError("Parent chunk sizes must be positive and MIN_PARENT_SIZE <= MAX_PARENT_SIZE.")
        if not 0 <= config.CHILD_CHUNK_OVERLAP < config.CHILD_CHUNK_SIZE:
            raise ValueError("CHILD_CHUNK_OVERLAP must be smaller than CHILD_CHUNK_SIZE.")
        if config.CHILD_CHUNK_OVERLAP >= config.MAX_PARENT_SIZE:
            raise ValueError("CHILD_CHUNK_OVERLAP must be smaller than MAX_PARENT_SIZE.")

        self.__parent_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=config.HEADERS_TO_SPLIT_ON,
            strip_headers=False
        )
        self.__child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHILD_CHUNK_SIZE,
            chunk_overlap=config.CHILD_CHUNK_OVERLAP
        )
        self.__min_parent_size = config.MIN_PARENT_SIZE
        self.__max_parent_size = config.MAX_PARENT_SIZE

    @staticmethod
    def __merge_metadata(target, source, prepend=False):
        for key, value in source.items():
            if key not in target:
                target[key] = value
            else:
                first, second = (value, target[key]) if prepend else (target[key], value)
                values = [
                    item.strip()
                    for raw in (first, second)
                    for item in str(raw).split(" -> ")
                    if item.strip()
                ]
                target[key] = " -> ".join(dict.fromkeys(values))

    def create_chunks(self, path_dir=config.MARKDOWN_DIR):
        all_parent_chunks, all_child_chunks = [], []
        for doc_path_str in sorted(glob.glob(os.path.join(path_dir, "*.md"))):
            doc_path = Path(doc_path_str)
            parent_chunks, child_chunks = self.create_chunks_single(doc_path)
            all_parent_chunks.extend(parent_chunks)
            all_child_chunks.extend(child_chunks)
        return all_parent_chunks, all_child_chunks

    def create_chunks_single(self, md_path, source_name=None):
        doc_path = Path(md_path)
        source_name = source_name or f"{doc_path.stem}.pdf"

        with open(doc_path, "r", encoding="utf-8") as f:
            parent_chunks = self.__parent_splitter.split_text(f.read())

        merged_parents = self.__merge_small_parents(parent_chunks)
        split_parents = self.__split_large_parents(merged_parents)
        cleaned_parents = self.__clean_small_chunks(split_parents)
        if any(len(chunk.page_content) > self.__max_parent_size for chunk in cleaned_parents):
            raise ValueError("Parent chunking produced a chunk larger than MAX_PARENT_SIZE.")

        all_parent_chunks, all_child_chunks = [], []
        self.__create_child_chunks(all_parent_chunks, all_child_chunks, cleaned_parents, doc_path, source_name)
        return all_parent_chunks, all_child_chunks

    def __merge_small_parents(self, chunks):
        # Greedily concatenate consecutive header-split chunks until each
        # accumulated block reaches min_parent_size.
        if not chunks:
            return []
        merged, current = [], None
        for chunk in chunks:
            if current is None:
                current = chunk
            else:
                current.page_content += "\n\n" + chunk.page_content
                self.__merge_metadata(current.metadata, chunk.metadata)
            if len(current.page_content) >= self.__min_parent_size:
                merged.append(current)
                current = None
        if current:
            if merged:
                merged[-1].page_content += "\n\n" + current.page_content
                self.__merge_metadata(merged[-1].metadata, current.metadata)
            else:
                merged.append(current)
        return merged

    def __split_large_parents(self, chunks):
        # Anything still over max_parent_size gets recursively character-split.
        split_chunks = []
        for chunk in chunks:
            if len(chunk.page_content) <= self.__max_parent_size:
                split_chunks.append(chunk)
            else:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.__max_parent_size,
                    chunk_overlap=config.CHILD_CHUNK_OVERLAP
                )
                split_chunks.extend(splitter.split_documents([chunk]))
        return split_chunks

    def __rebalance_pair(self, first, second):
        # Re-splits a boundary between two neighboring chunks at a cleaner
        # separator (paragraph > line > space) closest to the midpoint,
        # while respecting min/max size constraints on both sides.
        combined = first.page_content.rstrip() + "\n\n" + second.page_content.lstrip()
        lower = max(1, len(combined) - self.__max_parent_size)
        upper = min(self.__max_parent_size, len(combined) - 1)
        if len(combined) >= 2 * self.__min_parent_size:
            lower = max(lower, self.__min_parent_size)
            upper = min(upper, len(combined) - self.__min_parent_size)
        preferred = min(max(len(combined) // 2, lower), upper)

        split_at = preferred
        for separator in ("\n\n", "\n", " "):
            before = combined.rfind(separator, lower, preferred + 1)
            after = combined.find(separator, preferred, upper + 1)
            if before >= lower:
                split_at = before
                break
            if after != -1:
                split_at = after
                break

        left_text = combined[:split_at].rstrip()
        right_text = combined[split_at:].lstrip()
        if len(combined) >= 2 * self.__min_parent_size and (
            len(left_text) < self.__min_parent_size or len(right_text) < self.__min_parent_size
        ):
            split_at = preferred
            left_text, right_text = combined[:split_at], combined[split_at:]
        if not left_text or not right_text:
            return first, second

        metadata = dict(first.metadata)
        self.__merge_metadata(metadata, second.metadata)
        first.page_content, first.metadata = left_text, dict(metadata)
        second.page_content, second.metadata = right_text, dict(metadata)
        return first, second

    def __clean_small_chunks(self, chunks):
        # Pass 1: absorb undersized chunks into a neighbor if it fits within max size.
        cleaned = []
        for i, chunk in enumerate(chunks):
            if len(chunk.page_content) < self.__min_parent_size:
                if cleaned and len(cleaned[-1].page_content) + 2 + len(chunk.page_content) <= self.__max_parent_size:
                    cleaned[-1].page_content += "\n\n" + chunk.page_content
                    self.__merge_metadata(cleaned[-1].metadata, chunk.metadata)
                elif i < len(chunks) - 1 and len(chunk.page_content) + 2 + len(chunks[i + 1].page_content) <= self.__max_parent_size:
                    chunks[i + 1].page_content = chunk.page_content + "\n\n" + chunks[i + 1].page_content
                    self.__merge_metadata(chunks[i + 1].metadata, chunk.metadata, prepend=True)
                else:
                    cleaned.append(chunk)
            else:
                cleaned.append(chunk)

        # Pass 2: anything still undersized gets rebalanced against a neighbor.
        for i, chunk in enumerate(cleaned):
            if len(chunk.page_content) >= self.__min_parent_size or len(cleaned) == 1:
                continue
            if i < len(cleaned) - 1:
                cleaned[i], cleaned[i + 1] = self.__rebalance_pair(chunk, cleaned[i + 1])
            else:
                cleaned[i - 1], cleaned[i] = self.__rebalance_pair(cleaned[i - 1], chunk)
        return cleaned

    def __create_child_chunks(self, all_parent_pairs, all_child_chunks, parent_chunks, doc_path, source_name):
        for i, p_chunk in enumerate(parent_chunks):
            parent_id = f"{doc_path.stem}_p{i}"
            p_chunk.metadata.update({"source": source_name, "parent_id": parent_id})
            all_parent_pairs.append((parent_id, p_chunk))
            all_child_chunks.extend(self.__child_splitter.split_documents([p_chunk]))
```

Pipeline order: `MarkdownHeaderTextSplitter` (split on H1/H2/H3) → `__merge_small_parents` (greedily grow undersized header sections) → `__split_large_parents` (character-split oversized ones) → `__clean_small_chunks` (absorb or rebalance remaining undersized ones) → assign `parent_id = "{doc_stem}_p{i}"` and `source` metadata → `RecursiveCharacterTextSplitter` cuts each parent into fixed-size children.

### 7.3 Content-Hash Change Detection (`utils.py` + `core/document_manager.py`)

```python
# utils.py
def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
```

In `DocumentManager.add_documents`, per uploaded file:
1. Compute `source_hash = file_sha256(source_path)`; sidecar `hash_path = md_path.with_suffix(".md.sha256")`.
2. If `md_path` and `hash_path` both exist and the stored hash matches → **skip** (unchanged).
3. Otherwise (new file, or same filename with changed content):
   - **First purge stale artifacts** for this document name — this step is essential and easy to miss: `collection.add_documents(...)` assigns fresh random point IDs, so re-adding without purging leaves the *old* chunks as permanent stale duplicates in the vector index.
     ```python
     self.rag_system.vector_db.delete_by_source(self.rag_system.collection_name, source_path.name)
     for old_parent_file in Path(config.PARENT_STORE_PATH).glob(f"{doc_name}_p*.json"):
         old_parent_file.unlink(missing_ok=True)
     ```
   - Convert (PDF→MD with `overwrite=True`, or copy `.md` directly), chunk, save parents, add children to Qdrant, then write the new hash to the sidecar file.
4. On any exception mid-processing: roll back partial parent files, delete the (possibly partial) markdown file, and delete the hash sidecar so a retry isn't wrongly treated as "unchanged."

`VectorDbManager.delete_by_source` (the purge primitive):
```python
def delete_by_source(self, collection_name, source_name):
    if not self.__client.collection_exists(collection_name):
        return
    self.__client.delete(
        collection_name=collection_name,
        points_selector=qmodels.FilterSelector(
            filter=qmodels.Filter(
                must=[qmodels.FieldCondition(key="metadata.source", match=qmodels.MatchValue(value=source_name))]
            )
        ),
    )
```
This relies on `langchain-qdrant`'s default payload layout: document metadata is stored nested under a `"metadata"` payload key (and content under `"page_content"`), so `metadata.source` is a valid Qdrant filter path.

### 7.4 Storage Schemas

**Qdrant collection** (`document_child_chunks`): one dense vector (named default, size = embedding model's output dim, cosine distance) + one named sparse vector (`"sparse"`, BM25 via `FastEmbedSparse`). `RetrievalMode.HYBRID` fuses both at query time. Payload per point: `page_content` (child chunk text) + `metadata: {source, parent_id, ...header metadata}`.

**Parent store**: flat directory of JSON files, one per parent chunk, named `{doc_stem}_p{i}.json`:
```json
{"page_content": "...", "metadata": {"source": "manual.pdf", "parent_id": "manual_p0", "...header keys...": "..."}}
```
(`db/parent_store_manager.py` — `save`, `save_many`, `load`, `load_content`, `load_content_many` (sorted by trailing `_p{N}` index via regex), `list_sources` (dedup by `metadata.source` across all files), `delete_many`, `clear_store`.)

---

## 8. LangGraph State Schemas (`rag_agent/graph_state.py`, full file)

```python
from typing import List, Annotated, Set
from langgraph.graph import MessagesState
import operator

def accumulate_or_reset(existing: List[dict], new: List[dict]) -> List[dict]:
    if new and any(item.get('__reset__') for item in new):
        return []
    return existing + new

def set_union(a: Set[str], b: Set[str]) -> Set[str]:
    return a | b

def append_unique(existing: List[str], new: List[str]) -> List[str]:
    return list(dict.fromkeys(existing + new))

class State(MessagesState):
    """State for main agent graph"""
    questionIsClear: bool = False
    conversation_summary: str = ""
    originalQuery: str = ""
    pendingQuery: str = ""
    pendingClarifications: List[str] = []
    rewrittenQuestions: List[str] = []
    agent_answers: Annotated[List[dict], accumulate_or_reset] = []

class AgentState(MessagesState):
    """State for individual agent subgraph"""
    question: str = ""
    question_index: int = 0
    context_summary: str = ""
    retrieval_keys: Annotated[Set[str], set_union] = set()
    retrieved_contexts: Annotated[List[str], append_unique] = []
    final_answer: str = ""
    agent_answers: List[dict] = []
    tool_call_count: Annotated[int, operator.add] = 0
    iteration_count: Annotated[int, operator.add] = 0
```

`QueryAnalysis` (`rag_agent/schemas.py`, structured output for query rewriting):
```python
from typing import List
from pydantic import BaseModel, Field

class QueryAnalysis(BaseModel):
    is_clear: bool = Field(description="Indicates if the user's question is clear and answerable.")
    questions: List[str] = Field(description="List of rewritten, self-contained questions.")
    clarification_needed: str = Field(description="Explanation if the question is unclear.")
```

---

## 9. Graph Topology (`rag_agent/graph.py`, `nodes.py`, `edges.py`)

### 9.1 Agent Subgraph (one instance runs per rewritten sub-question, in parallel)

Nodes: `orchestrator` → (`tools` | `fallback_response` | `collect_answer`) → `should_compress_context` → (`compress_context` → back to `orchestrator`) → ... → `collect_answer` → END.

```python
agent_builder = StateGraph(AgentState)
agent_builder.add_node("orchestrator", partial(orchestrator, llm_with_tools=llm_with_tools))
agent_builder.add_node("tools", ToolNode(tools_list))
agent_builder.add_node("compress_context", partial(compress_context, llm=llm))
agent_builder.add_node("fallback_response", partial(fallback_response, llm=llm))
agent_builder.add_node("should_compress_context", should_compress_context)
agent_builder.add_node("collect_answer", collect_answer)

agent_builder.add_edge(START, "orchestrator")
agent_builder.add_conditional_edges("orchestrator", route_after_orchestrator_call,
    {"tools": "tools", "fallback_response": "fallback_response", "collect_answer": "collect_answer"})
agent_builder.add_edge("tools", "should_compress_context")
agent_builder.add_edge("compress_context", "orchestrator")
agent_builder.add_edge("fallback_response", "collect_answer")
agent_builder.add_edge("collect_answer", END)
agent_subgraph = agent_builder.compile()
```

**`orchestrator(state, llm_with_tools)`** — first call injects the question + a hard instruction to search first (`"YOU MUST CALL 'search_child_chunks' AS THE FIRST STEP..."`); subsequent calls just replay the accumulated message list. Tags its own AI response with `name="agent_response"` so it's excluded from "plain conversation" message filtering elsewhere. Tracks `tool_call_count` and `iteration_count` (both `operator.add` reducers, so they accumulate across every orchestrator visit).

**`route_after_orchestrator_call(state)`**:
```python
if not tool_calls: return "collect_answer"
if iteration >= MAX_ITERATIONS or tool_count > MAX_TOOL_CALLS: return "fallback_response"
return "tools"
```
Note: a final answer *at* the iteration boundary is still accepted (no tool calls in the response ⇒ `collect_answer` regardless of counters) — only pending *new* tool calls get vetoed by the budget check.

**`should_compress_context(state)`** — a `Command`-returning node (can both update state and pick the next node in one step). Extracts the most recent tool-call's search query / parent_id(s) into `retrieval_keys` (prefixed `"search::"` / `"parent::"`, deduplicated across the whole subgraph run so the orchestrator prompt's "don't repeat" instruction can be enforced), recomputes `retrieved_contexts` via `_retrieval_contexts(messages)`, then decides:
```python
current_tokens = estimate_context_tokens(messages) + estimate_context_tokens([HumanMessage(content=context_summary)])
max_allowed = BASE_TOKEN_THRESHOLD + int(current_token_summary * TOKEN_GROWTH_FACTOR)
goto = "compress_context" if current_tokens > max_allowed else "orchestrator"
```
(`current_token_summary` = token count of the *existing* compressed summary alone — so the allowed budget grows slightly each time a summary already exists, via `TOKEN_GROWTH_FACTOR`.)

**`compress_context(state, llm)`** — serializes all non-system messages (assistant text + tool call names/args, tool results) into one text blob, prepends any prior compressed summary, asks the LLM (via `get_context_compression_prompt()`) to produce a fresh Markdown summary, appends a machine-generated "Already executed (do NOT repeat)" block listing every parent id / search query seen so far, then **deletes** all those raw messages (`RemoveMessage`) — the agent's working context resets to just the new summary.

**`fallback_response(state, llm)`** — triggered only when the tool-call budget is exceeded. Collects every *unique* raw `ToolMessage.content` seen (JSON strings, un-parsed — the LLM reads them as-is) plus the compressed summary, and asks the LLM to answer with whatever evidence exists, via `get_fallback_response_prompt()`.

**`collect_answer(state)`** — accepts the last AI message as the final answer only if it has content and made no further tool calls; otherwise substitutes `"Unable to generate an answer."`. Packages `{index, question, answer, contexts}` into `agent_answers` (`contexts` is populated but not currently read downstream in the live app — it exists for future eval/tracing use).

### 9.2 Main Graph

```python
graph_builder = StateGraph(State)
graph_builder.add_node("summarize_history", partial(summarize_history, llm=llm))
graph_builder.add_node("rewrite_query", partial(rewrite_query, llm=llm))
graph_builder.add_node("request_clarification", request_clarification)
graph_builder.add_node("agent", agent_subgraph)
graph_builder.add_node("aggregate_answers", partial(aggregate_answers, llm=llm))

graph_builder.add_edge(START, "summarize_history")
graph_builder.add_edge("summarize_history", "rewrite_query")
graph_builder.add_conditional_edges("rewrite_query", route_after_rewrite)
graph_builder.add_edge("request_clarification", "rewrite_query")
graph_builder.add_edge(["agent"], "aggregate_answers")   # fan-in: waits for ALL spawned agents
graph_builder.add_edge("aggregate_answers", END)

conn = sqlite3.connect(config.CHECKPOINT_DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn)
agent_graph = graph_builder.compile(checkpointer=checkpointer, interrupt_before=["request_clarification"])
```

**`summarize_history(state, llm)`** — keeps only the most recent `MAIN_HISTORY_MESSAGES_TO_KEEP - 1` "plain" (non-tool, non-named) messages verbatim; everything older is merged into `conversation_summary` via `get_conversation_summary_prompt()` and then `RemoveMessage`-deleted from graph state. Also resets `agent_answers` to `[]` at the start of every new turn via the `{"__reset__": True}` sentinel understood by `accumulate_or_reset`.

**`rewrite_query(state, llm)`** — builds context from `conversation_summary` + recent plain conversation (excluding the unresolved query itself during a clarification round-trip) + either the plain current query, or (if `pendingQuery` is set) the original unresolved query plus every clarification reply collected so far. Calls `llm.with_structured_output(QueryAnalysis)` with `get_rewrite_query_prompt()`. If clear: sets `rewrittenQuestions`, clears pending state. If unclear: stores/extends `pendingQuery`/`pendingClarifications`, appends an `AIMessage(name="clarification")` asking the user for more detail.

**`route_after_rewrite(state)`**:
```python
if not state["questionIsClear"]: return "request_clarification"
return [Send("agent", {"question": q, "question_index": i, "messages": []}) for i, q in enumerate(state["rewrittenQuestions"])]
```
This `Send`-based fan-out is what spawns N parallel agent subgraphs, one per rewritten sub-question (map-reduce).

**`request_clarification(state)`** — a no-op node; its only purpose is to be the graph's `interrupt_before` target, pausing execution so the app can surface the clarification question and wait for the next user message before resuming (see §11.3).

**`aggregate_answers(state, llm)`** — prunes history the same way `summarize_history` does, then (if any `agent_answers` exist) sorts them by `index`, formats them as `"Retrieved response N:\n{answer}"` blocks, and asks the LLM to synthesize one final answer via `get_aggregation_prompt()`.

### 9.3 Compression Trigger Formula (recap)

```
current_tokens = tokens(all subgraph messages) + tokens(existing context_summary)
max_allowed    = BASE_TOKEN_THRESHOLD + floor(tokens(existing context_summary) * TOKEN_GROWTH_FACTOR)
compress if current_tokens > max_allowed
```
Token counting: `tiktoken.encoding_for_model("gpt-4")` (falls back to `cl100k_base`, then to `len(text)//4` if `tiktoken` itself is unavailable) — an approximation regardless of actual LLM provider, acceptable since it only gates a compression heuristic, not billing.

---

## 10. System Prompts (`rag_agent/prompts.py`, all 6, verbatim)

These encode most of the agent's actual behavior — reproduce exactly, tune later if desired.

```python
def get_conversation_summary_prompt() -> str:
    return """## Role
You are a compact memory manager for a retrieval-augmented chat assistant.

## Context
The input contains an existing rolling summary plus older user/assistant messages that will be removed from raw chat history.

## Instructions
- Merge the existing summary with the new older messages.
- Preserve context needed for future follow-up questions: topics, user preferences, important facts, unresolved questions, and referenced source file names.
- Discard greetings, tool calls, tool outputs, formatting chatter, duplicate details, and resolved misunderstandings.
- Keep the summary compact: 30-70 words unless more detail is essential.

## Output
Return exactly one merged summary and nothing else.
Do not include labels such as "Updated summary:", "Previous summary:", or "New messages:".
Do not include both old and new summaries.
If there is no meaningful context, return an empty string.
"""

def get_rewrite_query_prompt() -> str:
    return """## Role
You are a query rewriting specialist for document retrieval in a RAG system.

## Instructions
- Rewrite the current query so it is clear, self-contained, and useful for retrieval.
- Use the conversation summary and recent conversation only to resolve vague follow-ups that refer to prior context.
- When an unresolved query and one or more user clarifications are provided, combine all of them into one self-contained retrieval query.
- If the query is a follow-up, integrate only the minimal context needed to make it self-contained.
- Preserve product names, file names, versions, acronyms, numbers, and technical terms exactly.
- If the user asks about a named topic, product, file, acronym, term, or concept, treat the question as clear even if it is new.
- Standalone named terms, acronyms, or concepts are valid retrieval queries; do not require prior conversation context.
- Split only truly separate information needs, with a maximum of 3 rewritten questions.

## Clarification Boundary
Mark the query unclear only when it depends on an unresolved reference such as "it", "that", "this file", or "the previous one".
Do not mark a query unclear because the topic was not mentioned earlier.
Do not ask the user whether a new acronym or term is a typo; preserve it and search for it.

## Constraints
Do not add facts, expand acronyms, invent context, or broaden the user's meaning.
"""

def get_orchestrator_prompt() -> str:
    return """## Role
You are a document-grounded research assistant for an agentic RAG system. Your job is to answer using retrieved document evidence, not general knowledge.

## Available Context
- Current user question
- Optional compressed context from prior retrieval steps
- Tools for searching child chunks and loading full parent chunks

## Tool Guidance
- Search documents before answering unless compressed context already contains enough evidence.
- Use 'search_child_chunks' for missing or uncovered parts of the question.
- If searched or retrieved context is not useful, use the tools again with a different, simpler query or a more relevant parent chunk.
- Continue tool use until the available evidence is enough, tools stop adding useful information, or the operation limit is reached.
- Do not repeat search queries or parent IDs listed in compressed context.
- Do not retrieve the same parent ID twice.

## Response Framework
1. Check compressed context for already-known evidence and already-used searches or parents.
2. Search for missing evidence.
3. Retrieve parent chunks only when child excerpts are relevant but too fragmented.
4. Answer using the exact terms and scope in the retrieved evidence.
5. If evidence is incomplete, state the specific gap.

## Output
- Start directly with the substantive answer. Do not start with generic headings such as "Answer", "Final answer", or "Response".
- Provide the direct answer plus the key supporting details from retrieved evidence; avoid one-sentence fragments unless only one fact is available.
- Do not mention internal tool calls or reasoning.
- When sources exist, end with a Sources section in exactly this format:
  Sources:
  - filename.ext
- Put each source filename on its own bullet line. Never write sources inline, such as "Sources: filename.pdf".
- Do not invent or infer source filenames.
- Strip descriptions after file names, including text in parentheses.
"""

def get_fallback_response_prompt() -> str:
    return """## Role
You are a constrained evidence synthesizer for a retrieval-augmented assistant after the research loop reached its limit.

## Available Context
- Compressed Research Context from earlier retrieval steps
- Retrieved Data from current tool outputs

## Instructions
- Use only explicit facts from the provided context.
- Start directly with the substantive answer. Do not start with generic headings such as "Answer", "Final answer", or "Response".
- Prefer current Retrieved Data over compressed context if they conflict.
- If the answer is incomplete, mention only the missing parts that matter to the user query.
- Do not describe the retrieval process, limits, or internal reasoning.
- Be concise: answer in 1-3 short paragraphs or up to 5 bullets unless the user asks for detail.
- Provide the direct answer plus the key supporting details from retrieved evidence; avoid one-sentence fragments unless only one fact is available.
- End with a Sources section only when actual source file names are explicitly present in the context.
- Use exactly this format:
  Sources:
  - filename.ext
- Put each source filename on its own bullet line. Never write sources inline, such as "Sources: filename.pdf".
- Include only bare file names with extensions such as .pdf, .docx, .txt, or .md.
- Do not invent or infer source filenames.
"""

def get_context_compression_prompt() -> str:
    return """## Role
You are a research context compressor for an agentic RAG system.

## Instructions
- Keep only facts relevant to answering the user question.
- Preserve exact names, figures, versions, technical terms, configuration details, and source file names.
- Remove duplicates, tool chatter, search query wording, parent IDs, chunk IDs, and other internal identifiers.
- Organize findings by source file. Each source section heading must be the real filename found in retrieved data.
- Add a Gaps section only for missing information relevant to the question.
- Target 400-600 words. If there is too much content, keep the most answer-critical facts.

## Output
Return only Markdown in this structure:
# Research Context Summary

## Focus
[Brief technical restatement of the question]

## Structured Findings
For each source file, add a level-3 heading with its real filename and bullet the directly relevant facts below it.

## Gaps
- Missing or incomplete aspects
"""

def get_aggregation_prompt() -> str:
    return """## Role
You are a final-answer synthesizer for a retrieval-augmented assistant.

## Instructions
- Use only information present in the retrieved answers.
- Start directly with the substantive answer. Do not start with generic headings such as "Answer", "Final answer", or "Response".
- Preserve important names, numbers, versions, examples, and definitions.
- Do not expand acronyms or interpret terms unless the sources do it.
- If answers conflict, mention the conflict plainly.
- Be concise: answer in 1-3 short paragraphs or up to 5 bullets unless the user asks for detail.
- Provide the direct answer plus the key supporting details from retrieved evidence; avoid one-sentence fragments unless only one fact is available.
- End with a Sources section only when actual source file names are explicitly present in the retrieved answers.
- Use exactly this format:
  Sources:
  - filename.ext
- Put each source filename on its own bullet line. Never write sources inline, such as "Sources: filename.pdf".
- Include only bare file names with extensions such as .pdf, .docx, .txt, or .md.
- Do not invent or infer source filenames.
- If no useful information is available, say: "I couldn't find any information to answer your question in the available sources."
"""
```

---

## 11. Retrieval Tools (`rag_agent/tools.py`)

Both tools are created per-request via `ToolFactory(collection).create_tools()` (needs a live `QdrantVectorStore` collection handle) and bound to the LLM with `llm.bind_tools([search_tool, retrieve_tool])`.

### 11.1 `search_child_chunks(query: str, limit: int = DEFAULT_RETRIEVAL_K) -> str`
- Runs `collection.similarity_search(query, k=limit, score_threshold=RETRIEVAL_SCORE_THRESHOLD)` (hybrid dense+sparse).
- **No results:** returns the plain sentinel string `"NO_RELEVANT_CHUNKS"`.
- **Results:** returns a JSON array string: `[{"parent_id": "...", "source": "...", "content": "..."}, ...]`.
- **Exception:** returns `"RETRIEVAL_ERROR: {message}"`.

### 11.2 `retrieve_parent_chunks(parent_id: str) -> str`
- Loads `parent_store/{parent_id}.json` via `ParentStoreManager.load_content`.
- **Success:** JSON object string: `{"parent_id": "...", "source": "...", "content": "..."}`.
- **Missing file:** currently raises inside `ParentStoreManager.load()` (unconditional `read_text()`) and is caught by the tool's own `except`, surfacing as `"PARENT_RETRIEVAL_ERROR: [Errno 2] No such file or directory: ..."` rather than the nominal `"NO_PARENT_DOCUMENT"` sentinel the code also defines — a small pre-existing inconsistency, harmless (both are correctly treated as "ignored" sentinels downstream) but worth tightening in a rebuild by having `ParentStoreManager.load` return `None`/raise a dedicated `FileNotFoundError` subtype the tool can distinguish cleanly.

### 11.3 Sentinel/error prefixes recognized downstream (`_retrieval_contexts` in `nodes.py`)
```python
("NO_RELEVANT_CHUNKS", "NO_PARENT_DOCUMENT", "RETRIEVAL_ERROR:", "PARENT_RETRIEVAL_ERROR:")
```
Any `ToolMessage.content` starting with one of these is excluded from the accumulated `retrieved_contexts`; everything else is `json.loads`'d (list or single object) and reformatted into `"Parent ID: ...\nFile Name: ...\nContent: ..."` strings for downstream consumers, with a raw-string fallback if JSON parsing fails.

---

## 12. Conversation Memory & Sessions

- **Per-session isolation:** `RAGSystem` is a single shared instance (holds the expensive stuff: vector DB client, embeddings, compiled graph). The **thread id** is *not* stored on it — it's minted fresh per browser session via Gradio's `gr.State` + `demo.load(fn=rag_system.create_thread_id, outputs=session_state)`, and threaded explicitly through `chat(msg, hist, thread_id)` / `clear_session(thread_id)`. This is what keeps concurrent users' conversations from interleaving.
- **Persistence:** `SqliteSaver` over a `sqlite3.Connection(check_same_thread=False)` at `config.CHECKPOINT_DB_PATH` — conversation state (LangGraph checkpoints) survives process restarts. Schema (`checkpoints`, `writes` tables) auto-creates on first write.
- **Rolling summarization:** see §9.2 `summarize_history` — keeps the last `MAIN_HISTORY_MESSAGES_TO_KEEP - 1` plain messages verbatim, folds everything older into `conversation_summary`.
- **Human-in-the-loop clarification:** `agent_graph.compile(..., interrupt_before=["request_clarification"])` pauses the graph before that no-op node. The chat loop (`ChatInterface.chat`) checks `agent_graph.get_state(config).next` — if truthy, the next user message is injected via `update_state(...)` and the graph resumed with `stream(None, config, ...)` instead of a fresh `stream_input`.

---

## 13. UI (`ui/gradio_app.py`)

`gr.Blocks(title="Agentic RAG")` with two tabs:

**"Documents" tab:** `gr.File` (multi-file drop, PDF/MD) → "Add Documents" button → `DocumentManager.add_documents(files, progress_callback=...)` reporting `(added, skipped)` via `gr.Info`; a read-only `gr.Textbox` listing indexed sources (`DocumentManager.get_markdown_files()`, backed by `ParentStoreManager.list_sources()`); "Refresh" and "Clear All" (`DocumentManager.clear_all()` — drops the Qdrant collection, wipes `markdown_dir` and the parent store, recreates an empty collection) buttons.

**"Chat" tab:**
```python
session_thread_id = gr.State()
chatbot = gr.Chatbot(height=720, avatar_images=(None, ".../chatbot_avatar.png"), layout="bubble")
chatbot.clear(clear_chat_handler, inputs=[session_thread_id], outputs=[session_thread_id])
gr.ChatInterface(fn=chat_handler, chatbot=chatbot, additional_inputs=[session_thread_id])
demo.load(fn=rag_system.create_thread_id, outputs=session_thread_id)
```
`chat_handler` is a generator forwarding to `ChatInterface.chat`, which streams `stream_mode="messages"` graph output and reshapes it into Gradio "metadata" messages: `summarize_history`/`rewrite_query` tokens become a live-updating collapsible block (title from `SYSTEM_NODE_CONFIG`; `rewrite_query`'s partial JSON is parsed opportunistically via a regex `\{.*\}` + `json.loads` to render "✅ Query is clear" / "❓ Query is unclear" + any clarification live as it streams); tool calls become collapsible `"Running \`{name}\`..."` blocks later filled with pretty-printed JSON (or raw text for sentinels); only `aggregate_answers`' token stream is appended to the final plain chat bubble.

`ui/css.py` exports a `custom_css` string passed to `demo.launch(css=custom_css)` in `app.py`.

---

## 14. LLM Provider Abstraction (`core/rag_system.py`)

```python
llm_kwargs = {"temperature": config.LLM_TEMPERATURE}
if config.LLM_PROVIDER == "ollama":
    llm_kwargs["seed"] = config.LLM_SEED  # not all providers accept `seed`
llm = init_chat_model(config.LLM_MODEL, model_provider=config.LLM_PROVIDER, **llm_kwargs)
```
`init_chat_model` (from the `langchain` package) dispatches to the correct provider integration package based on `model_provider`, importing it lazily — so only the packages for providers you actually use need to be installed. Switching providers is a `config.py`-only change (see §6).

---

## 15. Observability (`core/observability.py`)

```python
class Observability:
    def __init__(self):
        self._enabled = config.LANGFUSE_ENABLED
        if not self._enabled: return
        if not config.LANGFUSE_PUBLIC_KEY or not config.LANGFUSE_SECRET_KEY:
            self._enabled = False; return
        from langfuse import get_client
        from langfuse.langchain import CallbackHandler
        self._client = get_client()
        if self._client.auth_check():
            self._handler = CallbackHandler()
        else:
            self._enabled = False

    def get_handler(self): return self._handler
    def flush(self):
        if self._client is not None:
            try: self._client.flush()
            except Exception: pass
```
`RAGSystem.get_config(thread_id)` attaches `cfg["callbacks"] = [handler]` when enabled, so every LLM call, tool call, and graph transition is traced automatically via LangChain's callback system — no per-call instrumentation needed elsewhere.

Optional `EXECUTION_LOGGING_ENABLED` in `config.py` (via `core/execution_logger.py`'s `logged_node` wrapper applied to every node in `graph.py`) prints colored, truncated before/after state previews of every node execution to the terminal — useful for local debugging without a Langfuse account.

---

## 16. Docker Deployment (`Dockerfile`)

Single-container pattern: base `python:3.13`, installs Ollama via the official install script, `pip install -r requirements.txt`, then a generated `start.sh` that backgrounds `ollama serve`, waits for it to respond, pulls `granite4.1:8b`, and runs `python project/app.py`. Runs as non-root user `user` (uid 1000). Exposes `7860`. `GRADIO_SERVER_NAME=0.0.0.0` / `GRADIO_SERVER_PORT=7860` env vars make Gradio bind externally inside the container.

---

## 17. Build Order (rebuilding from scratch)

Respect this dependency order — each layer only needs the ones below it:

1. `config.py` (no dependencies)
2. `utils.py` (needs `config`)
3. `document_chunker.py` (needs `config`)
4. `db/vector_db_manager.py`, `db/parent_store_manager.py` (need `config`, `utils`)
5. `rag_agent/schemas.py`, `rag_agent/graph_state.py` (standalone)
6. `rag_agent/prompts.py` (standalone)
7. `rag_agent/tools.py` (needs `db/parent_store_manager.py`, `config`, an execution logger stub)
8. `rag_agent/nodes.py`, `rag_agent/edges.py` (need everything above)
9. `rag_agent/graph.py` (assembles 5-8)
10. `core/observability.py`, `core/execution_logger.py` (standalone utilities)
11. `core/rag_system.py` (wires 4, 9, 10 + `init_chat_model`)
12. `core/document_manager.py` (needs `rag_system`, `document_chunker`, `utils`)
13. `core/chat_interface.py` (needs `rag_system`)
14. `ui/gradio_app.py`, `ui/css.py` (need `core/*`)
15. `app.py` (entry point, needs `ui/gradio_app.py`)

At each layer, the pure-logic pieces (3, 5, 6) can be unit-tested with no LLM/network calls; layers 4/7 can be tested against a real Qdrant with a fake `Embeddings` subclass (no model download needed) — see §18.

---

## 18. Verification Checklist

Reuse these to confirm a from-scratch rebuild actually matches this spec's behavior, without needing Ollama running or a real embedding model downloaded:

- **Chunking correctness:** feed `DocumentChunker.create_chunks_single` a synthetic Markdown file with headers of varying section sizes; assert no output chunk exceeds `MAX_PARENT_SIZE` and every parent has `source`/`parent_id` metadata.
- **Hybrid search + purge:** build a `QdrantVectorStore` with a trivial `Embeddings` stub (`embed_documents`/`embed_query` returning fixed-length fake vectors) against a temp `QdrantClient(path=...)`, add documents tagged with different `source` metadata, call `delete_by_source`, and confirm only the targeted source's points are removed.
- **Content-hash re-indexing:** upload the same file twice (expect skip), then a modified file under the same name (expect reprocess with zero stale chunks/parents remaining from the old version).
- **Tool JSON contracts:** invoke both tools directly against a fake-embedded collection; assert `search_child_chunks` returns a JSON array of `{parent_id, source, content}`, `retrieve_parent_chunks` a single such object, and that `_retrieval_contexts` correctly reconstructs readable strings from both while filtering out sentinel/error strings.
- **Checkpointer persistence:** construct `SqliteSaver` over a real sqlite file, call `.setup()`, confirm the `checkpoints`/`writes` tables exist.
- **Per-session isolation:** open two browser sessions against the running app; confirm independent thread ids (`demo.load` fires once per session) and that clearing one session's chat doesn't affect the other's.
- **End-to-end (needs Ollama or a cloud key):** upload a real PDF, ask a question, confirm a "Sources:" section citing the correct filename; ask a multi-part question and confirm parallel sub-agents fan out (visible via `EXECUTION_LOGGING_ENABLED=True` terminal logs or Langfuse traces).

---

## 19. Relationship to `IMPROVEMENTS.md`

- **§1 (bug fixes)** — fully reflected as the baseline in this document already (config-driven provider, SQLite checkpointer, per-session threads, content-hash re-indexing with stale-chunk purge, JSON tool outputs). Nothing left to do there.
- **§2 (scaffolding: pyproject/uv, ruff, mypy, CI, pydantic-settings config, docker-compose)** — not reflected here; still open.
- **§3 (RAG-quality: `init_chat_model` refactor is done via §14 above; reranker, page-level citations, real Qdrant server, multi-format ingestion, adaptive routing, self-grading)** — only the provider-init item is done; the rest are still open.
- **§4 (new UX features: live settings panel, graph visualization, retrieval-score display, token/cost tracking, CLI mode, conversation export)** — none implemented; still open.
- **§5 (testing)** — no test suite exists yet in the repo itself; the verification snippets in §18 above are ad hoc, not a checked-in `tests/` directory.

A from-scratch rebuild should treat **this file** as the target behavior for a first working version, then layer in `IMPROVEMENTS.md` §2–§5 (and any of §3/§4's remaining items) on top, in whatever order suits the rebuild's own priorities — §6 of `IMPROVEMENTS.md` has a suggested order if useful.
