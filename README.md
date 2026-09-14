# self-correcting-rag

An agentic, **self-correcting** RAG (Retrieval-Augmented Generation) system: you give it your PDFs
and Markdown files, and it answers questions about them by *researching* — searching, pulling in
wider context when an excerpt is too fragmented, grading its own evidence, and retrying when that
evidence doesn't actually support an answer.

It is built on [LangGraph](https://docs.langchain.com/oss/python/langgraph) (the agent is an
explicit state machine, not one giant prompt) and [Qdrant](https://qdrant.tech) (hybrid dense +
sparse retrieval).

> **Status: under construction, in public.** This repository is being built in deliberate stages so
> that each piece can be understood before the next is added. See **[PLAN.md](PLAN.md)** for what
> exists today and what is still to come, and **[docs/stages/](docs/stages/)** for a written
> explanation of every stage.

## Why this repo exists

This is a learning build. The goal is not only a working system but a codebase someone can *read*:
strictly layered, config-driven, fully typed, and tested where the logic is subtle. The design
documents that drive it are:

| Document | What it is |
|---|---|
| [`BLUEPRINT.md`](BLUEPRINT.md) | The technical specification of the target system |
| [`EXPLAINED.md`](EXPLAINED.md) | A plain-language conceptual tour of how agentic RAG works |
| [`IMPROVEMENTS.md`](IMPROVEMENTS.md) | The roadmap of upgrades layered on top of the baseline |
| [`PLAN.md`](PLAN.md) | Live build progress — what is done, what is next |

## Requirements

- Python 3.12 (managed automatically by [`uv`](https://docs.astral.sh/uv/))
- An API key for **Google Gemini** or **Groq** (both have free tiers)

No GPU is needed. Embeddings run on CPU through FastEmbed's ONNX runtime, so PyTorch is never
installed.

## Getting started

```bash
uv sync --all-extras          # create the virtualenv and install everything
cp .env.example .env          # then add your API key
uv run self-rag config        # print the resolved configuration
```

Run the checks the CI runs:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

## License

MIT
