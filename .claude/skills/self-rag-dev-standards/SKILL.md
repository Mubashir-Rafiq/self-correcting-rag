---
name: self-rag-dev-standards
description: Engineering standards for the self-correcting-rag project (an agentic RAG system built on LangGraph/Qdrant — see BLUEPRINT.md, EXPLAINED.md, IMPROVEMENTS.md at repo root). Use whenever writing, generating, scaffolding, implementing, refactoring, debugging, or reviewing ANY code in this repository — new modules, features, bug fixes, config changes, prompts, graph nodes/edges, tests, or tooling. Keeps the codebase modular, layered, config-driven, typed, tested, observable, and easy to debug/change/learn. Load before starting implementation work, not just when explicitly asked for "architecture."
---

# Self-RAG Engineering Standards

This project is being (re)built from a full spec in `BLUEPRINT.md` (target behavior),
`IMPROVEMENTS.md` (what to layer on top, in priority order), and `EXPLAINED.md` (plain-language
tour). Treat those three files as living documents, not one-time reading — re-check them before
adding anything non-trivial, and update them when a change alters the architecture they describe.

Apply the rules below to every piece of code you write or touch in this repo. They exist to make
one outcome true: **a stranger can open any single file, understand it without reading the whole
repo, change it safely, and have a test catch it if they broke something.**

## 1. Respect the layering — never invert a dependency

The project has a strict dependency order (`BLUEPRINT.md` §17):

```
config.py → utils.py / document_chunker.py → db/ → rag_agent/{schemas,graph_state,prompts,tools} →
rag_agent/{nodes,edges} → rag_agent/graph.py → core/{observability,execution_logger} →
core/rag_system.py → core/document_manager.py / core/chat_interface.py → ui/ → app.py
```

- A module may only import from layers at or below it in this chain. If you find yourself
  importing `core/` from `rag_agent/`, or `ui/` from `core/`, stop — that's a layering violation,
  not a shortcut.
- Every setting lives in one place (`config.py`, or its `pydantic-settings` successor per
  `IMPROVEMENTS.md` §2) — never hardcode a model name, path, threshold, or provider string inline
  where a config constant already exists or should exist. If a value might plausibly need tuning
  without a code change, it belongs in config, not in the function body.
- Provider/backend selection (LLM provider, embedding model, vector DB backend) must stay a
  config-only change — no `if provider == "openai": ...` branching hand-rolled in application code
  when a factory (`init_chat_model`, etc.) already does it.

## 2. Make illegal states hard to reach

- Use Pydantic models / `TypedDict` / dataclasses for anything crossing a boundary (LLM structured
  output, tool inputs/outputs, graph state) — never pass around bare dicts with string keys typed
  from memory.
- Tool outputs that can represent "no result" or "error" must use one **consistent, centralized**
  representation (a sentinel constant or a typed result), never a mix of magic strings improvised
  per call site. `BLUEPRINT.md` §11.2 documents a real inconsistency here
  (`retrieve_parent_chunks` meant to return `"NO_PARENT_DOCUMENT"` but actually raises and gets
  caught as a generic error) — don't reproduce that pattern; fix it forward if you touch that code.
- Validate only at actual system boundaries (file I/O, user input, LLM output parsing, external
  API responses). Don't add defensive checks for states that internal code already guarantees
  can't happen — that's noise, not safety.
- Fail loud, not silent: if a step is skipped or falls back, that must be visible in a return
  value, a log line, or a raised exception — never a swallowed exception that leaves the caller
  guessing.

## 3. Keep nodes/functions small, pure where possible, and independently testable

- Pure logic (chunking math, routing conditions, token-threshold formulas, prompt string
  builders) must have zero LLM/network/DB calls, so it's testable with plain `pytest` — no mocks
  needed. Keep this logic in its own function/module, separate from the I/O-performing code that
  calls it.
- Each LangGraph node/edge function should do one job and be callable/testable in isolation by
  constructing a minimal state dict — don't let a node reach into unrelated parts of the graph.
- When a function's behavior depends on a limit, threshold, or budget (tool-call counts, token
  counts, retry counts), make the arithmetic a named, tested function rather than an inline
  expression duplicated at each call site.

## 4. Tooling — modern Python project hygiene (`IMPROVEMENTS.md` §2)

Bring these in as the project is scaffolded, and keep them green as you go — don't let them rot:

- `pyproject.toml` + `uv` for dependencies/lockfile (`uv.lock`), not a bare `requirements.txt`.
- `ruff` for lint + format (run it; don't hand-format).
- `mypy` (or `pyright`) type checking — at minimum on `rag_agent/` and `db/`.
- `pytest` for tests — put pure-logic tests (chunker, edges, token math) in `tests/` as you write
  the logic they cover, not as an afterthought.
- `pre-commit` running ruff + mypy, and a GitHub Actions CI workflow running the same checks plus
  `pytest` on every push — set this up early so it catches regressions from the start, not after
  the fact.
- Prefer `pydantic-settings` over a bare `.py` config module with scattered `os.environ.get(...)`
  calls, once the config surface grows past the current `Langfuse`-only env usage.

## 5. Debuggability & observability

- New LLM calls or graph nodes should be traceable via the existing `Observability`
  (`core/observability.py`, Langfuse) and `execution_logger.py` hooks — wire new nodes into
  whatever wrapping mechanism already applies to node execution rather than adding ad hoc
  `print()`s.
- Log/trace at decision points (routing choices, budget/threshold triggers, fallbacks) — these are
  exactly the places where "why did it do that?" questions come up later.
- Prefer structured, parseable outputs (JSON) over hand-formatted strings anywhere a downstream
  consumer has to read the value back apart from a human — this was already fixed once for tool
  outputs (`BLUEPRINT.md` §11); don't reintroduce string-template contracts for new tools/nodes.

## 6. Documentation that stays true

- Default to **no comments**; add one only to record a non-obvious WHY (a subtle invariant, a
  workaround, a constraint from `BLUEPRINT.md`'s intricate logic like the chunk rebalancing
  algorithm) — never to restate what a well-named function already says.
- If a change alters the graph topology, config surface, or module layout, update the relevant
  section of `BLUEPRINT.md` (spec) and `EXPLAINED.md` (plain-language tour) in the same change —
  these documents are the "easy to learn the codebase" mechanism for this project; letting them
  drift defeats their purpose.
- When you finish an item from `IMPROVEMENTS.md`, mark it done there the same way earlier fixes
  were (status line + pointer to the spec section), so the punch list stays trustworthy.

## 7. Before calling anything "done"

Run through this checklist for any non-trivial change:

- [ ] Imports respect the layering in §1 (no upward/sideways dependency).
- [ ] Any new tunable is in config, not hardcoded.
- [ ] New boundary data has a schema (Pydantic/TypedDict), not a bare dict.
- [ ] Pure logic is separated from I/O and has a test.
- [ ] No silent failure path — errors/fallbacks are visible (log, return value, or raise).
- [ ] `ruff` and `mypy` (once set up) are clean; `pytest` passes.
- [ ] `BLUEPRINT.md`/`EXPLAINED.md` updated if this changed architecture or behavior a reader would
      expect those docs to reflect.
