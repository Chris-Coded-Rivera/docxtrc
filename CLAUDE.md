# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`docextract` is a registry-driven document-extraction harness. One declarative map
(`doc_type → {purpose, system_hint, fields}`) is the single source of truth that
generates four things that usually drift apart by hand: the **schema** the model must
return, the **prompt** (field descriptions are the per-field instructions), the
**routing** (classification picks a map key), and the **validation**. Adding a document
type is therefore a *data* change (edit YAML), not a code change.

The core engine is plain importable Python; it is exposed three ways over thin adapters —
`import docextract`, a CLI, and an **MCP server** (the flagship adapter, so any MCP host
can call it). `README.md` is unusually thorough on the *why* behind each design decision;
read it before changing architecture.

## Project objective (the direction to build toward)

Real backends so far: `AnthropicLLM`, `OpenAILLM`, `OpenAICompatLLM` (Azure/vLLM/LM Studio/
OpenRouter/… via a custom `base_url`), `GoogleLLM`, and `OllamaLLM` (local, open-source
models). The goal is to make this a genuinely **provider-agnostic, open-source extraction
harness** — equally friendly to Anthropic, OpenAI, Google, and local/open-source models —
without complicating the core.

**The entire seam for this already exists: the `LLM` Protocol in `llm.py`** (`classify`
+ `extract`). The pipeline only ever talks to that interface via dependency injection, so
new providers are *new backend classes, not pipeline changes*. When extending:

- Add a backend per provider implementing the two Protocol methods. Each provider has its own
  structured-output mechanism — Anthropic uses `output_config.format`, OpenAI uses
  `response_format` (the `.parse` helper strict-ifies a Pydantic model directly), Google
  passes `response_schema` on a `GenerateContentConfig` and returns `response.parsed`, Ollama
  takes a top-level `format` JSON schema and returns JSON in `response.message.content`. The
  job of each backend is to translate the generated Pydantic schema into that provider's
  constrained-decoding call and validate the result back. Every backend has an offline wiring
  test (`test_*_backend_wiring_offline`, a fake client) — the pattern for testing a new
  backend without a key or network.
- Keep provider SDKs **lazily imported inside the backend** (as `AnthropicLLM` does with
  `from anthropic import Anthropic`) and declared as **optional extras** in
  `pyproject.toml`, so the core stays dependency-light and installable without any one
  vendor's SDK.
- Keep the `LLM` Protocol narrow. If a provider lacks native structured outputs, emulate
  it behind `extract` (prompt for JSON + validate) rather than widening the interface or
  leaking provider concepts into `pipeline.py`.
- Selecting a backend should be data/config-driven (CLI flag / env var), mirroring how
  `--model` and `DOCEXTRACT_MODEL` work today, so multi-LLM setups (cheap classifier +
  strong extractor) stay a one-line construction.

### Coding conventions for this repo (the owner cares about these)

- **Simplicity over cleverness.** Prefer fewer stages and fewer dependencies. The README
  deliberately lists features (schema-repair retry loops, OCR fallback, a critic stage) as
  *documented hooks left unbuilt* — don't build them in unless asked.
- **Human-readable, explicitly commented code.** Every module starts with a docstring
  explaining *why it exists and what trade-off it makes*, not just what it does. Match that
  density — write for a co-developer who needs to extend and debug this later.

## Repository layout

```
docextract/        the importable package (modules use relative imports: from .ingest ...)
  __init__.py      public API surface
  registry.py schema.py llm.py ingest.py verify.py pipeline.py
  server.py        MCP adapter        cli.py   CLI adapter
examples/
  specs.yaml          declarative registry you edit
  sample_invoice.txt  demo document
tests/
  test_extractor.py   end-to-end, fully offline (imports `from docextract import ...`)
pyproject.toml uv.lock README.md CLAUDE.md
```

## Module map (all under `docextract/`)

- `registry.py` — `FieldSpec`/`DocSpec`, the built-in `DEFAULT_REGISTRY`, and
  `load_registry()` (YAML/dict → registry). The map. `TYPES` is the closed vocabulary of
  allowed field types (`str int float bool date datetime`) — intentionally small to keep
  generated JSON Schema simple.
- `schema.py` — `build_model()` compiles a `DocSpec` into a Pydantic model at runtime
  (`create_model`), always adding `summary` + a flat `evidence` list; `system_prompt()`
  assembles the prompt from the spec. `extra="forbid"` everywhere because structured
  outputs require `additionalProperties:false`.
- `llm.py` — the `LLM` Protocol, its backends (`EchoLLM` offline stub, `AnthropicLLM`,
  `OpenAILLM`, `OpenAICompatLLM` — a thin `OpenAILLM` subclass for any OpenAI-API server via
  `base_url`/`OPENAI_BASE_URL`, `GoogleLLM`, `OllamaLLM`), and the `PROVIDERS` map +
  `build_llm(provider, model=None)` factory that
  turns a provider *name* into a backend (using the provider's default model when none is
  given). **This is the extension point for new providers** (see objective above); CLI and
  server both build their backend through `build_llm`, so a new entry lights up everywhere.
- `ingest.py` — deterministic native-text extraction routed by file extension (txt/md/csv/
  json direct, pdf via pdfplumber, docx via python-docx). No model involved. Scanned PDFs
  raise a clear error at the seam where OCR/vision would plug in.
- `verify.py` — `verify_evidence()` checks each field's returned `quote` actually occurs in
  the source (whitespace-normalised substring match), with no extra model call. Returns a
  non-destructive `VerificationReport` (`grounded`, `coverage`, `ungrounded`); the caller
  decides policy. This is the hallucination gate — catches *unsupported* values, not
  *misread* ones.
- `pipeline.py` — `Extractor` runs `ingest → classify → build schema → extract(+summarise)
  → verify`. Accepts a single `LLM` or a `{role: LLM}` dict (`classifier`/`extractor`/
  `default`) for multi-LLM/multi-agent setups.
- `server.py` — FastMCP adapter exposing `list_document_types` and
  `extract_document(text, doc_type?)`. Returns a JSON **string** (some MCP hosts truncate
  nested results). Configured via `REGISTRY_PATH`, `DOCEXTRACT_PROVIDER`, and
  `DOCEXTRACT_MODEL` env vars.
- `cli.py` — argparse adapter; prints JSON to stdout. Supports `--type`, `--registry`,
  `--provider {anthropic,google,ollama,openai,openai-compat}`, `--model` (defaults to the
  provider's default), `--offline`.

## Commands

Tooling is `uv` (see `uv.lock`).

```bash
uv sync --all-extras            # install project (editable) + all extras incl. pytest
uv run pytest -q                # run the offline test suite (no API key needed)
uv run pytest -q tests/test_extractor.py::test_ungrounded_value_is_flagged   # single test
```

Pytest config lives in `pyproject.toml` (`[tool.pytest.ini_options]`): `pythonpath = ["."]`
puts the repo root on `sys.path` so a bare `pytest` imports the `docextract` package, and
`testpaths = ["tests"]` scopes collection.

The extraction pipeline runs fully **offline** via `EchoLLM` — no API key required for tests
or for exercising the plumbing/grounding report. Cloud backends need their own key in the env
(`ANTHROPIC_API_KEY` for anthropic, `OPENAI_API_KEY` for openai, `GEMINI_API_KEY` or
`GOOGLE_API_KEY` for google). The `ollama` backend needs no key — just a running local Ollama
daemon with the model pulled (`ollama pull llama3.1`).

```bash
# CLI
uv run python -m docextract.cli examples/sample_invoice.txt --offline      # offline stub
ANTHROPIC_API_KEY=sk-... uv run python -m docextract.cli doc.pdf --registry examples/specs.yaml
OPENAI_API_KEY=sk-...    uv run python -m docextract.cli doc.pdf --provider openai --model gpt-4o-mini
GEMINI_API_KEY=...       uv run python -m docextract.cli doc.pdf --provider google
uv run python -m docextract.cli doc.pdf --provider ollama --model llama3.1   # local, no key

# MCP server (stdio transport); pick provider via env
ANTHROPIC_API_KEY=sk-... REGISTRY_PATH=examples/specs.yaml uv run python -m docextract.server
OPENAI_API_KEY=sk-... DOCEXTRACT_PROVIDER=openai uv run python -m docextract.server
GEMINI_API_KEY=... DOCEXTRACT_PROVIDER=google uv run python -m docextract.server
DOCEXTRACT_PROVIDER=ollama uv run python -m docextract.server   # local, no key
```

Note: pip extras are defined in `pyproject.toml` (`anthropic`, `openai`, `google`, `ollama`,
`files`, `mcp`, `dev`); a bare `pip install -e .` installs only `pydantic` (core). The two
model calls on the normal path are classify (enum-constrained to registry keys) + extract
(summary folded in).
