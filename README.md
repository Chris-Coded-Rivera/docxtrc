# docextract

A small, reliable document-extraction harness. You give it a **map** of document
types and the fields each one should yield; it figures out which type a document
is, switches into the right domain expertise, extracts the fields into a
validated structure, summarises the document, and tells you which values it
could actually ground in the source text.

The core knows nothing about how it's called. The same engine is exposed three
ways — as a Python import, a CLI, and an **MCP server** — so it drops into any
workflow without changing the logic.

---

## The one idea everything hangs on

The `doc_type → {purpose, domain hint, fields}` map is the **single source of
truth**. It drives four things that are usually written by hand and drift apart:

1. the **schema** the model must return,
2. the **prompt** (the field descriptions *are* the per-field instructions; the
   per-type `system_hint` *is* the domain switch),
3. the **routing** (classification picks a map key), and
4. the **validation** (the schema validates the result).

Because all four are generated from one place, **adding a document type is a
data change, not a code change.** That is the whole reason the tool is reusable
across use cases instead of being rebuilt per project.

---

## Architecture, and why each piece is the way it is

### 1. The registry is declarative data, not code
`registry.py` defines two tiny dataclasses (`FieldSpec`, `DocSpec`) and a
`load_registry()` that reads the same shape from a YAML file
(`examples/specs.yaml`). A built-in `DEFAULT_REGISTRY` keeps the package runnable
out of the box.

*Why:* keeping the map declarative lets a non-engineer own it, lets you diff and
review it, and lets one binary serve many domains by pointing at a different
file. *Trade-off:* a declarative map can only express the field shapes the loader
knows about (the six scalar types). That's deliberate — exotic per-field logic
belongs in code, and a small type vocabulary keeps the generated schema simple,
which the model's decoder rewards (see §2).

### 2. The schema is generated at runtime, and the model is *forced* to fill it
`schema.build_model()` compiles a `DocSpec` into a Pydantic model with
`pydantic.create_model`. That model is handed to Claude's **Structured Outputs**
feature, which constrains decoding to the schema's grammar — the returned JSON is
*guaranteed* to match. (Structured Outputs went GA on the Claude API; the field
is `output_config.format`, and the Python SDK accepts Pydantic models directly.
See https://docs.claude.com/en/docs/build-with-claude/structured-outputs.)

*Why this changed the design for the better:* an earlier instinct here is to wrap
extraction in a "validate, and if it fails re-prompt the model to fix the JSON"
loop. With constrained decoding that loop is **dead code** — the shape can't come
back wrong. Removing it makes the harness simpler and cheaper. *Trade-off:*
Structured Outputs imposes JSON-Schema limits (no recursive schemas; flat is
better; `additionalProperties:false`; complexity caps). We stay inside them by
making `evidence` a **flat list** rather than an open dict keyed by field name,
and by setting `extra="forbid"` on every generated model.

### 3. Grounding + deterministic verification is the reliability backbone
Structured Outputs guarantees the *shape* of the answer, never its *truth*. So
every generated model carries an `evidence` list: for each field, the model
returns the **verbatim quote** it pulled the value from. `verify.py` then checks —
with no extra model call — that each quote actually occurs in the source text
(whitespace-normalised substring match). Any value whose quote isn't present is
flagged as ungrounded.

*Why:* this is the cheapest high-signal hallucination check available. A
fabricated value almost never comes with a quote that survives a substring test
against the real document. *Trade-off:* it catches *unsupported* values, not
*misread* ones (a real quote paired with a wrong interpretation can still pass).
It is a strong first gate, not a proof of correctness — which is why
`verify_evidence` returns a **non-destructive report** and lets the caller decide
policy (null the field, re-ask, or route to a human) rather than silently
mutating results.

### 4. Ingestion is deterministic and kept away from the model
`ingest.py` pulls the native text layer (txt/md/csv/json directly; pdf via
pdfplumber; docx via python-docx) before any model sees the document.

*Why:* native text is cheaper and more faithful than asking a model to retype a
page, and it keeps the model doing the thing it's good at — understanding and
locating, not transcription. *Trade-off:* scanned/image-only PDFs have no text
layer; rather than hide an OCR/vision fallback inside the happy path, the code
raises a clear error at exactly the seam where you'd plug one in.

### 5. The LLM is a one-method-ish Protocol — the swap and multi-agent point
`llm.py` defines an `LLM` Protocol (`classify`, `extract`). The pipeline only
ever talks to that interface, so a provider is a *backend class, not a pipeline
change*. Six backends ship today:

| Backend | What it is | Structured-output mechanism |
|---|---|---|
| `AnthropicLLM` | Claude | `output_config.format` (constrained decoding) |
| `OpenAILLM` | GPT | `response_format` — the SDK's `.parse` strict-ifies our Pydantic model |
| `OpenAICompatLLM` | any OpenAI-compatible server (Azure, vLLM, LM Studio, Together, OpenRouter, Groq, llama.cpp…) | reuses `OpenAILLM` against a custom `base_url` |
| `GoogleLLM` | Gemini | `response_schema` on a `GenerateContentConfig`, returns `.parsed` |
| `OllamaLLM` | local open-source models (Llama, Mistral, Qwen…), no key | top-level `format` JSON schema, returns JSON in `message.content` |
| `EchoLLM` | offline deterministic stub for tests/demos | n/a (no network) |

A tiny `PROVIDERS` map plus `build_llm(provider, model=None)` turns a provider
*name* into a backend (using that provider's default model when none is given),
so the CLI (`--provider`) and MCP server (`DOCEXTRACT_PROVIDER`) select a backend
by name and a new provider lights up in both at once.

*Why:* this is what makes "swap the core model" and "multi-agent" the *same*
mechanism — dependency injection. Construct the `Extractor` with one backend and
every stage uses it; pass a `{"classifier": cheap, "extractor": strong}` dict
(even mixing providers) and each stage uses its own. *Trade-off:* the Protocol is
intentionally narrow. A backend with a richer native capability (batch, streaming,
server-side tools) has to express it behind these two methods or extend the
Protocol; we chose a small stable seam over exposing every provider feature.
Each provider SDK is a **lazily-imported optional extra**, so the core stays
dependency-light and installs without any one vendor's package.

### 6. The pipeline is a few composable stages
`pipeline.py`'s `Extractor` runs `ingest → classify → build schema →
extract(+summarise) → verify`. Summarisation is folded into the extraction
schema, so the normal path is **two model calls** (classify + extract), with the
cheap call enum-constrained to the registry keys.

*Why:* fewer calls, less latency, less cost, and the summary is grounded in the
same pass as the fields. *Trade-off:* folding summary in means you can't run a
different (e.g. cheaper) model for summarisation alone — acceptable, since it
rides along on the extract call for free. Multi-agent extensions (a **critic**
that re-reads low-confidence fields, or **parallel** extraction of field groups
for very long documents) are additive stages, left as documented hooks rather
than built in, to honour the simplicity-over-complexity priority.

---

## Plugin, tool, or MCP? The decision

**Recommendation: build the core as a plain importable library, then expose it
through thin adapters — and make an MCP server the flagship adapter.** This repo
does exactly that.

The three words aren't really alternatives; they're different *exposure layers*
over one core, and they trade off along one axis — how tightly coupled the
consumer is to a specific host:

- A **"tool"** in the LLM sense (a function with a schema you register inside one
  agent framework) is coupled to *that framework's* tool-calling convention.
  Useful, but you re-wrap it for every framework.
- A **plugin** is coupled to a specific *host application's* plugin system
  (a particular IDE, a particular agent runtime). Same rewrap problem, per host.
- An **MCP server** speaks a host-agnostic wire protocol (JSON-RPC). Write it
  once and any MCP-aware client can discover and call it — Claude Desktop,
  Claude Code, Cursor, and the growing set of agent runtimes that speak MCP.
  That is the precise definition of "plug into any workflow/pipeline."

So MCP wins the headline because it *decouples* the extractor from any single
host. But MCP shouldn't be the *only* door: a pure Python `import` is the most
universal interface for any Python pipeline, and a CLI is the right call for
shells, cron, and CI. Hence the layered shape:

```
        registry (the map)
              │
        core library  ──────────────►  import docextract        (any Python code)
   (Extractor + schema + verify)
              ├───────────────────────►  python -m docextract.cli (shell / cron / CI)
              └───────────────────────►  docextract.server (MCP)  (any MCP host)  ◄── flagship
```

The MCP layer (`server.py`) is ~50 lines: it wraps the same `Extractor` in a
FastMCP `@mcp.tool()` and returns a JSON string (some hosts truncate deeply
nested tool results, so a single typed text blob is the robust choice). FastMCP
ships inside the official `mcp` Python SDK and turns a typed function into a
spec-compliant tool from its signature and docstring. See
https://github.com/modelcontextprotocol/python-sdk.

---

## Trade-offs at a glance

| Decision | We gained | We gave up |
|---|---|---|
| Map drives schema/prompt/routing/validation | New type = data change | Field shapes limited to the loader's type vocabulary |
| Structured Outputs (constrained decoding) | No JSON parsing, no schema-repair retries | Must respect JSON-Schema limits (flat, non-recursive) |
| Evidence + substring verification | Cheap hallucination gate, no extra call | Catches *unsupported* values, not *misread* ones |
| Deterministic ingestion | Faithful, cheap text | Scans need an explicit OCR/vision plug-in |
| LLM Protocol + injection | Swap model & multi-agent for free | Narrow seam hides provider-specific features |
| Summary folded into extract | Two calls total, grounded summary | Can't run a separate cheap summariser |
| MCP as flagship interface | Host-agnostic reuse | One more dependency vs a bare library |

---

## Runbook

### 0. Install
```bash
pip install -e .                      # core only (pydantic)
pip install -e ".[anthropic,files]"   # Claude + pdf/docx/yaml ingestion
pip install -e ".[openai]"            # GPT backend
pip install -e ".[google]"            # Gemini backend
pip install -e ".[ollama]"            # local open-source models (no key)
pip install -e ".[mcp]"               # to run as an MCP server
pip install -e ".[dev]"               # everything + pytest
```
One optional extra per provider; install only the backend(s) you use. (With `uv`:
`uv sync --all-extras` installs everything.)

### 1. Define your document map
Edit `examples/specs.yaml` (or pass your own path). Each type needs a `purpose`,
a `system_hint` (its domain brain), and `fields` (`type`, `description`, optional
`required: false`). This file is the thing you'll edit most; nothing else usually
changes when you add a document type.

### 2. Try it offline (no API key)
```bash
python -m docextract.cli examples/sample_invoice.txt --offline
python tests/test_extractor.py        # or: pytest -q
```
`--offline` uses `EchoLLM`, a deliberately naive stub — it proves the plumbing
and the grounding report, but real extraction needs a model backend (step 3).

### 3. Run it for real
Pick a provider with `--provider` (default `anthropic`); `--model` defaults to
that provider's default model, so switching providers needn't mean knowing each
one's model id. Each cloud provider reads its own key from the env; `ollama` is
local and needs no key.
```bash
ANTHROPIC_API_KEY=sk-... python -m docextract.cli path/to/invoice.pdf --registry examples/specs.yaml
OPENAI_API_KEY=sk-...    python -m docextract.cli path/to/doc.txt --provider openai --model gpt-4o-mini
GEMINI_API_KEY=...       python -m docextract.cli path/to/doc.txt --provider google
python -m docextract.cli path/to/doc.txt --provider ollama --model llama3.1   # local, no key

# Any OpenAI-compatible server (vLLM, LM Studio, Azure, OpenRouter, …) via env:
OPENAI_BASE_URL=http://localhost:8000/v1 \
  python -m docextract.cli path/to/doc.txt --provider openai-compat --model llama-3.1-8b-instruct
```
Keys per provider: `ANTHROPIC_API_KEY` (anthropic), `OPENAI_API_KEY` (openai),
`GEMINI_API_KEY`/`GOOGLE_API_KEY` (google); `ollama` needs a running local daemon
with the model pulled (`ollama pull llama3.1`). For `openai-compat`, set
`OPENAI_BASE_URL` to the server's `/v1` endpoint (and `OPENAI_API_KEY` if the
gateway requires one); structured outputs need a server that supports OpenAI's
`json_schema` response format.

Output is JSON on stdout (pipes into `jq` or a downstream step): the doc type,
the extracted `data`, and a `grounding` block with coverage and any ungrounded
fields.

### 4. Use it from Python
```python
from docextract import Extractor, load_registry
from docextract import AnthropicLLM, OpenAILLM, GoogleLLM, OllamaLLM, build_llm

registry = load_registry("examples/specs.yaml")

# Construct a backend directly...
ex = Extractor(registry, AnthropicLLM("claude-opus-4-8"))
# ...or by provider name (uses the provider's default model if you omit it):
ex = Extractor(registry, build_llm("openai"))
ex = Extractor(registry, OllamaLLM("llama3.1"))           # local, no key

result = ex.run("invoice.pdf")
print(result.doc_type, result.data.total_due, result.verification.coverage)

# Multi-LLM, even across providers: cheap classifier, strong extractor
ex = Extractor(registry, {
    "classifier": OpenAILLM("gpt-4o-mini"),
    "extractor":  AnthropicLLM("claude-opus-4-8"),
})
```

### 5. Run it as an MCP server (the "any workflow" path)
```bash
export ANTHROPIC_API_KEY=sk-...
export REGISTRY_PATH=examples/specs.yaml        # optional; defaults to built-in map
export DOCEXTRACT_PROVIDER=anthropic            # optional; openai | google | ollama
export DOCEXTRACT_MODEL=claude-opus-4-8         # optional; defaults to the provider's default
python -m docextract.server                     # stdio transport
# or, with the FastMCP CLI for hot-reload/inspector:
fastmcp dev docextract/server.py
```
The same server binary serves any provider — e.g.
`OPENAI_API_KEY=sk-... DOCEXTRACT_PROVIDER=openai python -m docextract.server`, or
`DOCEXTRACT_PROVIDER=ollama python -m docextract.server` for a local, keyless run.
It exposes two tools: `list_document_types` and `extract_document(text, doc_type?)`.
Point Claude Desktop / Cursor at it via their MCP config (command = the python
above), then ask the host to extract a document and it will call the tool.

### 6. Add a new document type
Add a block to `specs.yaml`. That's it — no code change. Re-run; classification
will start routing matching documents to it. (If your new type is easy to
confuse with an existing one, tighten the `purpose` strings, since classification
reads them.)

### 7. Operate & observe
Watch `grounding.coverage` and `grounding.ungrounded_fields` in the output. A
drop in coverage on a document class is your early-warning signal that a layout
changed or the map's hints have drifted from reality. Decide per use case whether
ungrounded fields should be nulled, re-asked, or sent to a human queue.

### 8. Troubleshoot
- *"no extractable text layer"* — the PDF is a scan; plug an OCR/vision reader
  into `ingest._read_pdf`.
- *Wrong document type chosen* — make the `purpose` strings more distinctive, or
  pass `--type` / the `doc_type` argument to skip classification.
- *Schema rejected by the API* — you likely added a deeply nested or recursive
  field; flatten it (Structured Outputs disallows recursion and caps complexity).
- *MCP result looks truncated in the host* — keep returning a JSON string, not a
  nested object (already the default in `server.py`).

---

## Extending toward multi-agent

The pipeline is stages over a shared `Context`, so new agents are new stages:

- **Critic** — after extract, run a second backend that re-reads only the fields
  with low grounding and either confirms or corrects them. Wire it as
  `models["critic"]` and call it inside `Extractor.run_text` for
  `verification.ungrounded` fields.
- **Parallel field groups** — for very long documents, split `spec.fields` into
  groups, build a sub-model per group, extract concurrently, and merge. The map
  already gives you the field list to partition.
- **New file types / OCR** — add a branch in `ingest.read_document`; the rest of
  the pipeline is unaffected.

---

## File map
```
docextract/
  registry.py   FieldSpec / DocSpec / DEFAULT_REGISTRY / load_registry   (the map)
  schema.py     build_model (create_model) + Evidence + system_prompt    (map → schema)
  llm.py        LLM Protocol + Anthropic/OpenAI/Google/Ollama + EchoLLM   (the swap point)
                + PROVIDERS map / build_llm() factory
  ingest.py     read_document, routed by file type                       (deterministic text)
  verify.py     verify_evidence + VerificationReport                     (grounding check)
  pipeline.py   Extractor + Context + ExtractionResult                   (the harness)
  server.py     FastMCP adapter                                          (MCP interface)
  cli.py        argparse adapter                                         (CLI interface)
examples/
  specs.yaml          declarative registry you edit
  sample_invoice.txt  demo document
tests/
  test_extractor.py   end-to-end, runs fully offline
```
