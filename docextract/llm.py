"""
llm.py — the swap point.

Every backend implements one small Protocol. The pipeline only ever talks to
this interface, so swapping the core model (or pointing different pipeline
stages at different models) is dependency injection, not a rewrite.

Two methods cover everything we need:

    classify(...) -> picks a doc_type from a fixed list (cheap, enum-constrained)
    extract(...)  -> returns an instance of a generated Pydantic model

We ship two backends:

  * EchoLLM       — runs fully offline. A deterministic test double / demo
                    backend so the harness is runnable with no API key. It does
                    a little naive regex so the demo looks alive, and accepts a
                    `script` for fully deterministic tests.
  * AnthropicLLM  — the real backend. Uses Structured Outputs (GA), so the
                    returned JSON is guaranteed to match the schema by
                    constrained decoding — no parse-and-retry needed.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from pydantic import BaseModel


@runtime_checkable
class LLM(Protocol):
    name: str

    def classify(self, *, document: str, doc_types: dict[str, str]) -> str:
        """Return the best-matching key from doc_types ({type: purpose})."""
        ...

    def extract(self, *, system: str, document: str, schema: type[BaseModel]) -> BaseModel:
        """Return an instance of `schema` populated from the document."""
        ...


# --------------------------------------------------------------------------- #
# Offline backend
# --------------------------------------------------------------------------- #
class EchoLLM:
    """
    A no-network backend for tests and demos.

    Pass `script` for deterministic behaviour:
        EchoLLM(script={"classify": "invoice",
                        "extract": {"invoice_number": "INV-1", ...}})
    With no script it falls back to a tiny keyword/regex heuristic so the
    canned sample document still produces something to look at.
    """

    name = "echo"

    def __init__(self, script: dict | None = None):
        self.script = script or {}

    def classify(self, *, document: str, doc_types: dict[str, str]) -> str:
        if "classify" in self.script:
            return self.script["classify"]
        text = document.lower()
        for doc_type in doc_types:
            if doc_type in text:               # naive: "invoice" appears in an invoice
                return doc_type
        return next(iter(doc_types))           # default to the first registered type

    def extract(self, *, system: str, document: str, schema: type[BaseModel]) -> BaseModel:
        if "extract" in self.script:
            return schema.model_validate(self.script["extract"])

        # Heuristic fallback: fill required fields with a best-effort guess and
        # attach a real quote so evidence verification has something to check.
        data: dict = {}
        evidence: list[dict] = []
        for name, fld in schema.model_fields.items():
            if name in ("summary", "evidence"):
                continue
            line = _first_line_mentioning(document, name)
            data[name] = _coerce(line, fld.annotation)
            if line:
                evidence.append({"field": name, "quote": line, "page": 1})
        data["summary"] = document.strip().split("\n")[0][:200]
        data["evidence"] = evidence
        return schema.model_validate(data)


def _first_line_mentioning(text: str, field_name: str) -> str:
    """Find a document line that looks related to a field name. Best effort."""
    word = field_name.split("_")[0]
    for line in text.splitlines():
        if word.lower() in line.lower() and line.strip():
            return line.strip()
    return ""


def _coerce(line: str, annotation) -> object:
    """Pull a crude value of the right shape out of a line of text."""
    ann = str(annotation)
    if "float" in ann or "int" in ann:
        m = re.search(r"[-+]?\d[\d,]*\.?\d*", line)
        if not m:
            return 0
        num = m.group().replace(",", "")
        return float(num) if "float" in ann else int(float(num))
    if "date" in ann:
        return None  # let optional dates stay null in the offline stub
    return line or "unknown"


# --------------------------------------------------------------------------- #
# Real backend (Anthropic Structured Outputs, GA)
# --------------------------------------------------------------------------- #
class AnthropicLLM:
    """
    Production backend using the Claude API's Structured Outputs.

    We pass the generated model's JSON Schema via `output_config.format`; the
    API constrains decoding to that grammar, so `response.content[0].text` is
    guaranteed-valid JSON for the schema. We then validate it back into the
    Pydantic model (validation here is a formality, not a retry gate).

    Requires:  pip install anthropic   and   ANTHROPIC_API_KEY in the env.
    """

    def __init__(self, model: str = "claude-opus-4-8", client=None, max_tokens: int = 4096):
        from anthropic import Anthropic  # imported lazily; only needed for real runs

        self.client = client or Anthropic()
        self.model = model
        self.name = model
        self.max_tokens = max_tokens

    def classify(self, *, document: str, doc_types: dict[str, str]) -> str:
        menu = "\n".join(f"- {t}: {purpose}" for t, purpose in doc_types.items())
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            system="Classify the document into exactly one of the listed types.",
            messages=[{"role": "user", "content": f"Types:\n{menu}\n\nDocument:\n{document}"}],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"doc_type": {"type": "string", "enum": list(doc_types)}},
                        "required": ["doc_type"],
                    },
                }
            },
        )
        import json

        return json.loads(resp.content[0].text)["doc_type"]

    def extract(self, *, system: str, document: str, schema: type[BaseModel]) -> BaseModel:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": document}],
            output_config={"format": {"type": "json_schema", "schema": schema.model_json_schema()}},
        )
        # Guaranteed-valid JSON for our schema thanks to constrained decoding.
        return schema.model_validate_json(resp.content[0].text)


# --------------------------------------------------------------------------- #
# Real backend (OpenAI Structured Outputs)
# --------------------------------------------------------------------------- #
class OpenAILLM:
    """
    Production backend using the OpenAI API's Structured Outputs.

    This is the OpenAI sibling of `AnthropicLLM`: the pipeline can't tell them
    apart because both satisfy the same `LLM` Protocol. Only the wire details
    differ — OpenAI carries the schema in `response_format` instead of
    Anthropic's `output_config.format`, and uses `system`/`user` chat messages.

    For `extract` we hand the SDK our generated Pydantic model directly via the
    `.parse` helper. The helper converts the model into a *strict* JSON Schema
    (it forces `additionalProperties:false` and marks every property required —
    our `Optional` fields become required-but-nullable, which is fine: the model
    returns null and `verify.py`/the caller treat null as "absent"), constrains
    decoding to it, and returns a ready-validated instance as `.parsed`. So,
    like the Anthropic path, there is no JSON parsing or schema-repair retry.

    Requires:  pip install openai   and   OPENAI_API_KEY in the env.
    """

    def __init__(self, model: str = "gpt-4o", client=None, max_tokens: int = 4096):
        from openai import OpenAI  # imported lazily; only needed for real runs

        self.client = client or OpenAI()
        self.model = model
        self.name = model
        self.max_tokens = max_tokens

    def classify(self, *, document: str, doc_types: dict[str, str]) -> str:
        menu = "\n".join(f"- {t}: {purpose}" for t, purpose in doc_types.items())
        # A hand-written strict schema with an `enum` is enough here: the only
        # field is `doc_type`, already required, so strict mode accepts it as-is.
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Classify the document into exactly one of the listed types."},
                {"role": "user", "content": f"Types:\n{menu}\n\nDocument:\n{document}"},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "classification",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"doc_type": {"type": "string", "enum": list(doc_types)}},
                        "required": ["doc_type"],
                    },
                },
            },
        )
        import json

        return json.loads(resp.choices[0].message.content)["doc_type"]

    def extract(self, *, system: str, document: str, schema: type[BaseModel]) -> BaseModel:
        completion = self.client.chat.completions.parse(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": document},
            ],
            response_format=schema,  # the SDK strict-ifies our Pydantic model
        )
        message = completion.choices[0].message
        # `.parsed` is None when the model refused or output was truncated; surface
        # that loudly rather than returning a half-empty result.
        if message.parsed is None:
            reason = getattr(message, "refusal", None) or "no parsed output (possible truncation)"
            raise RuntimeError(f"OpenAI structured extraction returned nothing: {reason}")
        return message.parsed


# --------------------------------------------------------------------------- #
# Any OpenAI-API-compatible server (Azure OpenAI, vLLM, LM Studio, Together,
# OpenRouter, Groq, a local llama.cpp server, …)
# --------------------------------------------------------------------------- #
class OpenAICompatLLM(OpenAILLM):
    """
    One backend for the whole ecosystem of servers that speak the OpenAI Chat
    Completions wire format. Because the protocol is identical, we reuse
    `OpenAILLM`'s `classify`/`extract` verbatim and only change *where the client
    points* — this subclass exists purely to construct the SDK client against a
    custom endpoint.

    base_url: the server's OpenAI-compatible endpoint, e.g. http://localhost:8000/v1
        (vLLM / LM Studio) or https://openrouter.ai/api/v1. Falls back to the
        OPENAI_BASE_URL env var.
    api_key:  hosted gateways require one; purely local servers usually accept any
        non-empty string. Falls back to OPENAI_API_KEY, then a literal placeholder
        so a keyless local server still works.
    model:    required — there's no universal default, it's whatever the target
        server serves (e.g. "llama-3.1-8b-instruct", "gpt-4o" on Azure).

    Caveat: structured outputs work only if the target server actually implements
    OpenAI's json_schema `response_format` (vLLM and recent llama.cpp do; some
    gateways don't). Uses the same `openai` extra — no extra dependency.
    """

    def __init__(self, model: str, *, base_url: str | None = None, api_key: str | None = None,
                 client=None, max_tokens: int = 4096):
        if client is None:
            import os

            from openai import OpenAI

            base_url = base_url or os.environ.get("OPENAI_BASE_URL")
            if not model:
                raise ValueError(
                    "OpenAICompatLLM needs a model name (the id your server serves); "
                    "pass it via --model / DOCEXTRACT_MODEL."
                )
            if not base_url:
                raise ValueError(
                    "OpenAICompatLLM needs base_url (or OPENAI_BASE_URL) pointing at the "
                    "server's OpenAI-compatible /v1 endpoint."
                )
            # Many local servers ignore the key but the SDK insists one is set.
            api_key = api_key or os.environ.get("OPENAI_API_KEY") or "not-needed"
            client = OpenAI(base_url=base_url, api_key=api_key)
        # Hand the ready client to OpenAILLM; it owns all the request logic.
        super().__init__(model=model, client=client, max_tokens=max_tokens)


# --------------------------------------------------------------------------- #
# Real backend (Google Gemini Structured Outputs)
# --------------------------------------------------------------------------- #
class GoogleLLM:
    """
    Production backend using the Gemini API's structured output (`response_schema`).

    Same `LLM` Protocol, same shape as the other two — only the wire details
    differ. Gemini takes the schema on a `config` object: set
    `response_mime_type="application/json"` and hand `response_schema` either our
    generated Pydantic model (extract) or an explicit enum schema (classify). The
    SDK then constrains decoding and, for a Pydantic schema, returns a
    ready-validated instance as `response.parsed` — so, like the Anthropic and
    OpenAI paths, there's no JSON parsing or schema-repair retry.

    Note: Gemini's structured output covers a subset of JSON Schema (e.g. it
    ignores `additionalProperties`); the SDK derives a compatible schema from our
    Pydantic model, so we don't hand-translate it.

    Uses the unified `google-genai` SDK (`from google import genai`).
    Requires:  pip install google-genai   and   GEMINI_API_KEY (or GOOGLE_API_KEY).
    """

    def __init__(self, model: str = "gemini-2.5-flash", client=None, max_tokens: int = 4096):
        from google import genai  # imported lazily; only needed for real runs

        self.client = client or genai.Client()
        self.model = model
        self.name = model
        self.max_tokens = max_tokens

    def classify(self, *, document: str, doc_types: dict[str, str]) -> str:
        import json

        from google.genai import types

        menu = "\n".join(f"- {t}: {purpose}" for t, purpose in doc_types.items())
        # Explicit one-field schema with an `enum`, mirroring the other backends'
        # classify, rather than Gemini's enum mode (which constrains type keys to
        # be enum-name-safe — our registry keys needn't be).
        resp = self.client.models.generate_content(
            model=self.model,
            contents=f"Types:\n{menu}\n\nDocument:\n{document}",
            config=types.GenerateContentConfig(
                system_instruction="Classify the document into exactly one of the listed types.",
                response_mime_type="application/json",
                response_schema=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "doc_type": types.Schema(type=types.Type.STRING, enum=list(doc_types))
                    },
                    required=["doc_type"],
                ),
            ),
        )
        return json.loads(resp.text)["doc_type"]

    def extract(self, *, system: str, document: str, schema: type[BaseModel]) -> BaseModel:
        from google.genai import types

        resp = self.client.models.generate_content(
            model=self.model,
            contents=document,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=schema,  # the SDK derives a Gemini schema from our model
                max_output_tokens=self.max_tokens,
            ),
        )
        # `.parsed` is None when the response was blocked or truncated; surface that
        # loudly rather than returning a half-empty result.
        if resp.parsed is None:
            raise RuntimeError(
                "Gemini structured extraction returned nothing "
                "(possible safety block or truncation)."
            )
        return resp.parsed


# --------------------------------------------------------------------------- #
# Local backend (Ollama structured outputs) — runs open-source models on-machine
# --------------------------------------------------------------------------- #
class OllamaLLM:
    """
    Local backend using Ollama, so the harness can run fully open-source models
    (Llama, Mistral, Qwen, …) on your own machine — no provider, no API key.

    Same `LLM` Protocol, same shape as the cloud backends. Ollama carries the
    schema in a top-level `format` field: hand it our generated model's JSON
    Schema and Ollama constrains decoding to it, returning JSON in
    `response.message.content`. We validate that back into the Pydantic model —
    the same "constrain, then validate as a formality" pattern as `AnthropicLLM`.

    `host` defaults to the SDK's own default (the `OLLAMA_HOST` env var, else
    http://localhost:11434), so a remote/containerised Ollama works by pointing
    it elsewhere. Pull the model first, e.g. `ollama pull llama3.1`.

    Requires:  pip install ollama   and a running Ollama daemon.
    """

    def __init__(self, model: str = "llama3.1", client=None, host: str | None = None,
                 max_tokens: int = 4096):
        from ollama import Client  # imported lazily; only needed for real runs

        self.client = client or Client(host=host)
        self.model = model
        self.name = model
        self.max_tokens = max_tokens

    def classify(self, *, document: str, doc_types: dict[str, str]) -> str:
        import json

        menu = "\n".join(f"- {t}: {purpose}" for t, purpose in doc_types.items())
        resp = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": "Classify the document into exactly one of the listed types."},
                {"role": "user", "content": f"Types:\n{menu}\n\nDocument:\n{document}"},
            ],
            format={
                "type": "object",
                "additionalProperties": False,
                "properties": {"doc_type": {"type": "string", "enum": list(doc_types)}},
                "required": ["doc_type"],
            },
        )
        return json.loads(resp.message.content)["doc_type"]

    def extract(self, *, system: str, document: str, schema: type[BaseModel]) -> BaseModel:
        resp = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": document},
            ],
            format=schema.model_json_schema(),  # Ollama constrains output to this schema
            options={"num_predict": self.max_tokens},  # Ollama's name for a max-tokens cap
        )
        # Constrained to our schema, so this validate is a formality, not a retry gate.
        return schema.model_validate_json(resp.message.content)


# --------------------------------------------------------------------------- #
# Provider factory — the one place a provider *name* maps to a backend class.
# --------------------------------------------------------------------------- #
# name -> (backend class, that provider's default model). Adding a provider is a
# one-line entry here plus its backend class above; the CLI and MCP server both
# build their backend through `build_llm`, so a new provider lights up in both
# at once and their --provider choices stay in sync.
PROVIDERS: dict[str, tuple[type, str]] = {
    "anthropic": (AnthropicLLM, "claude-opus-4-8"),
    "openai": (OpenAILLM, "gpt-4o"),
    "google": (GoogleLLM, "gemini-2.5-flash"),
    "ollama": (OllamaLLM, "llama3.1"),
    # OpenAI-compatible servers: no universal default model (it's whatever the
    # server serves) and base_url comes from OPENAI_BASE_URL — so the empty
    # default makes a missing --model fail loudly in OpenAICompatLLM.__init__.
    "openai-compat": (OpenAICompatLLM, ""),
}


def build_llm(provider: str, model: str | None = None) -> LLM:
    """
    Construct a real backend by provider name.

    `model` is optional: when omitted we use that provider's default model, so a
    caller can switch providers without also having to know each one's model id.
    The offline `EchoLLM` is deliberately not here — it's selected explicitly
    (e.g. the CLI's --offline flag), not by provider name.
    """
    try:
        cls, default_model = PROVIDERS[provider]
    except KeyError:
        raise ValueError(
            f"Unknown provider {provider!r}. Choices: {', '.join(sorted(PROVIDERS))}."
        )
    return cls(model or default_model)
