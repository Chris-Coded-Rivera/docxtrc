"""
server.py — expose the extractor over MCP so any MCP host can use it.

This is the "plug into any workflow" layer. The core (Extractor) knows nothing
about MCP; this file is a thin adapter that wraps it in a FastMCP tool. The same
core is reused by the CLI and by direct `import docextract`.

Run locally (stdio, for Claude Desktop / Cursor). Pick the backend with
DOCEXTRACT_PROVIDER (anthropic | openai | google | ollama; default anthropic) and
install that provider's extra plus a key in the env — ollama is local and keyless:
    pip install "mcp[cli]" anthropic pdfplumber python-docx pyyaml   # + provider extra
    ANTHROPIC_API_KEY=sk-... python -m docextract.server             # or: fastmcp run docextract/server.py
    DOCEXTRACT_PROVIDER=ollama python -m docextract.server           # local, no key

Why we return a JSON *string* rather than a nested dict: some MCP hosts
truncate deeply nested tool results in their UI, so a single typed text blob is
the robust choice.
"""

from __future__ import annotations

import json
import os

from mcp.server.fastmcp import FastMCP  # bundled with the official `mcp` SDK

from .llm import build_llm
from .pipeline import Extractor
from .registry import DEFAULT_REGISTRY, load_registry

mcp = FastMCP("docextract")

# Build the core once at startup. Point REGISTRY_PATH at a YAML file to own the
# document map outside the code; otherwise the built-in registry is used.
_registry = load_registry(os.environ["REGISTRY_PATH"]) if os.environ.get("REGISTRY_PATH") else DEFAULT_REGISTRY
# Choose the backend by env so the same server binary serves any provider:
#   DOCEXTRACT_PROVIDER=openai DOCEXTRACT_MODEL=gpt-4o-mini  (model is optional;
#   omit it to use the provider's default model).
_provider = os.environ.get("DOCEXTRACT_PROVIDER", "anthropic")
_model = os.environ.get("DOCEXTRACT_MODEL")  # None -> provider default
_extractor = Extractor(_registry, build_llm(_provider, _model))


@mcp.tool()
def list_document_types() -> str:
    """List the document types this server can extract, with their purpose."""
    return json.dumps({t: s.purpose for t, s in _registry.items()}, indent=2)


@mcp.tool()
def extract_document(text: str, doc_type: str | None = None) -> str:
    """
    Extract structured fields from a document's text.

    Args:
        text: The full plain text of the document.
        doc_type: Optional. Force a document type from list_document_types();
                  if omitted, the type is classified automatically.

    Returns:
        A JSON string with: doc_type, fields, summary, and a grounding report
        (which fields were verified against the source text).
    """
    result = _extractor.run_text(text, doc_type=doc_type)
    payload = result.data.model_dump(mode="json")
    summary = payload.pop("summary", None)
    payload.pop("evidence", None)
    return json.dumps(
        {
            "doc_type": result.doc_type,
            "fields": payload,
            "summary": summary,
            "grounding": {
                "coverage": round(result.verification.coverage, 3),
                "ungrounded_fields": result.verification.ungrounded,
            },
        },
        indent=2,
        default=str,
    )


if __name__ == "__main__":
    mcp.run()  # stdio transport by default
