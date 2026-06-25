"""
Live end-to-end tests — the only tests that hit a real provider.

These are deliberately *opt-in*: the whole module is marked `live`, and
`addopts = -m 'not live'` (in pyproject) excludes it from the normal suite, so
`uv run pytest -q` stays fully offline and key-free. Run these explicitly:

    uv run pytest -m live -q                 # every provider you have creds for
    uv run pytest -m live -q -k anthropic    # just one

Each test still *skips* (not fails) when its credential/endpoint isn't present,
so `-m live` does the right thing on a machine that only has one provider set up.

What we assert: a real model, given the sample invoice, must classify it as an
invoice and pull values that actually ground in the source text — i.e. the whole
pipeline (ingest → classify → schema → extract → verify) works against the live
API, not just our fake-client wiring. We check the known-exact identifiers from
examples/sample_invoice.txt rather than fuzzy values, since models copy those
reliably.
"""

import os
import socket
from pathlib import Path
from urllib.parse import urlparse

import pytest

from docextract import DEFAULT_REGISTRY, Extractor, build_llm

pytestmark = pytest.mark.live  # marks every test in this module

SAMPLE = Path(__file__).resolve().parents[1] / "examples" / "sample_invoice.txt"

# Optional per-run model override, mirroring the server's env var. Lets you point
# a test at a specific model (and is *required* for openai-compat, which has no
# default model).
_MODEL = os.environ.get("DOCEXTRACT_MODEL")


def _run_and_check(provider: str):
    """Run the real pipeline for `provider` and assert it genuinely extracted."""
    extractor = Extractor(DEFAULT_REGISTRY, build_llm(provider, _MODEL))
    result = extractor.run(str(SAMPLE))

    # Routing worked, and the values match the document's exact identifiers.
    assert result.doc_type == "invoice"
    assert result.data.invoice_number == "INV-2025-00842"
    assert result.data.total_due == pytest.approx(750.60)
    # Grounding ran and at least found support — the reliability backbone is live.
    assert result.verification.coverage > 0
    return result


def _ollama_up() -> bool:
    """True if a local Ollama daemon is accepting connections."""
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    parsed = urlparse(host if "://" in host else f"http://{host}")
    try:
        with socket.create_connection((parsed.hostname, parsed.port or 11434), timeout=1):
            return True
    except OSError:
        return False


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="no ANTHROPIC_API_KEY")
def test_live_anthropic():
    _run_and_check("anthropic")


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="no OPENAI_API_KEY")
def test_live_openai():
    _run_and_check("openai")


@pytest.mark.skipif(
    not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
    reason="no GEMINI_API_KEY / GOOGLE_API_KEY",
)
def test_live_google():
    _run_and_check("google")


@pytest.mark.skipif(not _ollama_up(), reason="no local Ollama daemon reachable")
def test_live_ollama():
    _run_and_check("ollama")


@pytest.mark.skipif(
    not (os.environ.get("OPENAI_BASE_URL") and _MODEL),
    reason="openai-compat needs OPENAI_BASE_URL and DOCEXTRACT_MODEL",
)
def test_live_openai_compat():
    _run_and_check("openai-compat")
