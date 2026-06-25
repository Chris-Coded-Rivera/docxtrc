"""
End-to-end tests that run fully offline with the EchoLLM stub.
    pytest -q     (or)     python -m docextract.tests.test_extractor
"""

import json
from pathlib import Path

import pytest

from pydantic import BaseModel

from docextract import (
    DEFAULT_REGISTRY,
    PROVIDERS,
    AnthropicLLM,
    Extractor,
    EchoLLM,
    GoogleLLM,
    OllamaLLM,
    OpenAILLM,
    OpenAICompatLLM,
    build_llm,
    build_model,
)
from docextract.cli import main as cli_main
from docextract.verify import verify_evidence

SAMPLE = Path(__file__).resolve().parents[1] / "examples" / "sample_invoice.txt"


def test_schema_is_generated_from_the_map():
    model = build_model("invoice", DEFAULT_REGISTRY["invoice"])
    fields = set(model.model_fields)
    # every mapped field, plus the two we always add
    assert {"invoice_number", "vendor_name", "total_due"} <= fields
    assert {"summary", "evidence"} <= fields


def test_scripted_extraction_and_grounding():
    """Deterministic path: the model returns known values with real quotes."""
    script = {
        "classify": "invoice",
        "extract": {
            "invoice_number": "INV-2025-00842",
            "vendor_name": "ACME SUPPLY CO.",
            "invoice_date": "2025-11-03",
            "total_due": 750.60,
            "currency": "USD",
            "summary": "Invoice from ACME Supply Co. to Globex for steel brackets and hardware.",
            "evidence": [
                {"field": "invoice_number", "quote": "Invoice Number: INV-2025-00842", "page": 1},
                {"field": "vendor_name", "quote": "ACME SUPPLY CO.", "page": 1},
                {"field": "invoice_date", "quote": "Invoice Date: 2025-11-03", "page": 1},
                {"field": "total_due", "quote": "Total Due: $750.60 USD", "page": 1},
                {"field": "currency", "quote": "$750.60 USD", "page": 1},
            ],
        },
    }
    ex = Extractor(DEFAULT_REGISTRY, EchoLLM(script=script))
    result = ex.run(str(SAMPLE))

    assert result.doc_type == "invoice"
    assert result.data.invoice_number == "INV-2025-00842"
    assert result.data.total_due == 750.60
    # all three grounded fields have quotes present in the source text
    assert result.verification.grounded["invoice_number"] is True
    assert result.verification.grounded["total_due"] is True
    assert result.verification.coverage == 1.0


def test_ungrounded_value_is_flagged():
    """A value whose quote isn't in the document is caught by verification."""
    script = {
        "classify": "invoice",
        "extract": {
            "invoice_number": "INV-9999",          # fabricated
            "vendor_name": "ACME SUPPLY CO.",
            "total_due": 750.60,
            "summary": "test",
            "evidence": [
                {"field": "invoice_number", "quote": "Invoice Number: INV-9999", "page": 1},
                {"field": "total_due", "quote": "Total Due: $750.60 USD", "page": 1},
            ],
        },
    }
    ex = Extractor(DEFAULT_REGISTRY, EchoLLM(script=script))
    result = ex.run(str(SAMPLE))
    assert "invoice_number" in result.verification.ungrounded
    assert result.verification.grounded["total_due"] is True


def test_per_stage_model_injection():
    """Different backends can drive different stages (multi-LLM)."""
    ex = Extractor(
        DEFAULT_REGISTRY,
        {"classifier": EchoLLM(), "extractor": EchoLLM()},
    )
    result = ex.run(str(SAMPLE))
    assert result.doc_type == "invoice"


# --------------------------------------------------------------------------- #
# OpenAI backend: verified offline with a fake client (no key, no network).
# The point is to prove the *wiring* — that OpenAILLM reads classification from
# `.choices[0].message.content` and extraction from `.choices[0].message.parsed`,
# and that it plugs into the pipeline like any other LLM. The real API call is
# the SDK's job, not ours, so we don't test it here.
# --------------------------------------------------------------------------- #
class _FakeMessage:
    def __init__(self, *, content=None, parsed=None, refusal=None):
        self.content = content
        self.parsed = parsed
        self.refusal = refusal


class _FakeCompletion:
    def __init__(self, message):
        self.choices = [type("Choice", (), {"message": message})()]


class _FakeChatCompletions:
    """Mimics `client.chat.completions` for the two calls OpenAILLM makes."""

    def __init__(self, classify_value, extract_payload):
        self._classify_value = classify_value
        self._extract_payload = extract_payload

    def create(self, **kwargs):  # classify path → JSON string in .content
        import json

        return _FakeCompletion(_FakeMessage(content=json.dumps({"doc_type": self._classify_value})))

    def parse(self, *, response_format, **kwargs):  # extract path → validated model in .parsed
        return _FakeCompletion(_FakeMessage(parsed=response_format.model_validate(self._extract_payload)))


class _FakeOpenAIClient:
    def __init__(self, classify_value, extract_payload):
        self.chat = type(
            "Chat", (), {"completions": _FakeChatCompletions(classify_value, extract_payload)}
        )()


def test_openai_backend_wiring_offline():
    pytest.importorskip("openai")  # the lazy `from openai import OpenAI` needs the SDK installed
    payload = {
        "invoice_number": "INV-2025-00842",
        "vendor_name": "ACME SUPPLY CO.",
        "invoice_date": "2025-11-03",
        "total_due": 750.60,
        "currency": "USD",
        "summary": "Invoice from ACME Supply Co. for steel brackets and hardware.",
        "evidence": [
            {"field": "invoice_number", "quote": "Invoice Number: INV-2025-00842", "page": 1},
            {"field": "total_due", "quote": "Total Due: $750.60 USD", "page": 1},
        ],
    }
    llm = OpenAILLM(client=_FakeOpenAIClient("invoice", payload))
    result = Extractor(DEFAULT_REGISTRY, llm).run(str(SAMPLE))

    assert result.doc_type == "invoice"                      # classify wired through .content
    assert result.data.invoice_number == "INV-2025-00842"    # extract wired through .parsed
    assert result.verification.grounded["total_due"] is True


# --------------------------------------------------------------------------- #
# OpenAI-compatible backend (Azure / vLLM / LM Studio / OpenRouter / …). It's a
# thin subclass of OpenAILLM, so an injected client exercises the exact same
# request path — we just confirm the subclass plugs in unchanged...
# --------------------------------------------------------------------------- #
def test_openai_compat_reuses_openai_path_offline():
    pytest.importorskip("openai")
    payload = {
        "invoice_number": "INV-2025-00842",
        "vendor_name": "ACME SUPPLY CO.",
        "total_due": 750.60,
        "summary": "Invoice from ACME Supply Co.",
        "evidence": [{"field": "total_due", "quote": "Total Due: $750.60 USD", "page": 1}],
    }
    llm = OpenAICompatLLM("local-model", client=_FakeOpenAIClient("invoice", payload))
    result = Extractor(DEFAULT_REGISTRY, llm).run(str(SAMPLE))
    assert result.doc_type == "invoice"
    assert result.data.invoice_number == "INV-2025-00842"
    assert llm.name == "local-model"


# ...and that endpoint config is wired from base_url / OPENAI_BASE_URL, with loud
# failures when the required model or base_url is missing.
def test_openai_compat_endpoint_config_and_validation(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "not-needed")

    # base_url from the env, no network on construction
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
    llm = build_llm("openai-compat", "local-model")  # factory path
    assert isinstance(llm, OpenAICompatLLM)
    assert "localhost:8000" in str(llm.client.base_url)

    # missing model (e.g. user forgot --model) fails loudly
    with pytest.raises(ValueError):
        build_llm("openai-compat")  # model defaults to "" for this provider

    # missing base_url fails loudly
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with pytest.raises(ValueError):
        OpenAICompatLLM("local-model")


# --------------------------------------------------------------------------- #
# Google backend: same offline-wiring approach. GoogleLLM makes one call shape —
# `client.models.generate_content(...)` — and reads classification from
# `response.text` (JSON) and extraction from `response.parsed`. The fake tells
# the two paths apart the way Gemini does: extract passes a Pydantic model as the
# response_schema, classify passes an explicit enum schema.
# --------------------------------------------------------------------------- #
class _FakeGoogleResponse:
    def __init__(self, *, parsed=None, text=None):
        self.parsed = parsed
        self.text = text


class _FakeGoogleModels:
    def __init__(self, classify_value, extract_payload):
        self._classify_value = classify_value
        self._extract_payload = extract_payload

    def generate_content(self, *, model, contents, config):
        import json

        schema = config.response_schema
        if isinstance(schema, type) and issubclass(schema, BaseModel):  # extract path
            return _FakeGoogleResponse(parsed=schema.model_validate(self._extract_payload))
        return _FakeGoogleResponse(text=json.dumps({"doc_type": self._classify_value}))  # classify


class _FakeGoogleClient:
    def __init__(self, classify_value, extract_payload):
        self.models = _FakeGoogleModels(classify_value, extract_payload)


def test_google_backend_wiring_offline():
    pytest.importorskip("google.genai")  # the lazy `from google import genai` needs the SDK
    payload = {
        "invoice_number": "INV-2025-00842",
        "vendor_name": "ACME SUPPLY CO.",
        "invoice_date": "2025-11-03",
        "total_due": 750.60,
        "currency": "USD",
        "summary": "Invoice from ACME Supply Co. for steel brackets and hardware.",
        "evidence": [
            {"field": "invoice_number", "quote": "Invoice Number: INV-2025-00842", "page": 1},
            {"field": "total_due", "quote": "Total Due: $750.60 USD", "page": 1},
        ],
    }
    llm = GoogleLLM(client=_FakeGoogleClient("invoice", payload))
    result = Extractor(DEFAULT_REGISTRY, llm).run(str(SAMPLE))

    assert result.doc_type == "invoice"                      # classify wired through .text
    assert result.data.invoice_number == "INV-2025-00842"    # extract wired through .parsed
    assert result.verification.grounded["total_due"] is True


# --------------------------------------------------------------------------- #
# Ollama backend (local models): same offline-wiring approach. OllamaLLM makes
# one call shape — `client.chat(...)` — and reads both classification and
# extraction from `response.message.content` (a JSON string it validates). The
# fake tells the paths apart by the `format` schema: classify sends a one-field
# {doc_type} schema, extract sends the full generated schema.
# --------------------------------------------------------------------------- #
class _FakeOllamaResponse:
    def __init__(self, content):
        self.message = type("Message", (), {"content": content})()


class _FakeOllamaClient:
    def __init__(self, classify_value, extract_payload):
        self._classify_value = classify_value
        self._extract_payload = extract_payload

    def chat(self, *, model, messages, format, **kwargs):
        if set(format.get("properties", {})) == {"doc_type"}:  # classify path
            return _FakeOllamaResponse(json.dumps({"doc_type": self._classify_value}))
        return _FakeOllamaResponse(json.dumps(self._extract_payload))  # extract path


def test_ollama_backend_wiring_offline():
    pytest.importorskip("ollama")  # the lazy `from ollama import Client` needs the SDK
    payload = {
        "invoice_number": "INV-2025-00842",
        "vendor_name": "ACME SUPPLY CO.",
        "invoice_date": "2025-11-03",
        "total_due": 750.60,
        "currency": "USD",
        "summary": "Invoice from ACME Supply Co. for steel brackets and hardware.",
        "evidence": [
            {"field": "invoice_number", "quote": "Invoice Number: INV-2025-00842", "page": 1},
            {"field": "total_due", "quote": "Total Due: $750.60 USD", "page": 1},
        ],
    }
    llm = OllamaLLM(client=_FakeOllamaClient("invoice", payload))
    result = Extractor(DEFAULT_REGISTRY, llm).run(str(SAMPLE))

    assert result.doc_type == "invoice"                      # classify wired through .message.content
    assert result.data.invoice_number == "INV-2025-00842"    # extract validated from .message.content
    assert result.verification.grounded["total_due"] is True


def test_provider_factory_maps_names_to_backends():
    """build_llm routes provider names to backend classes and default models."""
    assert PROVIDERS["anthropic"] == (AnthropicLLM, "claude-opus-4-8")
    assert PROVIDERS["openai"][0] is OpenAILLM
    assert PROVIDERS["google"][0] is GoogleLLM
    assert PROVIDERS["ollama"][0] is OllamaLLM
    assert PROVIDERS["openai-compat"][0] is OpenAICompatLLM
    with pytest.raises(ValueError):
        build_llm("nope")  # unknown provider names fail loudly with the valid choices


def test_cli_offline_accepts_provider_flag(capsys):
    """The --provider flag parses and is ignored under --offline (no API key needed)."""
    rc = cli_main([str(SAMPLE), "--offline", "--provider", "openai"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["doc_type"] == "invoice"


if __name__ == "__main__":
    test_schema_is_generated_from_the_map()
    test_scripted_extraction_and_grounding()
    test_ungrounded_value_is_flagged()
    test_per_stage_model_injection()
    test_openai_backend_wiring_offline()
    test_openai_compat_reuses_openai_path_offline()
    test_google_backend_wiring_offline()
    test_ollama_backend_wiring_offline()
    test_provider_factory_maps_names_to_backends()
    print("all offline tests passed")
