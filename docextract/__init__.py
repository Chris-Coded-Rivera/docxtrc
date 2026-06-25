"""
docextract — a registry-driven document extraction harness.

Public API:

    from docextract import Extractor, load_registry, DEFAULT_REGISTRY
    from docextract import AnthropicLLM, EchoLLM

    ex = Extractor(DEFAULT_REGISTRY, AnthropicLLM("claude-opus-4-8"))
    result = ex.run("invoice.pdf")
    print(result.doc_type, result.data, result.verification.coverage)
"""

from .registry import DEFAULT_REGISTRY, DocSpec, FieldSpec, load_registry
from .schema import Evidence, build_model, system_prompt
from .llm import (
    LLM,
    PROVIDERS,
    AnthropicLLM,
    OpenAILLM,
    OpenAICompatLLM,
    GoogleLLM,
    OllamaLLM,
    EchoLLM,
    build_llm,
)
from .ingest import read_document
from .verify import VerificationReport, verify_evidence
from .pipeline import Context, Extractor, ExtractionResult

__all__ = [
    "DEFAULT_REGISTRY",
    "DocSpec",
    "FieldSpec",
    "load_registry",
    "Evidence",
    "build_model",
    "system_prompt",
    "LLM",
    "PROVIDERS",
    "AnthropicLLM",
    "OpenAILLM",
    "OpenAICompatLLM",
    "GoogleLLM",
    "OllamaLLM",
    "EchoLLM",
    "build_llm",
    "read_document",
    "VerificationReport",
    "verify_evidence",
    "Context",
    "Extractor",
    "ExtractionResult",
]
