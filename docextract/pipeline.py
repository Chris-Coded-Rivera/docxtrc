"""
pipeline.py — the harness.

A document flows through a few small, composable stages:

    ingest -> classify -> build schema -> extract(+summarise) -> verify

Each stage that needs intelligence takes an `LLM` by injection, so:

  * "swap the core model"     = construct the Extractor with a different backend
  * "multi-agent / multi-LLM" = give different stages different backends, e.g. a
                                cheap model to classify and a strong one to
                                extract; or turn on the optional `critic` to
                                have a second model challenge low-confidence
                                fields.

Summarisation is folded into the extraction schema, so the normal path is two
model calls (classify + extract). That is the simple default; the critic adds a
third call only when you ask for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel

from .ingest import read_document
from .llm import LLM
from .registry import DocSpec
from .schema import build_model, system_prompt
from .verify import VerificationReport, verify_evidence


@dataclass
class ExtractionResult:
    doc_type: str
    data: BaseModel                      # the generated, populated Pydantic model
    verification: VerificationReport
    source_chars: int


@dataclass
class Context:
    """Carries state across stages. Handy for debugging and for adding stages."""

    path: Optional[str] = None
    text: str = ""
    doc_type: Optional[str] = None
    data: Optional[BaseModel] = None


class Extractor:
    """
    Orchestrates the stages.

    models: either a single LLM used for every stage, or a dict assigning a
            backend per role, e.g. {"classifier": haiku, "extractor": opus}.
            Missing roles fall back to models["default"] or the single LLM.
    """

    def __init__(self, registry: dict[str, DocSpec], models: "LLM | dict[str, LLM]"):
        self.registry = registry
        if isinstance(models, dict):
            self.models = models
        else:
            self.models = {"default": models}

    def _model(self, role: str) -> LLM:
        return self.models.get(role) or self.models.get("default") or next(iter(self.models.values()))

    # -- public API -------------------------------------------------------- #
    def run(self, path: str, *, doc_type: str | None = None) -> ExtractionResult:
        """Extract from a file path."""
        return self.run_text(read_document(path), doc_type=doc_type, path=path)

    def run_text(self, text: str, *, doc_type: str | None = None,
                 path: str | None = None) -> ExtractionResult:
        """Extract from raw text (useful when ingestion happens upstream)."""
        ctx = Context(path=path, text=text, doc_type=doc_type)

        # 1. classify (skipped if the caller already knows the type)
        if ctx.doc_type is None:
            menu = {t: s.purpose for t, s in self.registry.items()}
            ctx.doc_type = self._model("classifier").classify(document=ctx.text, doc_types=menu)
        if ctx.doc_type not in self.registry:
            raise ValueError(f"Unknown doc_type {ctx.doc_type!r}; not in registry.")

        # 2. build the schema for this type from the map
        spec = self.registry[ctx.doc_type]
        schema = build_model(ctx.doc_type, spec)

        # 3. extract (+ summarise, folded into the schema)
        ctx.data = self._model("extractor").extract(
            system=system_prompt(spec), document=ctx.text, schema=schema
        )

        # 4. verify groundedness against the source
        report = verify_evidence(ctx.data, ctx.text)

        return ExtractionResult(
            doc_type=ctx.doc_type,
            data=ctx.data,
            verification=report,
            source_chars=len(ctx.text),
        )
