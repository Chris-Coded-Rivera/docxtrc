"""
schema.py — turn a DocSpec into a Pydantic model at runtime.

This is the heart of "the map drives everything". We never hand-write an
extraction class per document type; we generate one from the spec with
pydantic.create_model. The generated model is what we hand to the LLM's
structured-output decoder, so the model is *forced* to return exactly this
shape — no JSON parsing, no schema-repair retries.

We add two things to every generated model:

  * `summary`   — a short plain-language summary of the document.
  * `evidence`  — a flat list of {field, quote, page}. The model tells us the
                  verbatim text it pulled each value from, so we can verify
                  values deterministically afterwards (see verify.py). We keep
                  this as a *list* (not a dict keyed by field name) because
                  open-ended object keys make the constrained-decoding grammar
                  larger and slower; a flat list stays well inside the limits.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, create_model

from .registry import DocSpec


class Evidence(BaseModel):
    # extra="forbid" makes Pydantic emit "additionalProperties": false, which
    # structured outputs require.
    model_config = ConfigDict(extra="forbid")

    field: str = Field(description="Name of the extracted field this supports.")
    quote: str = Field(description="Verbatim span from the document, copied exactly.")
    page: Optional[int] = Field(default=None, description="1-based page number, if known.")


def build_model(doc_type: str, spec: DocSpec, *, ground: bool = True) -> type[BaseModel]:
    """Compile a DocSpec into a Pydantic model named e.g. `InvoiceExtraction`."""
    fields: dict[str, tuple] = {}

    for name, fs in spec.fields.items():
        annotation = fs.type if fs.required else Optional[fs.type]
        default = ... if fs.required else None  # `...` means "required" in Pydantic
        fields[name] = (annotation, Field(default, description=fs.description))

    fields["summary"] = (
        str,
        Field(..., description="A 2-3 sentence plain-language summary of the document."),
    )
    if ground:
        fields["evidence"] = (
            list[Evidence],
            Field(default_factory=list, description="Supporting quote for each extracted field."),
        )

    model = create_model(
        f"{doc_type.title().replace('_', '')}Extraction",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )
    return model


def system_prompt(spec: DocSpec) -> str:
    """Assemble the extraction system prompt from the spec's domain hint."""
    return (
        f"{spec.system_hint}\n\n"
        f"Document purpose: {spec.purpose}\n\n"
        "Extract the requested fields from the document the user provides. "
        "Use only what is written in the document; if a value is genuinely "
        "absent, leave optional fields null rather than guessing. For every "
        "field you fill, add an evidence entry whose `quote` is copied verbatim "
        "from the document."
    )
