"""
registry.py — the single source of truth.

Everything in this tool is driven by one declarative map: doc type -> what it
means and what to pull out of it. The schema, the prompt, the routing, and the
validation are all *generated* from this map. Adding a new document type is a
data change here (or in a YAML file), never a code change.

Two tiny dataclasses are all the structure we need:

    FieldSpec  — one field to extract (its type, and a description that doubles
                 as the instruction we give the model for that field)
    DocSpec    — one document type (its purpose, a domain "system hint" that
                 switches the model into the right expert mindset, and its fields)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any


# Map the small set of type names we allow in the declarative file to real
# Python types. Keeping this list short keeps the generated JSON Schema simple,
# which is exactly what the model's structured-output decoder wants.
TYPES: dict[str, type] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "date": date,
    "datetime": datetime,
}


@dataclass(frozen=True)
class FieldSpec:
    """One piece of information to extract from a document."""

    type: type
    description: str          # doubles as the per-field instruction to the model
    required: bool = True


@dataclass(frozen=True)
class DocSpec:
    """One document type and everything the tool needs to handle it."""

    purpose: str              # why this document matters / what it's used for
    system_hint: str          # the "domain brain" — switches model expertise
    fields: dict[str, FieldSpec]


# --------------------------------------------------------------------------- #
# A small built-in registry so the package is runnable out of the box and the
# tests need no external files. Real deployments will usually load a YAML file
# (see load_registry) so non-engineers can own the map.
# --------------------------------------------------------------------------- #
DEFAULT_REGISTRY: dict[str, DocSpec] = {
    "invoice": DocSpec(
        purpose="Accounts-payable processing and three-way matching.",
        system_hint=(
            "You are an accounts-payable specialist. Read amounts in the "
            "document's own currency and never convert. Treat the largest "
            "labelled total as the amount payable unless a clearer 'balance "
            "due' is present."
        ),
        fields={
            "invoice_number": FieldSpec(str, "The unique invoice identifier."),
            "vendor_name": FieldSpec(str, "Legal name of the issuing vendor."),
            "invoice_date": FieldSpec(date, "Date the invoice was issued.", required=False),
            "total_due": FieldSpec(float, "Total amount payable, numeric only."),
            "currency": FieldSpec(str, "ISO currency code, e.g. USD.", required=False),
        },
    ),
    "receipt": DocSpec(
        purpose="Expense reconciliation for reimbursements.",
        system_hint=(
            "You are an expense-reconciliation clerk. Identify the merchant and "
            "the grand total actually paid, including tax."
        ),
        fields={
            "merchant": FieldSpec(str, "Name of the merchant or store."),
            "total_paid": FieldSpec(float, "Grand total paid, numeric only."),
            "purchase_date": FieldSpec(date, "Date of purchase.", required=False),
        },
    ),
    "contract": DocSpec(
        purpose="Obligation tracking and renewal alerts.",
        system_hint=(
            "You are a contracts paralegal. Identify the named parties and the "
            "key dates. Do not infer terms that are not written in the text."
        ),
        fields={
            "party_a": FieldSpec(str, "First named party to the agreement."),
            "party_b": FieldSpec(str, "Second named party to the agreement."),
            "effective_date": FieldSpec(date, "Date the agreement takes effect.", required=False),
            "term_months": FieldSpec(int, "Length of the term in months.", required=False),
        },
    ),
}


def load_registry(source: str | dict[str, Any]) -> dict[str, DocSpec]:
    """
    Build a registry from a YAML/JSON file path or an already-loaded dict.

    The declarative shape is intentionally boring:

        invoice:
          purpose: "Accounts-payable processing."
          system_hint: "You are an AP specialist..."
          fields:
            invoice_number: { type: str, description: "Unique id." }
            total_due:      { type: float, description: "Amount payable." }
            due_date:       { type: date, description: "When due.", required: false }
    """
    if isinstance(source, str):
        import yaml  # imported lazily so PyYAML is only needed if you use files

        with open(source, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    else:
        raw = source

    registry: dict[str, DocSpec] = {}
    for doc_type, spec in raw.items():
        fields = {
            name: FieldSpec(
                type=TYPES[fs["type"]],
                description=fs["description"],
                required=fs.get("required", True),
            )
            for name, fs in spec["fields"].items()
        }
        registry[doc_type] = DocSpec(
            purpose=spec["purpose"],
            system_hint=spec["system_hint"],
            fields=fields,
        )
    return registry
