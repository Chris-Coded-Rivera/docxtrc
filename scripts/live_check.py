"""
live_check.py — run one real extraction end-to-end against a live provider.

This is the human-facing companion to tests/test_live.py: point it at a provider
you have credentials for, and it runs the full pipeline (ingest → classify →
schema → extract → verify) on a document, prints a readable report, and exits
non-zero if the result doesn't look like a real extraction. Handy for smoke-
testing a backend against an actual endpoint without the pytest harness.

    # any provider the package supports (anthropic | openai | google | ollama | openai-compat)
    ANTHROPIC_API_KEY=sk-... uv run python scripts/live_check.py --provider anthropic
    uv run python scripts/live_check.py --provider ollama --model llama3.1

    # a local OpenAI-compatible server (e.g. vLLM):
    OPENAI_BASE_URL=http://localhost:8000/v1 \
      uv run python scripts/live_check.py --provider openai-compat --model <served-model-id>

Exit code is 0 on a sane extraction, 1 otherwise — so it drops into CI or a shell
`&&` chain.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from docextract import DEFAULT_REGISTRY, Extractor, build_llm, load_registry

# Default document + the values a correct extraction must recover from it.
SAMPLE = Path(__file__).resolve().parents[1] / "examples" / "sample_invoice.txt"
EXPECT = {"doc_type": "invoice", "invoice_number": "INV-2025-00842", "total_due": 750.60}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live end-to-end check against a real provider.")
    parser.add_argument("--provider", default="anthropic",
                        help="anthropic | openai | google | ollama | openai-compat")
    parser.add_argument("--model", default=None, help="Model id (defaults to provider's default).")
    parser.add_argument("--file", default=str(SAMPLE), help="Document to extract (default: sample invoice).")
    parser.add_argument("--registry", default=None, help="Path to a YAML registry (default: built-in).")
    parser.add_argument("--type", dest="doc_type", default=None, help="Force a document type.")
    parser.add_argument("--strict", action="store_true",
                        help="Also require the sample's known values (only valid with the default file).")
    args = parser.parse_args(argv)

    registry = load_registry(args.registry) if args.registry else DEFAULT_REGISTRY
    backend = build_llm(args.provider, args.model)

    print(f"→ provider={args.provider} model={backend.name} file={args.file}")
    result = Extractor(registry, backend).run(args.file, doc_type=args.doc_type)

    data = result.data.model_dump(mode="json")
    print(f"  doc_type   : {result.doc_type}")
    print("  fields     :")
    for k, v in data.items():
        if k not in ("summary", "evidence"):
            print(f"    - {k}: {v}")
    print(f"  coverage   : {result.verification.coverage:.2f}")
    print(f"  ungrounded : {result.verification.ungrounded or 'none'}")

    # Minimal sanity gate: the model must have produced *something* grounded, not
    # an empty shell. --strict additionally pins the sample's known values.
    ok = bool(result.doc_type) and result.verification.coverage > 0
    checks = [("doc_type present", bool(result.doc_type)),
              ("coverage > 0", result.verification.coverage > 0)]
    if args.strict and Path(args.file).resolve() == SAMPLE.resolve():
        strict_ok = (
            result.doc_type == EXPECT["doc_type"]
            and getattr(result.data, "invoice_number", None) == EXPECT["invoice_number"]
            and abs((getattr(result.data, "total_due", 0) or 0) - EXPECT["total_due"]) < 0.001
        )
        checks.append(("known sample values", strict_ok))
        ok = ok and strict_ok

    print("  result     :", "PASS ✅" if ok else "FAIL ❌")
    for name, passed in checks:
        print(f"    [{'x' if passed else ' '}] {name}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
