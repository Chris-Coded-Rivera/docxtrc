"""
cli.py — run the extractor from a shell, cron job, or CI step.

    python -m docextract.cli path/to/invoice.pdf
    python -m docextract.cli path/to/doc.txt --type invoice --registry specs.yaml
    python -m docextract.cli path/to/doc.txt --provider openai          # use OpenAI
    python -m docextract.cli path/to/doc.txt --provider openai --model gpt-4o-mini
    python -m docextract.cli path/to/doc.txt --offline                  # no API key needed

Prints a JSON result to stdout, so it pipes cleanly into jq or another step.
"""

from __future__ import annotations

import argparse
import json
import sys

from .llm import PROVIDERS, EchoLLM, build_llm
from .pipeline import Extractor
from .registry import DEFAULT_REGISTRY, load_registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Registry-driven document extractor.")
    parser.add_argument("path", help="Path to the document.")
    parser.add_argument("--type", dest="doc_type", default=None, help="Force a document type.")
    parser.add_argument("--registry", default=None, help="Path to a YAML registry file.")
    parser.add_argument(
        "--provider",
        default="anthropic",
        choices=sorted(PROVIDERS),  # kept in sync with the backend registry
        help="LLM provider to use (ignored with --offline).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model id. Defaults to the chosen provider's default model.",
    )
    parser.add_argument("--offline", action="store_true", help="Use the offline EchoLLM stub.")
    args = parser.parse_args(argv)

    registry = load_registry(args.registry) if args.registry else DEFAULT_REGISTRY
    backend = EchoLLM() if args.offline else build_llm(args.provider, args.model)
    extractor = Extractor(registry, backend)

    result = extractor.run(args.path, doc_type=args.doc_type)
    out = {
        "doc_type": result.doc_type,
        "data": result.data.model_dump(mode="json"),
        "grounding": {
            "coverage": round(result.verification.coverage, 3),
            "ungrounded_fields": result.verification.ungrounded,
        },
    }
    json.dump(out, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
