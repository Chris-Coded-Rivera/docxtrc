"""
ingest.py — get faithful text out of a file, deterministically.

This stage is deliberately *not* an LLM job. Pulling the native text layer is
cheaper and more faithful than asking a model to retype a document, and it
keeps the model focused on the part it is actually good at: understanding and
locating. We only fall back to model/OCR reading for documents that have no
text layer (scans), and that fallback is left as an explicit extension point
rather than hidden magic.
"""

from __future__ import annotations

from pathlib import Path


def read_document(path: str) -> str:
    """Return plain text for a document, routed by file extension."""
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix in (".txt", ".md", ".csv", ".json"):
        return p.read_text(encoding="utf-8", errors="replace")

    if suffix == ".pdf":
        return _read_pdf(p)

    if suffix in (".docx",):
        return _read_docx(p)

    raise ValueError(
        f"Unsupported file type {suffix!r}. Supported: .txt .md .csv .json .pdf .docx. "
        "For scanned/image PDFs, plug an OCR or vision-model reader in here."
    )


def _read_pdf(p: Path) -> str:
    try:
        import pdfplumber  # pip install pdfplumber
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError("Reading PDFs needs pdfplumber: pip install pdfplumber") from e

    pages = []
    with pdfplumber.open(str(p)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    text = "\n\n".join(pages).strip()
    if not text:
        raise ValueError(
            f"{p.name} has no extractable text layer (likely a scan). "
            "Add an OCR/vision fallback to handle it."
        )
    return text


def _read_docx(p: Path) -> str:
    try:
        import docx  # pip install python-docx
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError("Reading .docx needs python-docx: pip install python-docx") from e

    document = docx.Document(str(p))
    return "\n".join(par.text for par in document.paragraphs)
