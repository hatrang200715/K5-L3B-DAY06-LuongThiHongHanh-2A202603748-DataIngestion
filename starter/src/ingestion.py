"""Read and normalize source data."""
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

from .models import DocumentPage


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Extract each physical PDF page, retaining blank pages and page numbers."""
    reader = PdfReader(str(path))
    pages = []
    for page_number, pdf_page in enumerate(reader.pages, start=1):
        extracted = pdf_page.extract_text()
        text = normalize_text(extracted or "")
        pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=page_number,
                text=text,
                metadata={
                    "extraction_status": "extracted" if text else "no_text",
                },
            )
        )
    return pages


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace while preserving line/paragraph breaks."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[^\S\n]+", " ", line).strip()
        lines.append(line)
    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()
