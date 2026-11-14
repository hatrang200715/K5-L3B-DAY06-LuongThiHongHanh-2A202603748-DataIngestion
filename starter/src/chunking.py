"""Implement fixed-size and structure-aware chunking."""
import re

from .models import Chunk, DocumentPage


#split text into overlapping windows and retain page metadata.
def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Split a page into overlapping character windows with source metadata."""
    if size <= 0:
        raise ValueError("size must be greater than zero")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be non-negative and smaller than size")
    if not page.text or not page.text.strip():
        return []

    chunks = []
    step = size - overlap
    for index, start in enumerate(range(0, len(page.text), step)):
        text = page.text[start : start + size]
        chunks.append(
            Chunk(
                chunk_id=f"{page.document_id}-p{page.page}-fixed-{index:04d}",
                document_id=page.document_id,
                source=page.source,
                page=page.page,
                text=text,
                metadata=dict(page.metadata),
            )
        )
    return chunks


#group paragraphs, split oversized paragraphs and retain metadata.
def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Group paragraphs up to max_size, splitting paragraphs that exceed it."""
    if max_size <= 0:
        raise ValueError("max_size must be greater than zero")

    normalized_text = page.text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n[ \t]*\n+", normalized_text)
        if paragraph.strip()
    ]
    chunks = []
    current: list[str] = []
    current_size = 0

    def append_chunk(text: str, index: int) -> None:
        chunks.append(
            Chunk(
                chunk_id=f"{page.document_id}-p{page.page}-paragraph-{index:04d}",
                document_id=page.document_id,
                source=page.source,
                page=page.page,
                text=text,
                metadata=dict(page.metadata),
            )
        )

    def flush() -> None:
        nonlocal current_size
        if current:
            append_chunk("\n\n".join(current), len(chunks))
            current.clear()
            current_size = 0

    for paragraph in paragraphs:
        if len(paragraph) > max_size:
            flush()
            for start in range(0, len(paragraph), max_size):
                append_chunk(paragraph[start : start + max_size], len(chunks))
            continue

        added_size = len(paragraph) + (2 if current else 0)
        if current and current_size + added_size > max_size:
            flush()
            added_size = len(paragraph)
        current.append(paragraph)
        current_size += added_size

    flush()
    return chunks
