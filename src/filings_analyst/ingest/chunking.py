"""Page-aware chunking with contextual headers.

Chunks never cross a page boundary, so every chunk has an exact page
citation. Each chunk carries a header naming the company, form, period,
section and statement title. The header is indexed with the text, so a query
like "3M FY2018 cash flow statement capex" can match a chunk whose body only
says "Purchases of property, plant and equipment (1,577)".
"""

from __future__ import annotations

from ..models import Chunk, Document, Page


def make_header(doc: Document, page: Page) -> str:
    parts = [doc.company, doc.form, f"FY{doc.period}" if doc.period.isdigit() else doc.period]
    if page.section:
        parts.append(page.section)
    if page.title:
        parts.append(page.title)
    parts.append(f"page {page.page_no + 1}")
    return " | ".join(parts)


def chunk_page(doc: Document, page: Page, start_id: int, max_words: int = 220, overlap: int = 40) -> list[Chunk]:
    lines = [ln for ln in page.text.split("\n") if ln.strip()]
    if not lines:
        return []
    header = make_header(doc, page)
    chunks: list[Chunk] = []
    buf: list[str] = []
    count = 0
    fresh = 0  # lines added since the last flush (overlap lines do not count)

    def emit() -> None:
        chunks.append(Chunk(start_id + len(chunks), doc.doc_id, page.page_no, header, "\n".join(buf),
                            page.section, page.title))

    for line in lines:
        words = len(line.split())
        if count + words > max_words and fresh:
            emit()
            # carry the last few lines over so a table heading stays with its first rows
            tail: list[str] = []
            n = 0
            for prev in reversed(buf):
                n += len(prev.split())
                if n > overlap:
                    break
                tail.insert(0, prev)
            buf, count, fresh = tail, sum(len(t.split()) for t in tail), 0
        buf.append(line)
        count += words
        fresh += 1
    if fresh:
        emit()
    return chunks


def chunk_document(doc: Document, pages: list[Page], start_id: int = 0, **kw) -> list[Chunk]:
    out: list[Chunk] = []
    for page in pages:
        out.extend(chunk_page(doc, page, start_id + len(out), **kw))
    return out
