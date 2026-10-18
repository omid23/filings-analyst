"""EDGAR HTML filings to pseudo-pages.

EDGAR's primary documents are HTML. Tables become pipe-separated rows (one
row per line), other blocks become paragraphs, and the result is cut into
pages at the filing's own page breaks when present, otherwise every ~3,500
characters, so citations work the same way as for PDFs.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

from ..models import Page
from .pdf import detect_item, detect_title

_BLOCK = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section"}
_SKIP = {"script", "style", "head", "title", "ix:header"}
PAGE_BREAK = "\f"


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0
        self.in_cell = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP:
            self.skip += 1
        style = (dict(attrs).get("style") or "").replace(" ", "").lower()
        if "page-break-before:always" in style or "break-before:page" in style:
            self.parts.append(PAGE_BREAK)
        if tag in ("td", "th"):
            if self.in_cell:
                self.parts.append(" | ")
            self.in_cell = True
        elif tag in _BLOCK:
            self.parts.append("\n")
            if tag == "tr":
                self.in_cell = False

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP and self.skip:
            self.skip -= 1
        if tag == "hr":
            self.parts.append(PAGE_BREAK)
        if tag in _BLOCK:
            self.parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_data(self, data: str) -> None:
        if not self.skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    text = "".join(parser.parts).replace("\xa0", " ")
    return PAGE_BREAK.join(_clean(part) for part in text.split(PAGE_BREAK))


def _clean(text: str) -> str:
    lines: list[str] = []
    for raw in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", raw).strip()
        line = re.sub(r"(?:\|\s*){2,}", "| ", line).strip(" |")  # collapse empty cells
        if line or (lines and lines[-1]):
            lines.append(line)
    return "\n".join(lines)


def split_pages(text: str, doc_id: str, target_chars: int = 3500) -> list[Page]:
    raw_pages = [p for p in text.split(PAGE_BREAK) if p.strip()]
    if len(raw_pages) <= 1:
        raw_pages, buf = [], []
        size = 0
        for line in text.split("\n"):
            buf.append(line)
            size += len(line) + 1
            if size >= target_chars and not line.strip():
                raw_pages.append("\n".join(buf))
                buf, size = [], 0
        if buf:
            raw_pages.append("\n".join(buf))
    pages, section = [], None
    for i, body in enumerate(raw_pages):
        lines = [ln for ln in body.split("\n") if ln.strip()]
        item = detect_item(lines)
        if item:
            section = item
        pages.append(Page(doc_id, i, "\n".join(lines), section=section, title=detect_title(lines)))
    return pages


def extract_html_pages(html: str, doc_id: str) -> list[Page]:
    return split_pages(html_to_text(html), doc_id)
