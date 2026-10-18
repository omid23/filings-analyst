"""PDF filings to pages of layout-aware text.

Financial statements in PDF filings extract as one token per line with plain
text extraction ("Inventories\\n(509)\\n(387)\\n57"), which separates each
row label from its numbers. Here words are regrouped by vertical position so
every table row reads as one line ("Inventories (509) (387) 57"). Both
keyword search and the model benefit, because the label and its values stay
together.
"""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf

from ..models import Page

_ITEM = re.compile(r"^\s*ITEM\s+(\d{1,2}[A-C]?)\s*[.:\-–—]?\s*(.{0,90})$", re.I)
_STATEMENT = re.compile(
    r"\b(?:CONSOLIDATED|COMBINED|CONDENSED)?\s*(?:STATEMENTS?|BALANCE\s+SHEETS?)\s+OF\s+"
    r"(?:CASH\s+FLOWS?|OPERATIONS|INCOME|EARNINGS|COMPREHENSIVE\s+(?:INCOME|LOSS)|FINANCIAL\s+POSITION|"
    r"(?:CHANGES\s+IN\s+)?(?:SHAREHOLDERS|STOCKHOLDERS|SHAREOWNERS)['’]?\s+(?:EQUITY|DEFICIT))"
    r"|\b(?:CONSOLIDATED\s+|CONDENSED\s+)*BALANCE\s+SHEETS?\b",
    re.I,
)
_SPACES = re.compile(r"[ \t]+")


def layout_lines(page: pymupdf.Page) -> list[str]:
    """Regroup words into visual lines using vertical overlap rather than PDF line order."""
    words = page.get_text("words")
    if not words:
        return []
    words.sort(key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    lines: list[list[tuple]] = []
    current: list[tuple] = []
    cur_mid = cur_h = 0.0
    for w in words:
        mid, h = (w[1] + w[3]) / 2, max(w[3] - w[1], 1.0)
        if current and abs(mid - cur_mid) <= 0.5 * max(h, cur_h):
            current.append(w)
            cur_mid = (cur_mid * (len(current) - 1) + mid) / len(current)
            cur_h = max(cur_h, h)
        else:
            if current:
                lines.append(current)
            current, cur_mid, cur_h = [w], mid, h
    if current:
        lines.append(current)
    out = []
    for line in lines:
        line.sort(key=lambda w: w[0])
        out.append(_SPACES.sub(" ", " ".join(w[4] for w in line)).strip())
    return [line for line in out if line]


_SMALL = {"of", "in", "and", "the", "for", "to"}


def _title_case(text: str) -> str:
    words = text.lower().split()
    return " ".join(w if (i and w in _SMALL) else w[:1].upper() + w[1:] for i, w in enumerate(words))


def detect_title(lines: list[str]) -> str | None:
    """Financial statement title near the top of the page, if any."""
    for line in lines[:12]:
        m = _STATEMENT.search(line)
        if m and _is_heading(line, m):
            return _title_case(_SPACES.sub(" ", m.group(0)).strip())
    return None


def _is_heading(line: str, m: re.Match) -> bool:
    """The title is the line, not a mention inside prose ("...presented in the balance sheet and...")."""
    rest = re.sub(r"\([^)]*\)", "", line[: m.start()] + line[m.end():]).strip()
    return len(rest) <= 40 and not rest.endswith((".", ",", ";"))


def detect_item(lines: list[str]) -> str | None:
    for line in lines[:15]:
        m = _ITEM.match(line)
        if m and not re.search(r"\.{4,}|\s\d+\s*$", line):  # skip table-of-contents rows
            rest = m.group(2).strip(" .")
            return f"Item {m.group(1).upper()}. {rest}".strip(" .") if rest else f"Item {m.group(1).upper()}"
    return None


def extract_pages(path: str | Path, doc_id: str) -> list[Page]:
    pages: list[Page] = []
    section: str | None = None
    with pymupdf.open(str(path)) as doc:
        for i, page in enumerate(doc):
            lines = layout_lines(page)
            item = detect_item(lines)
            if item:
                section = item
            pages.append(Page(doc_id, i, "\n".join(lines), section=section, title=detect_title(lines)))
    return pages
