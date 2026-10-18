"""Tools the analyst agent can call, and the evidence ledger they write to.

Every passage a tool returns gets a source id (S1, S2, ...). The final answer
must cite those ids, and the verifier checks each cited id was actually
retrieved and each number appears in a cited source or the calculator log.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..index.retrieve import Searcher
from .llm import ToolSpec
from .numbers import CalcError, calculate


@dataclass
class Source:
    id: str
    doc_id: str
    page_no: int  # 0-based
    header: str
    text: str

    @property
    def citation(self) -> str:
        return f"{self.doc_id}, p. {self.page_no + 1}"


@dataclass
class Calculation:
    expression: str
    result: float


@dataclass
class Ledger:
    sources: dict[str, Source] = field(default_factory=dict)
    calculations: list[Calculation] = field(default_factory=list)
    _by_key: dict[tuple, str] = field(default_factory=dict)

    def add_source(self, doc_id: str, page_no: int, header: str, text: str, part: str = "") -> Source:
        key = (doc_id, page_no, part or text[:80])
        if key in self._by_key:
            src = self.sources[self._by_key[key]]
            if len(text) > len(src.text):
                src.text = text
            return src
        sid = f"S{len(self.sources) + 1}"
        src = Source(sid, doc_id, page_no, header, text)
        self.sources[sid] = src
        self._by_key[key] = sid
        return src


MAX_PAGE_CHARS = 7000


class Toolbox:
    def __init__(self, searcher: Searcher, ledger: Ledger | None = None, k: int = 6):
        self.searcher, self.ledger, self.k = searcher, ledger or Ledger(), k
        self._handlers: dict[str, Callable[[dict], str]] = {
            "list_filings": self.list_filings,
            "search_filings": self.search_filings,
            "read_page": self.read_page,
            "calculate": self.calculate,
        }

    @property
    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec("list_filings", "List the filings in the library (id, company, form, fiscal period, pages). "
                     "Use it when you are unsure which filing covers the company and period in the question.",
                     {"type": "object", "properties": {"company": {"type": "string", "description": "Optional company name filter"}}}),
            ToolSpec("search_filings", "Hybrid keyword + semantic search over the filings. The library automatically "
                     "narrows to the company and fiscal period named in the query; include them. Returns passages with "
                     "source ids like [S3] that you must cite. Name the statement or section you need (e.g. 'cash flow "
                     "statement capital expenditures') for better results.",
                     {"type": "object", "properties": {
                         "query": {"type": "string"},
                         "doc_ids": {"type": "array", "items": {"type": "string"},
                                     "description": "Optional: restrict to these filing ids"},
                     }, "required": ["query"]}),
            ToolSpec("read_page", "Read a full page of a filing, e.g. a whole financial statement table. Pages are 1-based "
                     "as shown in search results.",
                     {"type": "object", "properties": {"doc_id": {"type": "string"}, "page": {"type": "integer"}},
                      "required": ["doc_id", "page"]}),
            ToolSpec("calculate", "Evaluate an arithmetic expression exactly, e.g. '(1577 - 1373) / 1373 * 100'. Use it for "
                     "every computation; do not compute in your head. Supports + - * / ** % ( ) and abs, round, min, max, "
                     "sum, avg, sqrt.",
                     {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}),
            ToolSpec("submit_answer", "Submit the final answer. Cite sources inline as [S1], [S2]. Every figure must come "
                     "from a cited source or a calculate result.",
                     {"type": "object", "properties": {
                         "answer": {"type": "string", "description": "Concise answer with inline [S#] citations"},
                         "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                     }, "required": ["answer"]}),
        ]

    def call(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        handler = self._handlers.get(name)
        if handler is None:
            return f"Unknown tool '{name}'.", True
        try:
            return handler(args), False
        except (CalcError, KeyError, TypeError, ValueError) as exc:
            return f"Error: {exc}", True

    # -- handlers -------------------------------------------------------------------------

    def list_filings(self, args: dict) -> str:
        company = (args.get("company") or "").lower()
        rows = []
        for d in self.searcher.store.documents():
            if company and company not in d.company.lower() and company not in d.doc_id.lower():
                continue
            rows.append(f"{d.doc_id} | {d.company} | {d.form} | {d.period} | {self.searcher.store.page_count(d.doc_id)} pages")
        if not rows:
            return "No filings match."
        head = rows[:80]
        more = f"\n... {len(rows) - 80} more" if len(rows) > 80 else ""
        return "doc_id | company | form | period | pages\n" + "\n".join(head) + more

    def search_filings(self, args: dict) -> str:
        query = str(args["query"]).strip()
        doc_ids = args.get("doc_ids") or None
        result = self.searcher.search(query, k=self.k, doc_ids=doc_ids)
        lines = []
        if result.resolution.reason and not doc_ids:
            lines.append(f"(filings searched: {result.resolution.reason})")
        shown = 0
        for hit in result.hits:
            c = hit.chunk
            src = self.ledger.add_source(c.doc_id, c.page_no, c.header, c.text, part=str(c.chunk_id))
            lines.append(f"[{src.id}] {c.header}\n{c.text[:1200]}")
            shown += 1
            if shown >= self.k:
                break
        return "\n\n".join(lines) if shown else "No results. Try other terms or list_filings."

    def read_page(self, args: dict) -> str:
        doc_id, page = str(args["doc_id"]), int(args["page"])
        p = self.searcher.store.page(doc_id, page - 1)
        if p is None:
            n = self.searcher.store.page_count(doc_id)
            raise ValueError(f"{doc_id} has no page {page}" + (f" (it has {n} pages)" if n else " (unknown filing id)"))
        doc = self.searcher.store.document(doc_id)
        header = f"{doc.label if doc else doc_id} | page {page}" + (f" | {p.title}" if p.title else "")
        text = p.text[:MAX_PAGE_CHARS]
        src = self.ledger.add_source(doc_id, page - 1, header, text, part="page")
        return f"[{src.id}] {header}\n{text}"

    def calculate(self, args: dict) -> str:
        expr = str(args["expression"])
        value = calculate(expr)
        self.ledger.calculations.append(Calculation(expr, value))
        return json.dumps({"expression": expr, "result": round(value, 10)})
