"""Core records shared by ingestion, indexing, retrieval and the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    doc_id: str
    company: str
    form: str  # 10-K, 10-Q, 8-K, EARNINGS, ...
    period: str  # "2018", "2023Q2"
    source: str  # file path or URL
    ticker: str | None = None
    sector: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def fiscal_year(self) -> int | None:
        digits = self.period[:4]
        return int(digits) if digits.isdigit() else None

    @property
    def label(self) -> str:
        return f"{self.company} {self.form} {self.period}"


@dataclass
class Page:
    doc_id: str
    page_no: int  # 0-based page index in the source file
    text: str
    section: str | None = None  # e.g. "Item 8. Financial Statements and Supplementary Data"
    title: str | None = None  # e.g. "Consolidated Statements of Cash Flows"


@dataclass
class Chunk:
    chunk_id: int
    doc_id: str
    page_no: int
    header: str  # contextual header prepended for retrieval and shown to the model
    text: str
    section: str | None = None
    title: str | None = None


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int
    sources: dict[str, int] = field(default_factory=dict)  # retriever -> rank

    @property
    def citation(self) -> str:
        return f"{self.chunk.doc_id} p.{self.chunk.page_no + 1}"
