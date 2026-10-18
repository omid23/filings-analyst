"""FinanceBench (Patronus AI, 2023): 150 open questions over 361 real filings.

Each question names its filing and the evidence page(s), so retrieval can be
scored exactly (did we surface the page an analyst used?) without any LLM.
The PDFs are not redistributed here; point the loader at a clone of
github.com/patronus-ai/financebench.
"""

from __future__ import annotations

import json
import re
import zlib
from dataclasses import dataclass
from pathlib import Path

from ..models import Document

FORMS = {"10k": "10-K", "10k_annualreport": "10-K", "10q": "10-Q", "8k": "8-K", "earnings": "EARNINGS"}

# Tickers and short names people use in questions. Doc metadata carries them as aliases.
ALIASES: dict[str, list[str]] = {
    "3M": ["MMM", "3M Company"],
    "AES Corporation": ["AES"],
    "AMD": ["Advanced Micro Devices"],
    "Activision Blizzard": ["ATVI", "Activision"],
    "Adobe": ["ADBE"],
    "Amazon": ["AMZN", "Amazon.com"],
    "Amcor": ["AMCR"],
    "American Express": ["AXP", "Amex"],
    "American Water Works": ["AWK", "American Water"],
    "Apple": ["AAPL"],
    "Best Buy": ["BBY"],
    "Block": ["SQ", "Square", "Block Inc"],
    "Boeing": ["BA"],
    "CVS Health": ["CVS"],
    "Coca-Cola": ["KO", "Coca Cola", "Coke"],
    "Corning": ["GLW"],
    "Costco": ["COST"],
    "FedEx": ["FDX"],
    "Foot Locker": ["FL", "Footlocker"],
    "General Mills": ["GIS"],
    "Intel": ["INTC"],
    "JPMorgan": ["JPM", "JP Morgan", "JPMorgan Chase", "JPMC"],
    "Johnson & Johnson": ["JNJ", "J&J", "JnJ", "Johnson and Johnson"],
    "Kraft Heinz": ["KHC"],
    "Lockheed Martin": ["LMT", "Lockheed"],
    "MGM Resorts": ["MGM"],
    "McDonalds": ["MCD", "McDonald's", "McDonald"],
    "Microsoft": ["MSFT"],
    "Netflix": ["NFLX"],
    "Nike": ["NKE"],
    "Oracle": ["ORCL"],
    "PG&E Corporation": ["PCG", "PG&E", "Pacific Gas and Electric"],
    "Paypal": ["PYPL", "PayPal"],
    "PepsiCo": ["PEP", "Pepsi"],
    "Pfizer": ["PFE"],
    "Salesforce": ["CRM"],
    "Ulta Beauty": ["ULTA", "Ulta"],
    "Verizon": ["VZ"],
    "Walmart": ["WMT", "Wal-Mart"],
    "eBay": ["EBAY"],
}

_PERIOD_Q = re.compile(r"_(\d{4})(Q[1-4])_")


def period_from(doc_name: str, doc_period: int | str) -> str:
    m = _PERIOD_Q.search(doc_name + "_")
    return f"{m.group(1)}{m.group(2)}" if m else str(doc_period)


def split_for(company: str) -> str:
    """Split by company so no company appears in both dev and test."""
    return "dev" if zlib.crc32(company.encode("utf-8")) % 2 == 0 else "test"


@dataclass
class Evidence:
    doc_id: str
    page_no: int  # 0-based, matching PyMuPDF page indices
    text: str


@dataclass
class Question:
    id: str
    company: str
    doc_id: str
    question: str
    answer: str
    question_type: str
    reasoning: str | None
    evidence: list[Evidence]
    justification: str | None = None

    @property
    def split(self) -> str:
        return split_for(self.company)

    @property
    def gold_pages(self) -> set[tuple[str, int]]:
        return {(e.doc_id, e.page_no) for e in self.evidence}


def load_documents(root: str | Path, only: set[str] | None = None) -> list[Document]:
    root = Path(root)
    docs: dict[str, Document] = {}
    for line in (root / "data" / "financebench_document_information.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        name = d["doc_name"]
        if only and name not in only:
            continue
        pdf = root / "pdfs" / f"{name}.pdf"
        if not pdf.exists() or name in docs:
            continue
        company = d["company"]
        docs[name] = Document(
            doc_id=name, company=company, form=FORMS.get(d["doc_type"].lower(), d["doc_type"].upper()),
            period=period_from(name, d["doc_period"]), source=str(pdf), sector=d.get("gics_sector"),
            meta={"aliases": ALIASES.get(company, []), "link": d.get("doc_link")},
        )
    return sorted(docs.values(), key=lambda d: d.doc_id)


def load_questions(root: str | Path) -> list[Question]:
    out: list[Question] = []
    path = Path(root) / "data" / "financebench_open_source.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        out.append(Question(
            id=q["financebench_id"], company=q["company"], doc_id=q["doc_name"], question=q["question"].strip(),
            answer=q["answer"], question_type=q["question_type"], reasoning=q.get("question_reasoning"),
            evidence=[Evidence(e["doc_name"], int(e["evidence_page_num"]), e.get("evidence_text", ""))
                      for e in q.get("evidence") or []],
            justification=q.get("justification"),
        ))
    return out
