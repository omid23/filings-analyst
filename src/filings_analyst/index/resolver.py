"""Map a question to the filings it is about (company, fiscal period, form).

Searching every filing in the corpus is the main failure mode of naive RAG on
financial documents: a 10-K's cash flow statement looks like every other
company's. The resolver narrows retrieval to the right company and period
before any vector or keyword search runs. Matching is deterministic and
explainable: aliases (name, ticker, short forms) plus fiscal year and quarter
patterns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..models import Document

_SUFFIXES = re.compile(
    r"\b(?:inc|incorporated|corp|corporation|company|co|plc|ltd|limited|holdings?|group|n\.?v|s\.?a)\b\.?", re.I)
_NONWORD = re.compile(r"[^a-z0-9&]+")
_FY = re.compile(r"\b(?:fy|fiscal(?:\s+year)?)\s*'?(\d{4}|\d{2})\b", re.I)
_YEAR = re.compile(r"(?:\b|(?<=q[1-4])|(?<=q[1-4]'))(19[89]\d|20[0-4]\d)\b", re.I)
_QUARTER = re.compile(r"\b(?:q([1-4])(?=\b|'|\s|\d{4})|([1-4])q\b|(first|second|third|fourth)\s+(?:fiscal\s+)?quarter)", re.I)
_ORDINAL = {"first": 1, "second": 2, "third": 3, "fourth": 4}
_FORM_HINTS = [
    ("10-K", re.compile(r"\b10-?k\b|\bannual report\b", re.I)),
    ("10-Q", re.compile(r"\b10-?q\b", re.I)),
    ("8-K", re.compile(r"\b8-?k\b", re.I)),
    ("EARNINGS", re.compile(r"\bearnings (?:release|call|report)\b|\bpress release\b", re.I)),
]


def normalize(text: str) -> str:
    return " " + _NONWORD.sub(" ", text.lower()).strip() + " "


def company_aliases(doc: Document) -> set[str]:
    names = {doc.company, _SUFFIXES.sub("", doc.company)}
    if doc.ticker:
        names.add(doc.ticker)
    names.update(doc.meta.get("aliases", []))
    out = set()
    for name in names:
        n = normalize(name).strip()
        if len(n) >= 2:
            out.add(n)
            out.add(n.replace("&", " and "))
            out.add(n.replace(" ", ""))  # "coca cola" -> "cocacola"
    return {a for a in out if a.strip()}


@dataclass
class Resolution:
    companies: list[str] = field(default_factory=list)
    years: list[int] = field(default_factory=list)
    quarter: int | None = None
    forms: list[str] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)  # best first; empty means no filter
    reason: str = ""


class FilingResolver:
    def __init__(self, documents: list[Document]):
        self.documents = documents
        self._aliases: list[tuple[str, str]] = []  # (alias, company)
        for doc in documents:
            for alias in company_aliases(doc):
                self._aliases.append((alias, doc.company))
        # Longest alias first so "american express" wins over "american"
        self._aliases = sorted(set(self._aliases), key=lambda a: -len(a[0]))

    def companies_in(self, text: str) -> list[str]:
        norm = normalize(text)
        squashed = norm.replace(" ", "")
        found: list[str] = []
        for alias, company in self._aliases:
            hit = f" {alias} " in norm or (len(alias) >= 6 and " " not in alias and alias in squashed)
            if hit and company not in found:
                found.append(company)
        return found

    @staticmethod
    def years_in(text: str) -> list[int]:
        years: list[int] = []
        for m in _FY.finditer(text):
            y = int(m.group(1))
            years.append(y + 2000 if y < 100 else y)
        for m in _YEAR.finditer(text):
            years.append(int(m.group(1)))
        return sorted(set(years))

    @staticmethod
    def quarter_in(text: str) -> int | None:
        m = _QUARTER.search(text)
        if not m:
            return None
        return int(m.group(1) or m.group(2)) if (m.group(1) or m.group(2)) else _ORDINAL[m.group(3).lower()]

    def resolve(self, question: str) -> Resolution:
        res = Resolution(companies=self.companies_in(question), years=self.years_in(question),
                         quarter=self.quarter_in(question),
                         forms=[f for f, rx in _FORM_HINTS if rx.search(question)])
        if not res.companies:
            res.reason = "no company recognized; searching all filings"
            return res
        docs = [d for d in self.documents if d.company in res.companies]

        def score(d: Document) -> float:
            s = 0.0
            fy = d.fiscal_year
            if res.years and fy in res.years:
                s += 4 + (0.5 if fy == max(res.years) else 0)  # latest year's filing holds the comparatives
            elif res.years and fy is not None and fy - 1 == max(res.years) and d.form == "10-K":
                s += 1  # next year's 10-K restates the prior year
            if res.quarter:
                s += 2 if d.period.endswith(f"Q{res.quarter}") else 0
            elif d.form == "10-K":
                s += 1  # annual questions default to the annual report
            if res.forms and d.form in res.forms:
                s += 2
            return s

        # Ties go to the most recent filing: an undated question is usually about the latest one.
        ranked = sorted(docs, key=lambda d: (score(d), d.fiscal_year or 0, d.period, d.doc_id), reverse=True)
        if res.years:
            best = score(ranked[0]) if ranked else 0
            ranked = [d for d in ranked if score(d) >= best - 1.5] or ranked
        res.doc_ids = [d.doc_id for d in ranked]
        res.reason = (f"company={', '.join(res.companies)}; years={res.years or 'any'}; "
                      f"quarter={res.quarter or 'any'}; {len(res.doc_ids)} filing(s)")
        return res
