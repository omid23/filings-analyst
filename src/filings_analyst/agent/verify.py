"""Deterministic answer checks: citations exist, and every figure is grounded.

No model is involved, so the check cannot be talked out of a failure. A
figure is grounded if it matches (allowing for rounding and units) a number on
a cited page, a calculator result or operand, or a number in the question itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .numbers import extract_numbers, is_trivial, matches
from .tools import Ledger

_CITE = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")


def cited_ids(answer: str) -> list[str]:
    ids: list[str] = []
    for m in _CITE.finditer(answer):
        for part in m.group(1).split(","):
            sid = part.strip()
            if sid not in ids:
                ids.append(sid)
    return ids


@dataclass
class Verification:
    citations: list[str] = field(default_factory=list)
    unknown_citations: list[str] = field(default_factory=list)
    numbers_checked: int = 0
    ungrounded: list[str] = field(default_factory=list)

    @property
    def has_citations(self) -> bool:
        return bool(self.citations)

    @property
    def passed(self) -> bool:
        return self.has_citations and not self.unknown_citations and not self.ungrounded

    @property
    def grounded_ratio(self) -> float:
        if not self.numbers_checked:
            return 1.0
        return 1 - len(self.ungrounded) / self.numbers_checked

    def feedback(self) -> str:
        problems = []
        if not self.has_citations:
            problems.append("the answer cites no sources; cite the passages you used as [S#]")
        if self.unknown_citations:
            problems.append(f"these citations were never retrieved: {', '.join(self.unknown_citations)}")
        if self.ungrounded:
            problems.append("these figures do not appear in any cited source or calculate result: "
                            + ", ".join(self.ungrounded)
                            + ". Cite the source that contains each one, compute it with calculate, or remove it")
        return "Verification failed: " + "; ".join(problems) + "."

    def to_dict(self) -> dict:
        return {"passed": self.passed, "citations": self.citations, "unknown_citations": self.unknown_citations,
                "numbers_checked": self.numbers_checked, "ungrounded": self.ungrounded,
                "grounded_ratio": round(self.grounded_ratio, 3)}


def verify(answer: str, ledger: Ledger, question: str = "", require_cited_source: bool = True) -> Verification:
    v = Verification()
    v.citations = cited_ids(answer)
    v.unknown_citations = [c for c in v.citations if c not in ledger.sources]
    sources = [ledger.sources[c] for c in v.citations if c in ledger.sources]
    if not require_cited_source:
        sources = list(ledger.sources.values())

    candidates: list[float] = [n.absolute for s in sources for n in extract_numbers(s.text)]
    candidates += [n.absolute for n in extract_numbers(question)]
    candidates += [c.result for c in ledger.calculations]
    # Operands of a calculation are grounded too, so an answer can show its formula ("... * 100").
    candidates += [n.absolute for c in ledger.calculations for n in extract_numbers(c.expression)]
    body = _CITE.sub(" ", answer)
    for n in extract_numbers(body):
        if is_trivial(n):
            continue
        v.numbers_checked += 1
        if not matches(n, candidates):
            v.ungrounded.append(n.text)
    return v
