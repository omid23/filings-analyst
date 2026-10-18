"""End-to-end answer evaluation on FinanceBench.

Scores per question:

* ``numeric``: for questions whose gold answer is a number, does the agent's
  answer contain that number (rounding- and unit-aware, 1% tolerance)?
* ``judge``: optional LLM-as-judge verdict (correct / incorrect / refusal) using
  the gold answer and justification, for narrative questions.
* ``grounded``: the deterministic verifier passed (citations exist, every figure
  traceable to a cited page or calculation).
* ``cited_gold_page``: at least one cited page is a page the annotator used.
"""

from __future__ import annotations

import json
import re
import statistics
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

from ..agent.analyst import Analyst, AnalystResult
from ..agent.llm import LLM, LLMError
from ..agent.numbers import extract_numbers, is_trivial
from .financebench import Question

JUDGE_PROMPT = """You grade answers to financial questions against a reference answer written by an analyst.
Reply with exactly one word:
correct - the answer reaches the same conclusion as the reference; numbers match within normal rounding
incorrect - the answer contradicts the reference, has wrong numbers, or misses the main point
refusal - the answer declines or says the information is unavailable"""


@dataclass
class AnswerRow:
    id: str
    type: str
    split: str
    answer: str
    gold: str
    numeric: bool | None
    judge: str | None
    grounded: bool
    grounded_ratio: float
    cited_gold_page: bool
    steps: int
    input_tokens: int
    output_tokens: int
    latency_ms: float
    stopped: str


def gold_number(gold: str):
    nums = [n for n in extract_numbers(gold) if not is_trivial(n) or n.decimals or n.percent]
    return nums[0] if nums else None


def numeric_correct(answer: str, gold: str, rel_tol: float = 0.01) -> bool | None:
    g = gold_number(gold)
    if g is None:
        return None
    target = g.absolute
    for n in extract_numbers(answer):
        for k in (1.0, 1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9, 100.0, 0.01):
            v = abs(n.absolute * k)
            if target == 0:
                if v == 0:
                    return True
                continue
            if abs(v - abs(target)) / abs(target) <= rel_tol:
                return True
    return False


def judge(llm: LLM, q: Question, answer: str) -> str:
    content = (f"Question: {q.question}\n\nReference answer: {q.answer}\n"
               f"Reference justification: {q.justification or 'n/a'}\n\nAnswer to grade: {answer}")
    try:
        resp = llm.complete(JUDGE_PROMPT, [{"role": "user", "content": content}], [])
    except LLMError:
        return "error"
    word = re.findall(r"[a-z]+", resp.text.lower())
    return word[0] if word and word[0] in ("correct", "incorrect", "refusal") else "unparsed"


def score(q: Question, r: AnalystResult, judge_llm: LLM | None) -> AnswerRow:
    cited_pages = {(s.doc_id, s.page_no) for sid in r.verification.citations if (s := r.ledger.sources.get(sid))}
    return AnswerRow(
        id=q.id, type=q.question_type, split=q.split, answer=r.answer, gold=q.answer,
        numeric=numeric_correct(r.answer, q.answer),
        judge=judge(judge_llm, q, r.answer) if judge_llm else None,
        grounded=r.verification.passed, grounded_ratio=r.verification.grounded_ratio,
        cited_gold_page=bool(cited_pages & q.gold_pages), steps=len(r.steps),
        input_tokens=r.input_tokens, output_tokens=r.output_tokens, latency_ms=r.latency_ms, stopped=r.stopped,
    )


def run(questions: list[Question], make_analyst: Callable[[], Analyst], *, judge_llm: LLM | None = None,
        oracle_filing: bool = False, workers: int = 4, price_per_mtok: tuple[float, float] = (0.0, 0.0)) -> dict:
    def one(q: Question) -> AnswerRow:
        analyst = make_analyst()
        result = analyst.run(q.question, doc_ids=[q.doc_id] if oracle_filing else None)
        return score(q, result, judge_llm)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        rows = list(pool.map(one, questions))
    return {"n": len(rows), "oracle_filing": oracle_filing, "summary": summarize(rows, price_per_mtok),
            "rows": [asdict(r) for r in rows]}


def summarize(rows: list[AnswerRow], price_per_mtok: tuple[float, float] = (0.0, 0.0)) -> dict:
    n = len(rows) or 1
    numeric = [r for r in rows if r.numeric is not None]
    judged = [r for r in rows if r.judge in ("correct", "incorrect", "refusal")]
    s = {
        "n": len(rows),
        "numeric_n": len(numeric),
        "numeric_accuracy": round(sum(r.numeric for r in numeric) / len(numeric), 4) if numeric else None,
        "judge_n": len(judged),
        "judge_correct": round(sum(r.judge == "correct" for r in judged) / len(judged), 4) if judged else None,
        "judge_refusal": round(sum(r.judge == "refusal" for r in judged) / len(judged), 4) if judged else None,
        "grounded": round(sum(r.grounded for r in rows) / n, 4),
        "cited_gold_page": round(sum(r.cited_gold_page for r in rows) / n, 4),
        "avg_steps": round(statistics.mean(r.steps for r in rows), 2) if rows else 0,
        "latency_p50_s": round(statistics.median(r.latency_ms for r in rows) / 1000, 2) if rows else 0,
        "cost_per_question_usd": round(sum(r.input_tokens * price_per_mtok[0] + r.output_tokens * price_per_mtok[1]
                                           for r in rows) / 1e6 / n, 5),
    }
    return s


def to_markdown(report: dict, title: str) -> str:
    s = report["summary"]

    def pct(x):
        return "n/a" if x is None else f"{x:.1%}"

    return "\n".join([
        f"# {title}", "",
        f"{s['n']} questions · filing {'given' if report['oracle_filing'] else 'resolved by the agent'}", "",
        "| metric | value |", "|---|---:|",
        f"| numeric answers correct ({s['numeric_n']}) | {pct(s['numeric_accuracy'])} |",
        f"| judged correct ({s['judge_n']}) | {pct(s['judge_correct'])} |",
        f"| judged refusal | {pct(s['judge_refusal'])} |",
        f"| passed verification (cited + grounded) | {pct(s['grounded'])} |",
        f"| cited an annotator's evidence page | {pct(s['cited_gold_page'])} |",
        f"| avg steps | {s['avg_steps']} |",
        f"| p50 latency | {s['latency_p50_s']} s |",
        f"| cost per question | ${s['cost_per_question_usd']:.4f} |", "",
    ])


def dump(report: dict, path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
