"""Page-level retrieval evaluation on FinanceBench (no LLM involved)."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

from ..index.retrieve import Searcher
from .financebench import Question

KS = (1, 3, 5, 10, 20)


@dataclass
class Config:
    name: str
    mode: str  # bm25 | dense | hybrid
    scope: str  # oracle (gold filing given) | resolved | corpus (no filter)
    rerank: bool = False


def evaluate(searcher: Searcher, questions: list[Question], configs: list[Config], structure_reranker=None) -> dict:
    out: dict = {"n": len(questions), "configs": {}}
    for cfg in configs:
        searcher.rerankers = [structure_reranker] if (cfg.rerank and structure_reranker) else []
        rows = []
        started = time.perf_counter()
        for q in questions:
            kwargs: dict = {"mode": cfg.mode}
            if cfg.scope == "oracle":
                kwargs["doc_ids"] = [q.doc_id]
            elif cfg.scope == "corpus":
                kwargs["resolve"] = False
            t = time.perf_counter()
            res = searcher.search(q.question, k=max(KS), **kwargs)
            latency = (time.perf_counter() - t) * 1000
            pages = res.pages(max(KS))
            gold = q.gold_pages
            first = next((i for i, p in enumerate(pages, start=1) if p in gold), None)
            rows.append({
                "id": q.id, "type": q.question_type, "split": q.split,
                "first_rank": first,
                "doc_resolved": q.doc_id in res.resolution.doc_ids if cfg.scope == "resolved" else None,
                "doc_top1": (res.resolution.doc_ids[:1] == [q.doc_id]) if cfg.scope == "resolved" else None,
                "recall": {k: len(gold & set(pages[:k])) / len(gold) for k in KS} if gold else None,
                "latency_ms": latency,
            })
        out["configs"][cfg.name] = {"config": cfg.__dict__, "summary": summarize(rows),
                                    "wall_s": round(time.perf_counter() - started, 1), "rows": rows}
    return out


def summarize(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["recall"] is not None]
    n = len(scored)
    s = {"n": n}
    for k in KS:
        s[f"hit@{k}"] = round(sum(r["first_rank"] is not None and r["first_rank"] <= k for r in scored) / n, 4)
        s[f"recall@{k}"] = round(sum(r["recall"][k] for r in scored) / n, 4)
    s["mrr@20"] = round(sum(1 / r["first_rank"] for r in scored if r["first_rank"]) / n, 4)
    resolved = [r for r in rows if r["doc_resolved"] is not None]
    if resolved:
        s["filing_in_candidates"] = round(sum(r["doc_resolved"] for r in resolved) / len(resolved), 4)
        s["filing_top1"] = round(sum(r["doc_top1"] for r in resolved) / len(resolved), 4)
    lat = [r["latency_ms"] for r in rows]
    s["latency_p50_ms"] = round(statistics.median(lat), 1)
    s["latency_p95_ms"] = round(sorted(lat)[int(0.95 * (len(lat) - 1))], 1)
    by_type: dict[str, list] = {}
    for r in scored:
        by_type.setdefault(r["type"], []).append(r)
    s["hit@5_by_type"] = {t: round(sum(x["first_rank"] is not None and x["first_rank"] <= 5 for x in rs) / len(rs), 3)
                          for t, rs in sorted(by_type.items())}
    return s


def to_markdown(report: dict, title: str) -> str:
    lines = [f"# {title}", "", f"{report['n']} questions. Page-level: a hit means a gold evidence page "
             "(the page the FinanceBench annotator used) is among the top-k retrieved pages.", "",
             "| config | scope | hit@1 | hit@5 | hit@10 | recall@10 | MRR@20 | filing found | p50 ms |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, c in report["configs"].items():
        s = c["summary"]
        filing = f"{s['filing_in_candidates']:.0%}" if "filing_in_candidates" in s else "given"
        if c["config"]["scope"] == "corpus":
            filing = "n/a"
        lines.append(f"| {name} | {c['config']['scope']} | {s['hit@1']:.1%} | {s['hit@5']:.1%} | {s['hit@10']:.1%} | "
                     f"{s['recall@10']:.1%} | {s['mrr@20']:.3f} | {filing} | {s['latency_p50_ms']:.0f} |")
    lines.append("")
    return "\n".join(lines)
