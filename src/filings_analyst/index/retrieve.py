"""Retrieval: BM25 (FTS5), dense vectors, reciprocal rank fusion and reranking."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from ..models import Chunk, Hit
from .embed import Embedder, load_embedder
from .resolver import FilingResolver, Resolution
from .store import Store

RRF_K = 60


def rrf(rankings: dict[str, list[int]], k: int = RRF_K, weights: dict[str, float] | None = None) -> list[tuple[int, float, dict[str, int]]]:
    """Reciprocal rank fusion. Returns (chunk_id, fused score, {retriever: rank})."""
    scores: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}
    for name, ids in rankings.items():
        w = (weights or {}).get(name, 1.0)
        for rank, cid in enumerate(ids, start=1):
            scores[cid] = scores.get(cid, 0.0) + w / (k + rank)
            ranks.setdefault(cid, {})[name] = rank
    return [(cid, s, ranks[cid]) for cid, s in sorted(scores.items(), key=lambda kv: -kv[1])]


class DenseIndex:
    def __init__(self, embedder: Embedder, matrix: np.ndarray, chunk_ids: np.ndarray, doc_of: dict[int, str]):
        self.embedder, self.matrix, self.chunk_ids = embedder, matrix, chunk_ids
        self._row_doc = np.array([doc_of[int(c)] for c in chunk_ids])

    @classmethod
    def build(cls, store: Store, embedder: Embedder, directory: Path) -> DenseIndex:
        rows = store.all_chunk_rows()
        texts = [f"{h}\n{t}" for _, h, t in rows]
        embedder.fit(texts)
        matrix = embedder.embed_documents(texts).astype(np.float32)
        ids = np.array([r[0] for r in rows], dtype=np.int64)
        directory.mkdir(parents=True, exist_ok=True)
        embedder.save(directory)
        np.save(directory / "vectors.npy", matrix)
        np.save(directory / "chunk_ids.npy", ids)
        (directory / "dense.json").write_text(json.dumps({"provider": embedder.name, "dims": int(matrix.shape[1]),
                                                          "chunks": int(len(ids))}), encoding="utf-8")
        return cls(embedder, matrix, ids, store.chunk_doc_ids())

    @classmethod
    def load(cls, store: Store, directory: Path) -> DenseIndex:
        meta = json.loads((directory / "dense.json").read_text(encoding="utf-8"))
        embedder = load_embedder(meta["provider"], directory)
        return cls(embedder, np.load(directory / "vectors.npy"), np.load(directory / "chunk_ids.npy"),
                   store.chunk_doc_ids())

    def search(self, query: str, k: int = 50, doc_ids: Sequence[str] | None = None) -> list[tuple[int, float]]:
        q = self.embedder.embed_query(query)
        scores = self.matrix @ q
        if doc_ids:
            scores = np.where(np.isin(self._row_doc, list(doc_ids)), scores, -np.inf)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(self.chunk_ids[i]), float(scores[i])) for i in top if np.isfinite(scores[i])]


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]: ...


@dataclass(frozen=True)
class Route:
    """If the question matches ``when``, boost chunks whose page title or section matches ``target``."""

    name: str
    when: re.Pattern[str]
    target: re.Pattern[str]
    field: str  # "title" (financial statement name) or "section" (10-K/10-Q item)
    weight: float


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.I)


INCOME = _rx(r"statements? of (?:income|operations|earnings)|income statements?|statements? of comprehensive income")
BALANCE = _rx(r"balance sheets?|financial position")
CASHFLOW = _rx(r"cash flows?")
EQUITY = _rx(r"equity|deficit")
MDNA = _rx(r"management[’']?s discussion|^item 7\b(?!a)|^item 2\b")
BUSINESS = _rx(r"^item 1\b(?![a-c])|\bbusiness\b")
RISK = _rx(r"^item 1a\b|risk factors")
LEGAL = _rx(r"^item 3\b|legal proceedings")

# Questions name statements ("based on the cash flow statement") or imply them through the
# metric they ask about ("quick ratio" lives on the balance sheet). Narrative questions
# ("what drove...") are answered in MD&A. These routes encode that analyst knowledge.
ROUTES: tuple[Route, ...] = (
    Route("named:cash_flow", _rx(r"cash[\s-]*flows?\s+statement|statements? of cash flows?"), CASHFLOW, "title", 1.0),
    Route("named:balance_sheet", _rx(r"balance[\s-]*sheets?|statements? of financial position"), BALANCE, "title", 1.0),
    Route("named:income", _rx(r"\b(?:income|p\s*&\s*l|profit and loss|operations|earnings)\s+statements?|"
                              r"statements? of (?:income|operations|earnings)"), INCOME, "title", 1.0),
    Route("named:equity", _rx(r"statements? of (?:shareholders|stockholders)['’]? equity|equity statement"), EQUITY,
          "title", 1.0),
    Route("metric:income", _rx(r"\b(?:gross|operating|ebitda|net|profit) margins?\b|gross profit|operating income|"
                               r"effective tax rate|\beps\b|earnings per share|cost of (?:goods sold|sales|revenue)|"
                               r"\bcogs\b|interest expense|net income|\bebitda\b|interest coverage|"
                               r"\b(?:roa|roe)\b|return on (?:assets|equity)|turnover"), INCOME, "title", 0.5),
    Route("metric:balance", _rx(r"quick ratio|current ratio|working capital|liquidity|debt[\s-]to[\s-](?:equity|capital)|"
                                r"total assets|\binventor(?:y|ies)\b|receivables?|payables?|total liabilit|"
                                r"largest liabilit|book value|\b(?:roa|roe)\b|return on (?:assets|equity)|"
                                r"turnover|days (?:sales|inventory|payable)"), BALANCE, "title", 0.5),
    Route("metric:cash_flow", _rx(r"cap(?:ital)?\s*ex(?:penditures?)?|capital[\s-]intensive|free cash flow|"
                                  r"dividends? (?:paid|payout|distribution)|dividend payout|share (?:repurchase|buyback)s?|"
                                  r"buybacks?|depreciation(?: and| &) amortization|\bd&a\b|operating cash flow|"
                                  r"cash from operations|cash conversion"), CASHFLOW, "title", 0.5),
    Route("narrative:mdna", _rx(r"\bdr(?:ove|ivers?|iving)\b|\bwhy\b|\bexplain|reasons? for|primary (?:factor|reason)|"
                                r"dragged|performed|performance|\bsegments?\b|organic|guidance|outlook"),
          MDNA, "section", 0.35),
    Route("narrative:business", _rx(r"products? and services|primary customers|customer concentration|"
                                    r"geographies|operates? in|business model|competitors|employees"),
          BUSINESS, "section", 0.35),
    Route("narrative:risk", _rx(r"\brisks?\b|cyclical|seasonal|uncertaint"), RISK, "section", 0.35),
    Route("narrative:legal", _rx(r"legal|litigation|lawsuits?|proceedings|settlement"), LEGAL, "section", 0.35),
)


class SectionRouter:
    """Promote chunks from the statement or filing section the question points to.

    Rule-based, deterministic and explainable (each boost names its route).
    Routes and weights were chosen on the FinanceBench dev split only.
    """

    name = "router"

    def __init__(self, routes: tuple[Route, ...] = ROUTES, scale: float = 2.0):
        self.routes, self.scale = routes, scale

    def active(self, query: str) -> list[Route]:
        return [r for r in self.routes if r.when.search(query)]

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        routes = self.active(query)
        if not routes:
            return hits
        top = max((h.score for h in hits), default=1.0) or 1.0
        for h in hits:
            boost = 0.0
            for r in routes:
                value = (h.chunk.title if r.field == "title" else h.chunk.section) or ""
                if value and r.target.search(value):
                    boost = max(boost, r.weight)
            h.score += self.scale * boost * top
        return _resort(hits)


StructureReranker = SectionRouter  # backwards-compatible name


class CrossEncoderReranker:
    """Local cross-encoder via fastembed (downloads the model on first use)."""

    name = "cross-encoder"

    def __init__(self, model: str = "Xenova/ms-marco-MiniLM-L-6-v2", top_n: int = 30):
        self.model_name, self.top_n = model, top_n
        self._model = None

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder  # type: ignore

            self._model = TextCrossEncoder(self.model_name)
        head, tail = hits[: self.top_n], hits[self.top_n:]
        scores = list(self._model.rerank(query, [f"{h.chunk.header}\n{h.chunk.text}" for h in head]))
        for h, s in zip(head, scores, strict=True):
            h.score = float(s)
        return _resort(head) + tail


def _resort(hits: list[Hit]) -> list[Hit]:
    hits = sorted(hits, key=lambda h: -h.score)
    for i, h in enumerate(hits, start=1):
        h.rank = i
    return hits


@dataclass
class SearchResult:
    query: str
    resolution: Resolution
    hits: list[Hit]
    mode: str
    timings_ms: dict[str, float] = field(default_factory=dict)

    def pages(self, k: int | None = None) -> list[tuple[str, int]]:
        """Unique (doc_id, page_no) in rank order."""
        seen: list[tuple[str, int]] = []
        for h in self.hits:
            key = (h.chunk.doc_id, h.chunk.page_no)
            if key not in seen:
                seen.append(key)
            if k and len(seen) >= k:
                break
        return seen


class Searcher:
    """Resolve the filing, retrieve with BM25 and/or dense vectors, fuse, rerank."""

    def __init__(self, store: Store, dense: DenseIndex | None = None, rerankers: Sequence[Reranker] = (),
                 resolver: FilingResolver | None = None, weights: dict[str, float] | None = None):
        self.store, self.dense, self.rerankers = store, dense, list(rerankers)
        self.weights = weights or {"bm25": 1.0, "dense": 0.5}  # chosen on the FinanceBench dev split
        self.resolver = resolver or FilingResolver(store.documents())

    def search(self, query: str, k: int = 10, *, mode: str = "hybrid", doc_ids: Sequence[str] | None = None,
               resolve: bool = True, candidates: int = 100) -> SearchResult:
        import time

        t0 = time.perf_counter()
        resolution = self.resolver.resolve(query) if resolve and doc_ids is None else Resolution()
        filt = list(doc_ids) if doc_ids else (resolution.doc_ids or None)
        timings = {"resolve": (time.perf_counter() - t0) * 1000}

        rankings: dict[str, list[int]] = {}
        raw: dict[str, dict[int, float]] = {}
        if mode in ("bm25", "hybrid"):
            t = time.perf_counter()
            res = self.store.bm25(query, candidates, filt)
            rankings["bm25"] = [c for c, _ in res]
            raw["bm25"] = dict(res)
            timings["bm25"] = (time.perf_counter() - t) * 1000
        if mode in ("dense", "hybrid"):
            if self.dense is None:
                if mode == "dense":
                    raise ValueError("no dense index built; run `filings-analyst index --embedder lsa`")
            else:
                t = time.perf_counter()
                res = self.dense.search(query, candidates, filt)
                rankings["dense"] = [c for c, _ in res]
                raw["dense"] = dict(res)
                timings["dense"] = (time.perf_counter() - t) * 1000

        if len(rankings) == 1:
            name, ids = next(iter(rankings.items()))
            fused = [(cid, raw[name][cid], {name: i}) for i, cid in enumerate(ids, start=1)]
        else:
            fused = rrf(rankings, weights=self.weights)
        chunks: dict[int, Chunk] = self.store.chunks(c for c, _, _ in fused)
        hits = [Hit(chunks[c], s, i, src) for i, (c, s, src) in enumerate(fused, start=1) if c in chunks]
        t = time.perf_counter()
        for rr in self.rerankers:
            hits = rr.rerank(query, hits)
        timings["rerank"] = (time.perf_counter() - t) * 1000
        return SearchResult(query, resolution, hits[: max(k, 1) * 5], mode, timings)
